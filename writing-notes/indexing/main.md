---
layout: post
title: "Indexing in RAG pipeline - Part I"
---



# The Architecture of Speed: A Deep Dive into Indexing, RAG, and the Limits of Search Efficiency

## Introduction
In this article, I deeply explore indexing in **retrieval** systems. Amidst the current hype surrounding agentic AI, retrieval systems operating on specialized data remain the leading AI application in the industry—and will continue to be. Inherently, this means dealing with **millions** of messy data points, searching through them, and finding the most relevant results as fast as possible. If you want to operate *fast* on *big* data, you must use **indexing**. 

The name suggests the core idea: assigning **indexes** to data to facilitate search. Without an index, every search degrades into a brute-force scan. In traditional databases, this means checking every row linearly. In modern vector spaces, it means calculating the distance between a query vector and every single embedded document in your dataset. 

Indexing is the invisible backbone of high-performance data systems—the architectural bridge between storing information and retrieving it at low latency. As data scales to millions of records, unindexed systems inevitably hit a wall where real-time retrieval becomes mathematically impossible.

You have likely heard the term **RAG** (Retrieval-Augmented Generation). It is simply a retrieval system endowed with generative AI. One lesson from the past is that as time goes by, existing methods must adapt to new use cases. With the rise of LLMs and their growing acceptance in everyday workflows, researchers found that traditional indexing methods were not naturally suited to these new applications. The nature and scale of the data have changed; modern systems must handle text, images, audio, and sometimes combinations of these modalities.

Many earlier indexing methods were designed mainly for structured data or **exact** search, whereas LLM-based applications often require **semantic** search over high-dimensional vector representations. To build scalable RAG pipelines, we must bridge the gap between classical structural indexing and modern semantic indexing. 

However, an engineering trap awaits: no matter how computationally efficient or mathematically brilliant your indexing algorithm is, it is entirely at the mercy of your data preparation. If your tokenization is flawed or your embedding models fail to capture semantic nuance, you are simply accelerating the retrieval of irrelevant data.

This guide provides a comprehensive, mathematically grounded, and engineering-focused deep dive into indexing paradigms, tracking their evolution from classic databases to modern RAG architectures.

### What This Guide Covers 

I **Part I** we mainly cover
*   **The Fundamentals:** An abstract look at how indexes trade space complexity for time complexity, and the exact cost of running an unindexed system.
*   **The Traditional vs. Vector Divide:** Disentangling exact, deterministic lookups (B-Trees, Hash maps) from probabilistic, high-dimensional **Approximate Nearest Neighbor (ANN)** search.
*   **Core Algorithmic Archetypes:**
    *   **Inverted File Indexing (IVF):** Voronoi partitioning, $k$-means quantization, and inverted lists. 

Then in **Part II** we continue to disucss on 

*   **Graph & Quantization Variants:** The mechanics of HNSW (Hierarchical Navigable Small World) and Product Quantization (PQ).
*   **The "Garbage In, Garbage Out" Trap:** Why indexing cannot save bad tokenization, poor chunking strategies, or misaligned embedding models.
*   **The "No-Index" Regime:** When exhaustive brute-force search is actually superior to building an index.

## 1. The Foundational Mechanics: Why Indexing Matters

The most noamal and common question is: **What happens if we do not index our data?**

To answer this question, let's start with a metaphor. Imagine you have a massive library filled with books on math, physics, chemistry, sports, and various other topics.

Suppose you place all these books onto your bookshelves completely at random, without any arrangement. This initial setup is incredibly simple. Because you don't have to worry about order or placement, throwing the books onto the shelves takes the least amount of time and energy.

But the real problem starts when I ask you to find a specific book. To find it, you have to walk through the library and check the books one by one. This approach works fine if you only have a few dozen books, but it becomes an absolute nightmare if your library grows to several million.

Now, let's go a step further. Suppose we group the books by subject, placing all books with a common topic next to each other. However, we don't know which shelves hold which subjects. If I ask you for a specific book, you would first consider its subject, and then look for the shelf where that subject is located.

To find the correct shelf, you have to check the first book of every single aisle. If that first book matches your subject, you stop and search that shelf; otherwise, you skip it and move to the next. This is significantly more efficient than the first method because once you find the right shelf, you can skip all the others. However, you still waste a lot of time searching for the shelf itself.

Now, let's introduce a master directory or clear aisle labels—which acts exactly like a database index. If you not only group common subjects together, but also create a master map showing exactly where each specific subject lives, the search changes completely. For example, the map tells you that math books are always on the first floor, on the left. When you are asked to find a math book, you don't wander or guess; you immediately go straight to that specific shelf. And since all the math books are consolidated right there, your search is narrowed down to just a tiny fraction of the library.

With this example in mind let's talk about indexing in a technical way. When you write data to a raw storage medium without an index, it is typically appended to a log or organized in a heap structure. This makes writes incredibly cheap ($O(1)$), but it turns reads into an expensive computational nightmare.

### The Mechanics of the "No-Index" Regime

Without an index, any lookup requires a sequential, exhaustive scan. The system must load every record from storage into memory and evaluate it against the query predicate.

* **In Traditional Databases:** In this case each data is a point (has $1$ dimension). A point lookup on an unindexed column forces a Full Table Scan. The time complexity scales strictly linearly:

$$O(N)$$

where $N$ is the number of rows.

* **In Vector Spaces (RAG Pipelines):** Each data is a vector with domesnion $d$. Searching an unindexed vector space requires an exhaustive flat search. To find the nearest neighbor, the system must compute the distance (e.g., cosine or Euclidean) between the query vector and every stored vector. The time complexity scales as:

$$O(N \cdot d)$$

where $N$ is the number of vectors and $d$ is the dimensionality of the vector space (typically $d = 768$ or $d = 1536$ in modern embedding models).

For a dataset of 10 million vectors with 1,536 dimensions, single query requires
*   **10 million floating-point operations (FLOPs)** in traditional case. 
*   **15.3 billion FLOPs** in vector spaces.

At scale, running an unindexed system converts real-time retrieval into a mathematical and financial challenge.

### How Indexing Works in the Abstract

An index works by organizing data into deterministic or probabilistic structural maps *before* the query arrives. Instead of looking at the data itself, the query engine traverses the index structure to isolate a tiny fraction of the total dataset.

```
[Raw Appended Data] ---> [Index Construction] ---> [Structured Layout]
                                                          │
   Query Process:                                         ▼
   User Query ─────────> [Traverse Index Map] ────> [Target Sub-segment]
                         (Skips 99% of data)        (Instant Retrieval)

```

By imposing geometric, hierarchical, or mathematical order onto the data during ingestion, the index allows the query processor to discard the vast majority of the search space immediately. This shifts the runtime boundary from linear ($O(N)$) down to logarithmic ($O(\log N)$) or near-constant ($O(1)$) complexities. Exactly like knowing the math book is always on the top shelf on the left in the metaphor. 


## 2. Relational Databases vs. Vector RAG Pipelines

In this section, we will look at what has changed regarding indexing since the rise of generative AI, especially LLMs. For traditional use cases—which still make up the vast majority of software applications—we continue to use the same highly efficient indexing methods we always have. However, as mentioned earlier, these legacy indexing methods are not quite aligned with new paradigms like semantic search.

Despite this shift, the two main goals of indexing remain unchanged, regardless of the use case: **speed** and **scalability**. You want to find whatever you need as *fast* as possible, and you want to do it across *all* the data you have.

While the macro goal of indexing remains uniform, the underlying math and engineering split into two completely different paradigms when moving from traditional databases to RAG pipelines:

* **Deterministic (Exact)**
* **Probabilistic (Semantic)**

### Exact Match and Routing

Traditional database indexing relies entirely on **exact match** and **deterministic routing**. Suppose you query a database with `WHERE user_id = 49201` (an example from SQL, or Structured Query Language, which is highly effective for managing relational tables in databases like MySQL or PostgreSQL). The system's answer is strictly binary: the record matches, or it does not.

To understand how this works, suppose you are using your university's library. If you want to borrow a book, you *must* know the exact title, the author's name, or the specific ISBN code (the index). When you ask the librarian to bring you that exact book, they look at the code and search the library's catalog based purely on that index. You **cannot** ask this traditional librarian for "a book about the common ways of organizing data," because they do not know how to search for concepts without exact, character-for-character information.

Behind the scenes, this "librarian" is powered by three highly optimized data structures:

1. **B-Trees:** Like a decision tree, the database asks "is the ID higher or lower?" at each branch until it finds the exact record.
2. **Hash Indexes:** A mathematical algorithm that acts like a coat check, converting a specific key (like a user ID) into a direct, exact memory address.
3. **Inverted Indexes:** The engine behind traditional text search, which acts like the glossary at the back of a textbook, mapping an exact keyword directly to a list of documents that contain it.



### **The Two Phases of Indexing**

To understand why this exact-match system effortlessly handles massive scale—like Amazon or Google searching billions of items before the AI era—it helps to break indexing down into two distinct steps:

1. **Indexing the Document (Creation):** When new data enters the database, the system extracts exact attributes (keywords, IDs) and files them into a B-Tree or Inverted Index. This requires upfront work, but it maps the data perfectly.
2. **Indexing in the Search Phase (Retrieval):** When a user types a query, the system doesn't scan millions of documents. It takes the exact keyword or ID, checks the pre-built index, and follows the pointer directly to the item.

Because the search phase doesn't read the documents, big data isn't a problem. These data structures operate on logarithmic time complexity, expressed mathematically as $O(\log n)$. Practically, this means if a database grows from one million to one billion records, finding an exact match doesn't take a thousand times longer—it only takes a few extra computational steps, because the system skips half the remaining data with every single step. Furthermore, comparing exact strings (`"Apple"` vs `"Apple"`) requires almost zero computational overhead.


Traditional indexing is undeniably fast, infinitely scalable, and perfectly suited for exact searches. However, what happens if there is no exact match for your query? what happens if you onyl know a partial  information of  you query? Let's talk about searching for words and sentences.  An exact-match index only knows what a word *is*, not what a word *means*. If you search a traditional database for "puppy," it will instantly return every document containing the word "puppy." But it will completely ignore a document about a "young dog," because those exact letters do not match. 

To move forward, we must transition from finding the **exact** item to finding the **closest** item. More professionally, vector indexing in RAG pipelines operates in the realm of **Approximate Nearest Neighbor (ANN)** search.

However, when we talk about concepts like "close," "near," "far," or "similar," we must take into account **dimensionality** and the **concept of distance**. In the exact-match paradigm, the world is binary: either there is a match, or there isn't. In the non-exact paradigm, we are hunting for the best options that reside closest to our query. If you have closely followed the indexing concept so far, you might notice that this phase doesn't actually involve indexing at all yet. Instead, it relies entirely on **how** you represent your documents and **how** you calculate their similarity.

This representation step is called **embedding**, which we will discuss in deep detail in a future article. For now, it is enough to know that embedding transforms each data point—whether it is a word, a sentence, or an entire paragraph—into a vector with dimension $d$. When a user submits a query, the system transforms that query into a vector too, and compares it against all the other vectors to find the closest matches.

While this approach beautifully solves the semantic search problem, it comes at a high price. In terms of latency and computation, it is far more expensive than traditional exact search. If you have millions of data points, comparing a new query vector against every single vector in your database is incredibly inefficient. We need a way to **skip** the vast majority of irrelevant data. This is exactly **where the indexing step happens**.

To understand this more easily, imagine that you go to a massive bookstore to buy a book. You don't know the exact name of the book, but you tell the bookseller that you are looking for something related to the subject of mathematics.

When you describe what you want, the seller's ability to help depends entirely on their mental understanding of the subject and how they have framed the store's inventory in their mind. Two distinct steps happen here:

* **Understanding the Concept:** The bookseller must grasp the meaning of your request and mentally map it to the subjects in the store. They must realize that a book on geometry or calculus fits your needs, even if the word "mathematics" isn't on the cover. This conceptual mapping is the embedding.

* **Knowing Where to Look:** Once the bookseller knows what concepts to look for, they need to know exactly which aisles and shelves hold those subjects so they can walk straight there, skipping the fiction and cooking sections entirely. This structural organization that allows them to skip irrelevant books is the indexing.


In the rest of this article, we will explore the most common ways to index data for non-exact semantic search and walk through sample code showing how to leverage these methods in practice.


| Dimension              | Traditional Database Indexing           | Modern Vector (RAG) Indexing                             |
| ------------------------| -----------------------------------------| ----------------------------------------------------------|
| **Core Paradigm**      | Deterministic / Exact Match             | Probabilistic / Approximate Nearest Neighbor (ANN)       |
| **Primary Structures** | B-Trees, B+ Trees, LSM-Trees, Hash Maps | IVF, HNSW, ScaNN, DiskANN                                |
| **Search Space**       | 1D scalar or structured multi-column    | High-dimensional dense vectors ($d = 512$ to $d = 3072$) |
| **Target Complexity**  | $O(\log N)$ or $O(1)$                   | $O(\log N)$ or $O(\sqrt{N})$ approximately               |
| **Output**             | Exact record match                      | Ranked list of semantically similar vectors              |


## 3. Algorithmic Archetypes: Tree, IVF, and Modern Vector Methods

In this section we will mention most common indexing algorithms used in modern database libraries such as `FAISS` , `LlamaIndex` and `ChromaDB`. These are open-source libraries and have a good compatibility with RAG and Agentic frameworks. 

For understanding this section no prior knowledge about embedding models needed. The only point that needs to be considered is that every input after passing throught the embedding model is transformed into a vector of dimension $d$. 

Although the first method `Flat Indexing` is not a *real* indexing method it is just comparison between all other vectors with Eucledean metric. But in the AI community it is often mentioned as a way of indexing.

### 1. Flat Indexing (Brute Force)

Before we jump into all the fancy approximate search methods (ANN), we have to establish our exact-match baseline: the Flat Index. In a Flat Index, we just store the vectors exactly as they are generated. No structural organization, no compression, just raw data. When a query vector $q$ arrives, the system does an exhaustive (through all vectors of the database) search. Basically we are done with indexing. The indexing step is done at this stage, **however** the interesting part is **how** the search is done. 

In general the system calculates the mathematical distance between $q$ and every single document vector $x$ in the entire dataset $X$ of size $N$. Because we are comparing every single dimension ($d$) of every single vector ($N$), the time complexity is $O(N \cdot d)$. It guarantees 100% perfect recall, but as our dataset grows, this brute-force approach becomes computationally unappealing to the marketing team.

In the following we will mention some of these metrics. Again these (metrics) happen **after** indexing. 

Throughout this section  $q$ is a query vector and $x$ is a sample point in our dataset. Both $q$ and $x$ can be represented as: 

$$q = (q_1,q_2,\cdots,q_d)$$

and 

$$x = (x_1,x_2,\cdots,x_d)$$

for $ q_1,q_2,\cdots,q_d, x_1,x_2,\cdots,x_d \in \mathbb{R}$. And for every vector $x$ the $L2$ norm of $x$ is defined as:

$$  \Vert x\Vert_2 =  \sqrt{\sum_{j=1}^{d} x_{j}^2} $$

#### **Cosine (aka Cosine Similarity) and Inner Product** 

Between $q$ and $x$ the  **cosine similarity** for 

$$S_C(q, x) = \frac{q \cdot x}{\Vert q\Vert \Vert x\Vert} = \frac{\sum_{j=1}^{d} q_j x_{j}}{\sqrt{\sum_{j=1}^{d} q_j^2} \sqrt{\sum_{j=1}^{d} x_{j}^2}}$$

However, in modern RAG architectures, we almost always prefer the **Inner Product** (Dot Product) instead. It’s simply the unnormalized projection of one vector onto another:

$$IP(q, x) = q \cdot x = \sum_{j=1}^{d} q_j x_{j}$$

Why the shift to Inner Product? It comes down to pure hardware efficiency. If we $L2$-normalize (divide each vector by its $L2$ norm) our vectors before indexing them, the denominators in the Cosine Similarity equation become $1$. This makes Cosine Similarity and Inner Product mathematically equivalent. By stripping out the square roots and division, the Inner Product drastically reduces the CPU/GPU cycles needed during a massive brute-force scan while giving us the exact same ranking.

Here we would like to mention that in pure mathematics, a "metric" (or distance function) on an $n$-dimensional space $\mathbb{R}^n$ is a specific function $d: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ that *must* satisfy three unbending rules:

1. **Non-negativity:** The distance between two points is always $\ge 0$, and the distance is exactly $0$ if and only if the two points are identical. ($d(x, y) = 0 \iff x = y$).
2. **Symmetry:** The distance from $x$ to $y$ is the exact same as $y$ to $x$. ($d(x, y) = d(y, x)$).
3. **The Triangle Inequality:** The direct path from $x$ to $z$ is always shorter than or equal to going from $x$ to $y$, and then $y$ to $z$. ($d(x, z) \le d(x, y) + d(y, z)$).

As engineers, we accept this jargon, but strictly mathematically speaking, **this is heavily misused.**  

**Inner product and cosine Similarity completely fail these mathematical tests.** They are *similarity comparisons*, not actual distances. Inner product can yield negative numbers, failing rule #1. cosine similarity increases the closer two vectors get, which is the exact opposite of a distance function. Even if you invert it into $1 -$ cosine Similarity, it still famously violates the Triangle Inequality (For more information of how we can create a distance from inner product you can refer to my article on positional encodding [here](https://erfansoheil.github.io/writing-notes/positional_embeddings/main.html)).

If you read the documentation for FAISS, Milvus, or ChromaDB, you will constantly see the word "metric" used to describe how vectors are compared (e.g., `metric_type="IP"`). So, when vector databases talk about "metrics," remember that they are using it as a sloppy catch-all term for **scoring functions**.


#### **The $L_p$ Metric Family (Minkowski Distances)**

If Cosine Similarity and Inner Product are just "scoring functions," what does a *real* distance metric look like? Enter the Minkowski distance, denoted as the $L_p$ metric (where $p \ge 1$).

Minkowski isn't just one metric; it is a mathematical generalization of geometric distance in $d$-dimensional space ($\mathbb{R}^d$). It is defined as:

$$D_p(q, x) = \left( \sum_{j=1}^{d} \vert q_j - x_{j} \vert^p \right)^{\frac{1}{p}}$$

Unlike Inner Product, the $L_p$ family satisfies all three strict mathematical rules of a metric (Non-negativity, Symmetry, and the Triangle Inequality). By simply changing the value of the parameter $p$, we completely alter the "geometry" of our vector space and how the database ranks neighbors.

To truly understand how this parameter behaves, we have to look at the "Unit Ball" — the shape created by plotting every possible point that is exactly a distance of $1$ away from the origin ($R=1$).

Play with the slider in the interactive visualization below to see how changing $p$ physically morphs our definition of distance in both 2D and 3D space:

As you can see from the visualization, depending on the $p$ value, the "shape" of our search radius drastically changes. Here is how the most common variations perform in practice:

**Euclidean Distance - $p=2$ ($L_2$ Norm)**
This is the standard "straight-line" distance you learned in high school geometry. In the visualization, $p=2$ yields a perfect circle (2D) or sphere (3D).

$$D_{L2}(q, x) = \sqrt{\sum_{j=1}^{d} (q_j - x_{j})^2}$$

While $L_2$ is the default distance metric in many libraries, the squaring mechanism makes it highly sensitive to outliers. A massive difference between vectors in just *one* latent dimension will blow up the entire distance score, potentially pushing a highly relevant document to the bottom of the search results.

**Manhattan Distance - $p=1$ ($L_1$ Norm)**
Instead of a straight line, $L_1$ calculates distance as a grid-like path—the sum of absolute differences. In the visualization, $p=1$ creates a rigid diamond (2D) or octahedron (3D).

$$D_{L1}(q, x) = \sum_{j=1}^{d} \vert q_j - x_{j} \vert$$

$L_1$ is especially powerful for highly sparse vector representations (like TF-IDF or SPLADE).

**Chebyshev Distance ($L_\infty$ Norm)**
If we push $p$ all the way to infinity, the shape morphs into a perfect square (2D) or cube (3D). Mathematically, the metric stops summing differences and looks *only* at the single maximum difference across all dimensions:

$$D_{\infty}(q, x) = \lim_{p \to \infty} \left( \sum_{j=1}^{d} \vert q_j - x_{j} \vert^p \right)^{\frac{1}{p}} = \max_{j} \vert q_j - x_{j} \vert$$

Think of $L_\infty$ as the ultimate strict bounding box. If you want a query to completely reject a document just because it drastically fails on *one* specific latent feature—even if the other 1535 features are a perfect match—$L_\infty$ is the exact tool for the job.

<iframe 
    src="/writing-notes/indexing/asset/lp_weight.html" 
    width="100%" 
    height="600px" 
    style="border: 1px solid #ddd; border-radius: 8px; overflow: hidden;" 
    scrolling="no">
</iframe>



**Jaccard Distance for Sparse Vectors**

We usually think of Jaccard distance as a way to measure the overlap of sets, but we can adapt it for continuous vectors using the Ruzicka (or MinMax) formulation:

$$D_J(q, x) = 1 - \frac{\sum_{j=1}^{d} \min(q_j, x_{j})}{\sum_{j=1}^{d} \max(q_j, x_{j})}$$

While we don't use this for dense embeddings like BERT, it becomes incredibly powerful when we start using Sparse Retrieval (like SPLADE or BM25). In sparse spaces, our vectors represent token vocabularies where 99% of the dimensions are zero. Jaccard is perfect here because it zeroes in on the exact overlap of activated tokens without being heavily skewed by the sheer volume of mutual zeros.


#### **The Vector Database Reality: Native Support vs. Custom Code**

When you move from mathematical theory into production tools like FAISS, Milvus, ChromaDB, LlamaIndex, or Qdrant, you quickly realize that not all distance metrics are treated equally. Vector databases rely on extreme hardware optimization (C++ routines, SIMD instructions, and GPU kernels) to make searches fast. Because of this, they are highly opinionated about which metrics they actually let you use.

We can break these down into three tiers of production readiness:

**Tier 1: The Universal Defaults (Native Everywhere)**
If you are using **inner Product (IP), cosine Similarity, or Euclidean Distance ($L_2$)**, you are in the safe zone. Every major vector database supports these natively out-of-the-box. Their underlying graph algorithms (like HNSW) and quantization techniques are heavily optimized for these three specific calculations.

**Tier 2: The Conditional Natives ($L_1$ and Jaccard)**
Metrics like Manhattan ($L_1$) and Jaccard are supported natively, but with major asterisks attached to them:

* **Manhattan ($L_1$):** Qdrant and FAISS support $L_1$ natively for dense floating-point vectors. However, if you are using ChromaDB or Pinecone, $L_1$ is simply not exposed in their standard APIs.
* **Jaccard Distance:** If you try to run Jaccard on standard dense embeddings, it will fail. Databases like Milvus and FAISS natively support Jaccard, but *only for Binary or Sparse vectors*. The hardware executes Jaccard using fast bitwise operations (like AND/OR logic gates) rather than floating-point math.

**Tier 3: The "Custom Code" Territory ($L_3$ and $L_\infty$)**
If you want to use $L_3$ or Chebyshev ($L_\infty$), you are essentially stepping off the paved road. Modern vector databases do not support $L_3$ or $L_\infty$ natively.

More importantly, **you cannot simply write a custom Python function to replace them.** Because approximate search indexes like HNSW are written in compiled C++ or Rust for speed, injecting a custom Python distance function into the loop would destroy the database's performance.

If you absolutely must use $L_\infty$ or $L_3$, you have two choices:

1. **The Brute-Force Route:** Skip the database's built-in HNSW index entirely, pull the vectors into memory, and write a custom Numpy/PyTorch script to do a Flat (brute-force) scan.
2. **The Hardcore Route:** Fork the underlying open-source C++ library (like `hnswlib` or FAISS), write your custom $L_\infty$ metric in C++, recompile the library, and bind it back to your Python environment.


#### **Summary Checklist for RAG Indexing**

| Metric                | Mathematical Focus        | Database Support                                                     |
| -----------------------| ---------------------------| ----------------------------------------------------------------------|
| **Inner Product**     | Unnormalized projection   | Tier 1 (Universal). Fastest to compute.                              |
| **Cosine**            | Pure orientation / angle  | Tier 1 (Universal). Often mapped to IP via L2-normalization.         |
| **$L_2$ (Euclidean)** | Straight-line geometry    | Tier 1 (Universal). Default for metric-space embeddings.             |
| **$L_1$ (Manhattan)** | Axis-aligned differences  | Tier 2 (Conditional). Great for high dimensions; limited DB support. |
| **Jaccard**           | Intersection over Union   | Tier 2 (Conditional). Restricted to binary/sparse vectors only.      |
| **$L_\infty$**        | Maximum single divergence | Tier 3 (Custom). Requires writing your own brute-force or C++ code.  |



### **2. Inverted File Indexing (IVF)**

Go back to the library metaphor from Section 1 for a second. The moment you stopped scattering books randomly and started grouping them by subject — math here, physics there — with a master map telling you which aisle holds which subject, you had already invented IVF. Inverted File Indexing is that idea, made mathematical: instead of one giant undifferentiated pile of vectors, you carve the space into regions, and you only search the regions that could plausibly contain your answer.

IVF avoids comparing the query with every vector in the dataset. It first divides the vector space into clusters. During search, it identifies the clusters closest to the query and compares the query only with the vectors stored in those clusters.


Each cluster is represented by a centroid. Assigning every vector to its nearest centroid divides the space into regions known as Voronoi cells.

```text
          Voronoi Cells (Clustered Vector Space)
          ┌─────────────────┬─────────────────┐
          │     •    •      │       •         │
          │   •   c1 (Centroid)  •   c2       │
          │     •    •      │    •     •      │
          ├─────────────────┼─────────────────┤
          │       •         │      •   •      │
          │   •  c3         │   •   c4  •     │
          │    •    •       │      •   •      │
          └─────────────────┴─────────────────┘

```

The mechanism has three distinct phases, and it's worth being precise about which phase does which job — a lot of confusion about IVF comes from conflating "building the index" with "searching the index," when they are mathematically separate operations.

1. **Training Phase:** The system runs a $k$-means clustering algorithm on a representative sample of the dataset to partition it into $k$ clusters, determining a set of centroids $C = \{c_1, c_2, \dots, c_k\}$. Suppose that you have $N$ data points (vectors). Running $k$-means on all $N$ vectors can be expensive because every iteration requires assigning each training vector to one of the k centroids. For a sufficiently large and representative dataset, a random sample often contains enough information about the main regions of the vector distribution. The centroids learned from this sample can therefore approximate the centroids that would be obtained from the full dataset, while requiring much less computation.

However, the sample must represent the original distribution. If rare document types or small semantic regions are missing from the sample, the resulting centroids may not allocate suitable clusters to them. This can create unbalanced inverted lists and reduce recall.

2. **Ingestion Phase:** Every incoming vector $x$ is mapped to its nearest centroid $c_i$ such that the distance $D(x, c_i)$ is minimized. The index stores this as an **inverted list**—a mapping of Centroid ID $\rightarrow$ List of Vector IDs. Mathematically, it places the vector in a Voronoi cell $V_i$:

$$V_i = \{ x \in X \mid D(x, c_i) \le D(x, c_j) \text{ for all } j \neq i \}$$

After assigning the vector, IVF stores it inside an inverted list associated with that centroid. The structure can be represented as:
```text
Centroid ID → IDs of vectors assigned to that centroid
```

$$
c1 \rightarrow [v2, v5, v9] 
$$

$$
c2 \rightarrow [v1, v4] 
$$

$$
c3 \rightarrow [v3, v6, v7, v8] 
$$

It is called an inverted list because it reverses the natural mapping. During ingestion, we determine:
```text
Vector ID → Centroid ID
```

For example:

$$
v1 \rightarrow c2
$$

$$
v2 \rightarrow c1
$$

$$
v3 \rightarrow c3
$$

However, during search, the system needs to perform the **opposite** operation: after selecting a centroid, it must immediately retrieve all vectors assigned to it. Therefore, the stored index uses:
```text
Centroid ID → Vector IDs
```

This is similar to a traditional text inverted index, where a word points to the documents containing it:
```text
"physics" → [doc_2, doc_8, doc_15]
```

In IVF, the word is replaced by a centroid:

```text
centroid_3 → [vector_12, vector_41, vector_96]
```

The vector IDs connect the search results to the original documents, chunks, images, or database records. In an IVF-Flat index, the corresponding full vectors are also stored in, or referenced by, each inverted list. Other IVF variants may instead store compressed vector representations to reduce memory usage.

3. **Query Phase:** The query vector $q$ is first compared against only the $k$ centroids. The system selects the $n$ closest centroids (a hyperparameter called `nprobe`) and the system then compares the query with every candidate vector stored in the selected inverted lists. This local scan uses the same distance or similarity measure introduced in the Flat Indexing section, such as Euclidean distance, inner product, or cosine similarity. The difference is that Flat Indexing scans the entire dataset, whereas IVF scans only the vectors belonging to the selected clusters.


#### **The Boundary Problem: Why `nprobe` Exists**

Assigning every vector to only one centroid creates a problem near the boundaries between clusters. Two vectors can be very close to each other while being assigned to different inverted lists.

Consider a simple one-dimensional example with two centroids:


$$c_1 = 0, \qquad c_2 = 10$$

The boundary between their **Voronoi cells** is located at $5$. Any point smaller than $5$ is assigned to $c_1$, while any point larger than $5$ is assigned to $c_2$.

Now consider a query vector $q$ and a database vector $x$:


$$q = 4.9, \qquad x = 5.1$$

Their positions can be visualized as follows:

```text
       Cell assigned to c1       │       Cell assigned to c2
─────────────────────────────────│─────────────────────────────────
 c1                            q │ x                            c2
 0                           4.9 │ 5.1                          10
                                 ↑
                           Cell boundary

```

The query $q$ is assigned to $c_1$ because:


$$D(q,c_1) = 4.9 \quad \text{and} \quad D(q,c_2) = 5.1$$

The vector $x$ is assigned to $c_2$ because:


$$D(x,c_1) = 5.1 \quad \text{and} \quad D(x,c_2) = 4.9$$

However, the distance between the query and the vector is only:


$$D(q,x) = \vert{}4.9 - 5.1\vert{} = 0.2$$

Therefore, $x$ may be the true nearest neighbor of $q$, even though they belong to different inverted lists.

#### **How `nprobe` Fixes This**

If `nprobe = 1`, IVF searches only the list associated with the closest centroid, $c_1$. Since $x$ is stored in the list of $c_2$, it is not examined and may be missed.

```text
[nprobe = 1]

Query q
   │
   ▼
Search list[c1] only
   │
   └── x is not considered because x is stored in list[c2]

```

If `nprobe = 2`, IVF searches both the closest and the second-closest centroid lists:

```text
[nprobe = 2]

Query q
   │
   ├── Search list[c1]
   └── Search list[c2]  ──>  x is examined

```

> **The Takeaway:** This is the reason `nprobe` exists. Searching multiple nearby cells reduces the probability of missing close vectors that lie on the other side of a cluster boundary. However, increasing `nprobe` also increases the number of candidate vectors that must be compared with the query, creating a tradeoff between accuracy and speed.

#### **Advantages**

* **Tunable trade-off:** `nprobe` gives you a single, intuitive knob to trade recall for latency at query time — no re-indexing required.
* **Low memory overhead:** Unlike graph-based methods, IVF doesn't store dense adjacency structures. You're storing centroids and flat lists, which is cheap.
* **Fast to build:** Training $k$-means on a sample and assigning vectors to clusters is dramatically cheaper than constructing a multi-layer graph over the entire dataset.

#### **Flaws**


* **Boundary errors:** Each vector is normally assigned to only one centroid. As shown in the previous example, a query and its true nearest neighbor may lie on opposite sides of a Voronoi boundary and therefore be stored in different inverted lists. Searching more lists by increasing `nprobe` reduces this problem, but also increases the number of distance computations.

* **Uneven inverted-list sizes:** Standard $k$-means minimizes the distance between vectors and their assigned centroids, but it does not guarantee that every cluster will contain approximately the same number of vectors. Dense regions of the embedding space may create very large inverted lists, while sparse regions may create small or nearly empty ones. This makes search latency less predictable because the cost depends not only on `nprobe`, but also on how many vectors are stored in the selected lists.

* **Sensitive to the shape of the data distribution:** (k)-means works best when clusters are reasonably compact and can be represented well by a single centroid. Real embedding distributions may contain elongated, overlapping, or irregularly shaped regions. In such cases, the centroid partitioning may not reflect the actual nearest-neighbor structure of the data.

* **The number of clusters must be selected carefully:** The number of centroids, commonly called `nlist`, directly affects index quality and search cost. If `nlist` is too small, each inverted list contains many vectors and the local flat search remains expensive. If `nlist` is too large, centroid search becomes more expensive, some lists may contain very few vectors, and more training data is required to estimate reliable centroids.

* **Centroids can become outdated:** The centroids describe the distribution of the data at the time the index is trained. If the dataset later changes—for example, because documents from a new domain are added—the new vectors must still be assigned to the old centroids. Some lists may become overloaded, while new regions of the embedding space may be represented poorly. Correcting this usually requires retraining the centroids and reassigning the indexed vectors.

* **Updates require index maintenance:** Adding a vector is relatively simple: the system finds its nearest centroid and appends it to the corresponding inverted list. Updating an existing vector may require moving it from one list to another. Deletions may require additional ID mappings, deletion markers, or periodic index rebuilding, depending on the implementation.

* **Exact recall is not guaranteed when only part of the index is searched:** IVF is approximate when `nprobe` is smaller than the total number of lists. A relevant vector may exist inside a list that was not selected. When exact nearest-neighbor results are required, IVF must either search every list or be followed by an exact verification stage over a sufficiently complete candidate set.



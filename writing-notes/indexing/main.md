---
layout: post
title: "Indexing in RAG piepline"
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

*   **The :** An abstract look at how indexes trade space complexity for time complexity, and the exact cost of running an unindexed system.
*   **The Traditional vs. Vector Divide:** Disentangling exact, deterministic lookups (B-Trees, Hash maps) from probabilistic, high-dimensional **Approximate Nearest Neighbor (ANN)** search.
*   **Core Algorithmic Archetypes:**
    *   **Inverted File Indexing (IVF):** Voronoi partitioning, $k$-means quantization, and inverted lists.
    *   **Tree-Based Indexing:** KD-Trees, Ball Trees, and why they fail in high dimensions ($d > 50$).
    *   **Graph & Quantization Variants:** The mechanics of HNSW (Hierarchical Navigable Small World) and Product Quantization (PQ).
*   **The "Garbage In, Garbage Out" Trap:** Why indexing cannot save bad tokenization, poor chunking strategies, or misaligned embedding models.
*   **The "No-Index" Regime:** When exhaustive brute-force search is actually superior to building an index.

## 1. The Foundational Mechanics: Why Indexing Matters

The most noraml and common question is: **What happens if we do not index our data?**

To answer this question, let's start with a metaphor. Imagine you have a massive library filled with books on math, physics, chemistry, sports, and various other topics.

Suppose you place all these books onto your bookshelves completely at random, without any arrangement. This initial setup is incredibly simple. Because you don't have to worry about order or placement, throwing the books onto the shelves takes the least amount of time and energy.

But the real problem starts when I ask you to find a specific book. To find it, you have to walk through the library and check the books one by one. This approach works fine if you only have a few dozen books, but it becomes an absolute nightmare if your library grows to several million.

Now, let's go a step further. Suppose we group the books by subject, placing all books with a common topic next to each other. However, we don't know which shelves hold which subjects. If I ask you for a specific book, you would first consider its subject, and then look for the shelf where that subject is located.

To find the correct shelf, you have to check the first book of every single aisle. If that first book matches your subject, you stop and search that shelf; otherwise, you skip it and move to the next. This is significantly more efficient than the first method because once you find the right shelf, you can skip all the others. However, you still waste a lot of time searching for the shelf itself.

Now, let's introduce a master directory or clear aisle labels—which acts exactly like a database index. If you not only group common subjects together, but also create a master map showing exactly where each specific subject lives, the search changes completely. For example, the map tells you that math books are always on the first floor, on the left. When you are asked to find a math book, you don't wander or guess; you immediately go straight to that specific shelf. And since all the math books are consolidated right there, your search is narrowed down to just a tiny fraction of the library.

With this example in mind lets talk about indexing in technical way. When you write data to a raw storage medium without an index, it is typically appended to a log or organized in a heap structure. This makes writes incredibly cheap ($O(1)$), but it turns reads into an expensive computational nightmare.

### The Mechanics of the "No-Index" Regime

Without an index, any lookup requires a sequential, exhaustive scan. The system must load every record from storage into memory and evaluate it against the query predicate.

* **In Traditional Databases:** In this case each data is a point (has $1$ dimenstion). A point lookup on an unindexed column forces a Full Table Scan. The time complexity scales strictly linearly:

$$O(N)$$

where $N$ is the number of rows.

* **In Vector Spaces (RAG Pipelines):** Each data is a vector with domesntion $d$. Searching an unindexed vector space requires an exhaustive flat search. To find the nearest neighbor, the system must compute the distance (e.g., Cosine or Euclidean) between the query vector and every stored vector. The time complexity scales as:

$$O(N \cdot d)$$

where $N$ is the number of vectors and $d$ is the dimensionality of the vector space (typically $d = 768$ or $d = 1536$ in modern embedding models).

For a dataset of 10 million vectors with 1,536 dimensions, single query requires
*   **10 million floating-point operations (FLOPs)** in traditionl case. 
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

By imposing geometric, hierarchical, or mathematical order onto the data during ingestion, the index allows the query processor to discard the vast majority of the search space immediately. This shifts the runtime boundary from linear ($O(N)$) down to logarithmic ($O(\log N)$) or near-constant ($O(1)$) complexities. Exactly like knowing the math book is always on the top shelf on the left in the metheafor. 


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



### The Two Phases of Indexing (And Why It Scales)

To understand why this exact-match system effortlessly handles massive scale—like Amazon or Google searching billions of items before the AI era—it helps to break indexing down into two distinct steps:

1. **Indexing the Document (Creation):** When new data enters the database, the system extracts exact attributes (keywords, IDs) and files them into a B-Tree or Inverted Index. This requires upfront work, but it maps the data perfectly.
2. **Indexing in the Search Phase (Retrieval):** When a user types a query, the system doesn't scan millions of documents. It takes the exact keyword or ID, checks the pre-built index, and follows the pointer directly to the item.

Because the search phase doesn't read the documents, big data isn't a problem. These data structures operate on logarithmic time complexity, expressed mathematically as $O(\log n)$. Practically, this means if a database grows from one million to one billion records, finding an exact match doesn't take a thousand times longer—it only takes a few extra computational steps, because the system skips half the remaining data with every single step. Furthermore, comparing exact strings (`"Apple"` vs `"Apple"`) requires almost zero computational overhead.


Traditional indexing is undeniably fast, infinitely scalable, and perfectly suited for exact searches. However, what happens if there is not exact match for you query? what happens if you onyl knw a partia lexact information of  you query? Lets talk about searching about words and sentences.  An exact-match index only knows what a word *is*, not what a word *means*. If you search a traditional database for "puppy," it will instantly return every document containing the word "puppy." But it will completely ignore a document about a "young dog," because those exact letters do not match. 

To move forward, we must transition from finding the **exact** item to finding the **closest** item. More professionally, vector indexing in RAG pipelines operates in the realm of **Approximate Nearest Neighbor (ANN)** search.

However, when we talk about concepts like "close," "near," "far," or "similar," we must take into account **dimensionality** and the **concept of distance**. In the exact-match paradigm, the world is binary: either there is a match, or there isn't. In the non-exact paradigm, we are hunting for the best options that reside closest to our query. If you have closely followed the indexing concept so far, you might notice that this phase doesn't actually involve indexing at all yet. Instead, it relies entirely on **how** you represent your documents and **how** you calculate their similarity.

This representation step is called **embedding**, which we will discuss in deep detail in a future article. For now, it is enough to know that embedding transforms each data point—whether it is a word, a sentence, or an entire paragraph—into a vector with dimension $d$. When a user submits a query, the system transforms that query into a vector too, and compares it against all the other vectors to find the closest matches.

While this approach beautifully solves the semantic search problem, it comes at a high price. In terms of latency and computation, it is far more expensive than traditional exact search. If you have millions of data points, comparing a new query vector against every single vector in your database is incredibly inefficient. We need a way to **skip** the vast majority of irrelevant data. This is exactly **where the indexing step happens**.

To understand this more easily, imagine that you go to a massive bookstore to buy a book. You don't know the exact name of the book, but you tell the bookseller that you are looking for something related to the subject of mathematics.

When you describe what you want, the seller's ability to help depends entirely on their mental understanding of the subject and how they have framed the store's inventory in their mind. Two distinct steps happen here:

** **Understanding the Concept:** The bookseller must grasp the meaning of your request and mentally map it to the subjects in the store. They must realize that a book on geometry or calculus fits your needs, even if the word "mathematics" isn't on the cover. This conceptual mapping is the embedding.

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

In this section we will mention most common indexing algorithms used in modern database libraries such as `FAISS` , `LlamaIndex` and `ChromaDB`. These are open source libraies and have a good compatiblity with RAG and Agentic frameworks. 

For undestadning this section no prior knowledge about embedding models needed. The only point that needs to be considered is that every input after passing trought the embedding model transforms into a vector of dimension $d$. 

Altough the first method `Flat Indexing` is not a *real* indexing method it is just comparision between all other vectors with eucledean metric. But in the AI community it is often mentioned as a way of indexing.

### 1. Flat Indexing (Brute Force)

Before we jump into all the fancy approximate search methods (ANN), we have to establish our exact-match baseline: the Flat Index. In a Flat Index, we just store the vectors exactly as they are generated. No structural organization, no compression, just raw data. When a query vector $q$ arrives, the system does an exhaustive (through all vectors of the database) search. Basically we are done with indexing. The indexing step is done at this stage, **however** the intresting part is **how** the search is done. 

In general the system calculates the mathematical distance between $q$ and every single document vector $x$ in the entire dataset $X$ of size $N$. Because we are comparing every single dimension ($d$) of every single vector ($N$), the time complexity is $O(N \cdot d)$. It guarantees 100% perfect recall, but as our dataset grows, this brute-force approach becomes computationally non appealing to marketing team.

In the follwing we will mention some of these metrics. Again these (metrics) happen **after** indexing. 

Throughout this section  $q$ is a query vector and $x$ is a sample point in our dataset. Both $q$ and $x$ cane be represented as: 

$$q = (q_1,q_2,\cdots,q_d)$$

and 

$$x = (x_1,x_2,\cdots,x_d)$$

for $ q_1,q_2,\cdots,q_d, x_1,x_2,\cdots,x_d \in \mathbb{R}$. And for every vector $x$ the $L2$ norm of $x$ is defines as:

$$  \Vert x\Vert_2 =  \sqrt{\sum_{j=1}^{d} x_{j}^2} $$

#### **Cosine (aka Cosine Similarity) and Inner Product** 

Between $q$ and $x$ the  **Cosine similarity** for 

$$S_C(q, x) = \frac{q \cdot x}{\Vert q\Vert \Vert x\Vert} = \frac{\sum_{j=1}^{d} q_j x_{j}}{\sqrt{\sum_{j=1}^{d} q_j^2} \sqrt{\sum_{j=1}^{d} x_{j}^2}}$$

However, in modern RAG architectures, we almost always prefer the **Inner Product** (Dot Product) instead. It’s simply the unnormalized projection of one vector onto another:

$$IP(q, x) = q \cdot x = \sum_{j=1}^{d} q_j x_{j}$$

Why the shift to Inner Product? It comes down to pure hardware efficiency. If we $L2$-normalize (divide each vector by its $L2$ norm) our vectors before indexing them, the denominators in the Cosine Similarity equation become $1$. This makes Cosine Similarity and Inner Product mathematically equivalent. By stripping out the square roots and division, the Inner Product drastically reduces the CPU/GPU cycles needed during a massive brute-force scan while giving us the exact same ranking.

Here we would like to mention that in pure mathematics, a "metric" (or distance function) on an $n$-dimensional space $\mathbb{R}^n$ is a specific function $d: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ that *must* satisfy three unbending rules:

1. **Non-negativity:** The distance between two points is always $\ge 0$, and the distance is exactly $0$ if and only if the two points are identical. ($d(x, y) = 0 \iff x = y$).
2. **Symmetry:** The distance from $x$ to $y$ is the exact same as $y$ to $x$. ($d(x, y) = d(y, x)$).
3. **The Triangle Inequality:** The direct path from $x$ to $z$ is always shorter than or equal to going from $x$ to $y$, and then $y$ to $z$. ($d(x, z) \le d(x, y) + d(y, z)$).

As engineers, we accept this jargon, but strictly mathematically speaking, **this is heavily misused.**  

**Inner Product and Cosine Similarity completely fail these mathematical tests.** They are *similarity comparisons*, not actual distances. Inner product can yield negative numbers, failing rule #1. Cosine similarity increases the closer two vectors get, which is the exact opposite of a distance function. Even if you invert it into $1 -$ Cosine Similarity, it still famously violates the Triangle Inequality (For more information of how we can create a distance frim inner product you can refer to my article on postional encodding [here](https://erfansoheil.github.io/writing-notes/positional_embeddings/main.html)).

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
If you are using **Inner Product (IP), Cosine Similarity, or Euclidean Distance ($L_2$)**, you are in the safe zone. Every major vector database supports these natively out-of-the-box. Their underlying graph algorithms (like HNSW) and quantization techniques are heavily optimized for these three specific calculations.

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

1. **Query Phase:** The query vector $q$ is first compared against only the $k$ centroids. The system selects the $n$ closest centroids (a hyperparameter called `nprobe`) and the system then compares the query with every candidate vector stored in the selected inverted lists. This local scan uses the same distance or similarity measure introduced in the Flat Indexing section, such as Euclidean distance, inner product, or cosine similarity. The difference is that Flat Indexing scans the entire dataset, whereas IVF scans only the vectors belonging to the selected clusters.

Here is the improved Markdown, utilizing proper LaTeX formatting for the math equations and code blocks to preserve the alignment of your diagrams.

---

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

* **Uneven inverted-list sizes:** Standard (k)-means minimizes the distance between vectors and their assigned centroids, but it does not guarantee that every cluster will contain approximately the same number of vectors. Dense regions of the embedding space may create very large inverted lists, while sparse regions may create small or nearly empty ones. This makes search latency less predictable because the cost depends not only on `nprobe`, but also on how many vectors are stored in the selected lists.

* **Sensitive to the shape of the data distribution:** (k)-means works best when clusters are reasonably compact and can be represented well by a single centroid. Real embedding distributions may contain elongated, overlapping, or irregularly shaped regions. In such cases, the centroid partitioning may not reflect the actual nearest-neighbor structure of the data.

* **The number of clusters must be selected carefully:** The number of centroids, commonly called `nlist`, directly affects index quality and search cost. If `nlist` is too small, each inverted list contains many vectors and the local flat search remains expensive. If `nlist` is too large, centroid search becomes more expensive, some lists may contain very few vectors, and more training data is required to estimate reliable centroids.

* **The quality of the training sample matters:** IVF usually trains its centroids on a sample rather than the complete dataset. If the sample does not represent the full distribution, rare document types or semantic regions may be missing during training. Their vectors will then be assigned to centroids that do not represent them well, which can reduce recall and produce unbalanced inverted lists.

* **Centroids can become outdated:** The centroids describe the distribution of the data at the time the index is trained. If the dataset later changes—for example, because documents from a new domain are added—the new vectors must still be assigned to the old centroids. Some lists may become overloaded, while new regions of the embedding space may be represented poorly. Correcting this usually requires retraining the centroids and reassigning the indexed vectors.

* **Training is not deterministic:** (k)-means can converge to different solutions depending on its initialization, training sample, and number of iterations. Two training runs may therefore produce different centroids, cluster sizes, and retrieval performance. Initialization methods such as (k)-means++ can reduce this variability, but they do not remove it completely.

* **Updates require index maintenance:** Adding a vector is relatively simple: the system finds its nearest centroid and appends it to the corresponding inverted list. Updating an existing vector may require moving it from one list to another. Deletions may require additional ID mappings, deletion markers, or periodic index rebuilding, depending on the implementation.

* **Exact recall is not guaranteed when only part of the index is searched:** IVF is approximate when `nprobe` is smaller than the total number of lists. A relevant vector may exist inside a list that was not selected. When exact nearest-neighbor results are required, IVF must either search every list or be followed by an exact verification stage over a sufficiently complete candidate set.



### **3. HNSW (Hierarchical Navigable Small World)**

The Hierarchical Navigable Small World (HNSW) algorithm is currently one of the most efficient and widely used structures for approximate nearest neighbor (ANN) search in high-dimensional spaces. To deeply understand how HNSW works, we must deconstruct it into its two foundational concepts: the probabilistic hierarchy of a **Skip List** (which operates in 1D space) and the decentralized graph routing of **Navigable Small Worlds (NSW)** (which operates in high-dimensional space). 



HNSW adapts two key properties from two mentioned algorithms: 1- Skipping most of data 2- Navigating in a compact subsapce of data. Wit hthes two proeptires, HNSW elegantly solves the problem of searching massive vector databases with $O(\log n)$ complexity.






#### **1. The Skip List**

Imagine a long sorted linked list. Every element knows only its immediate successor. If we are searching for a target value, the ordering tells us whether to continue moving forward or stop. We do not need to search in arbitrary directions, but we still have to advance one node at a time.

For a short list, this is acceptable. For millions of elements, however, it becomes inefficient. Reaching a value near the end of the list may require visiting almost every element that comes before it. Therefore, searching an ordinary linked list has a worst-case time complexity of $O(n)$.

The natural idea is to skip over some elements instead of visiting every node. However, a single system of large jumps is not sufficient. If a jump takes us beyond the target, a singly linked list does not provide an efficient way to move backward and recover. We need large jumps when we are far from the target, followed by progressively smaller and more precise steps as we approach it.

A skip list provides exactly this mechanism by representing the same ordered collection at multiple levels. The bottom level, $L_0$, contains every element. Each higher level contains only a subset of the elements in the level below it.

The upper levels therefore act as express lanes. They allow the search to cross large parts of the list quickly. When the next jump would pass the target, the search descends to a denser level, where the jumps are shorter and more precise. Eventually, it reaches the complete bottom level, where the exact target can be found.

The structure can be summarized as follows:

* The bottom level provides completeness and precision.
* The upper levels provide speed.
* Searching alternates between moving right and moving down.

---

#### **Probabilistic Construction: The Coin Flip**

A skip list is constructed incrementally. Whenever a new node is inserted, it is always placed in the base level $L_0$. A randomized procedure then determines whether the node should also appear in the higher levels.

Let $p$ be the probability that a node is promoted to the next level.

1. Every node is inserted into $L_0$.
2. With probability $p$, it is also promoted to $L_1$.
3. If it reaches $L_1$, it is promoted to $L_2$ with probability $p$ again.
4. This process continues until the first unsuccessful promotion.

Equivalently, we may imagine repeatedly flipping a biased coin. A successful flip promotes the node by one level, while an unsuccessful flip stops the process.

The probability that a node reaches level $k$ is

$$
p^k.
$$

For example, when $p=\frac{1}{2}$:

* Every node appears in $L_0$.
* Approximately half of the nodes appear in $L_1$.
* Approximately one quarter appear in $L_2$.
* Approximately one eighth appear in $L_3$.

In general, the expected number of nodes in level $L_k$ is

$$
np^k,
$$

where $n$ is the total number of elements.

The number of nodes therefore decreases exponentially as we move upward. This produces sparse upper levels containing long-range shortcuts and dense lower levels containing shorter, more precise connections.

---

#### **The Role of the Promotion Probability**

The promotion probability $p$ directly controls the shape of the skip list.

A larger value of $p$ causes more nodes to be promoted. This produces:

* more populated upper levels;
* more pointers and greater memory usage;
* shorter distances between nodes within each level;
* usually fewer horizontal steps before descending.

A smaller value of $p$ causes fewer nodes to be promoted. This produces:

* sparser upper levels;
* fewer pointers and lower memory usage;
* larger jumps between promoted nodes;
* potentially more horizontal work during the search.

For example, suppose that the list contains $1{,}000$ elements.

If $p=\frac{1}{2}$, the expected level sizes are approximately

$$
1000,\ 500,\ 250,\ 125,\ 62,\ldots
$$

If $p=\frac{1}{4}$, the expected level sizes are approximately

$$
1000,\ 250,\ 62,\ 15,\ 4,\ldots
$$

The second skip list has fewer levels and requires less memory, but its levels are much sparser.

Therefore, changing $p$ changes the number of levels, the number of nodes in each level, the average jump length, and the memory–search trade-off.

---

#### **Randomness as Both a Strength and a Weakness**

The probabilistic construction is the central strength of a skip list. It creates a balanced hierarchy without requiring expensive rotations, global reorganization, or deterministic balancing rules.

However, randomness also introduces an important weakness: the exact structure is not guaranteed.

Even when two skip lists contain the same elements and use the same probability $p$, they may have different structures because their promotion outcomes can differ. A node may reach several upper levels in one construction but remain only in the base level in another.

The expected shape is well controlled, but the exact shape is random.

This means that:

* the number of nodes in each level is only an expected value;
* the actual maximum height may vary;
* search paths may differ between constructions;
* performance is guaranteed in expectation rather than for every possible random outcome;
* a poor sequence of random promotions can produce an unbalanced structure.

In an extreme but unlikely case, no node may be promoted beyond $L_0$. The skip list would then behave like an ordinary linked list and require $O(n)$ search time. Conversely, too many promotions could create unnecessarily dense upper levels and increase memory consumption.

Thus, probability is not simply a defect of the skip list. It is both its balancing mechanism and a source of variability. The parameter $p$ must be chosen carefully because it controls the trade-off between memory usage, hierarchy height, and horizontal search cost.

In practical implementations, a maximum permitted height is usually imposed to prevent the structure from growing without bound.

---

#### **Search Mechanics**

To search for a target value $T$:

1. Begin at the head node of the highest nonempty level.
2. Inspect the next node on the current level.
3. If the next node has key $T$, the target has been found.
4. If the next key is smaller than $T$, move horizontally to that node.
5. If the next key is greater than $T$, or if no next node exists, descend by one level.
6. Repeat until the target is found or its possible position is passed in $L_0$.

The search follows a simple rule:

> Move right while the next jump does not overshoot the target. Move down when the next jump is too large.

Suppose we are searching for $73$. At a sparse upper level, the search might move from $10$ to $40$ and then to $70$. If the next available node is $100$, moving to it would overshoot the target. The search therefore descends to a lower, denser level from $70$ and continues with smaller steps until it either reaches $73$ or determines that $73$ is absent.

The algorithm never needs to move backward. Descending to a more detailed level replaces backward recovery.

---

#### **Expected Height**

At level $k$, the expected number of nodes is

$$
np^k.
$$

The highest useful level is approximately the level at which only one node is expected to remain. Therefore, we set

$$
np^h \approx 1.
$$

Rearranging gives

$$
p^h \approx \frac{1}{n}.
$$

Taking logarithms gives

$$
h \approx \log_{1/p}n.
$$

Therefore, the expected height of a skip list grows logarithmically with the number of elements.

For $p=\frac{1}{2}$, this becomes

$$
h \approx \log_2 n.
$$

For example, if $n=1{,}000{,}000$, then

$$
\log_2(1{,}000{,}000)\approx 20.
$$

Thus, a skip list containing approximately one million elements may require only around twenty useful levels.

---

#### **Expected Search Complexity**

At each level, the search performs some horizontal movements before descending. The expected amount of horizontal work per level is a constant that depends on $p$.

A common intuitive approximation is

$$
\text{Expected horizontal work per level}\approx\frac{1}{p}.
$$

Since the expected number of levels is approximately

$$
\log_{1/p}n,
$$

the expected search cost can be described as

$$
\text{Expected search cost}
\approx
\frac{1}{p}\log_{1/p}n.
$$

When $p$ is treated as a fixed constant independent of $n$, both $\frac{1}{p}$ and the logarithm base are constant factors. Therefore,

$$
\frac{1}{p}\log_{1/p}n=O(\log n).
$$

A skip list consequently provides expected search complexity

$$
O(\log n),
$$

compared with

$$
O(n)
$$

for an ordinary linked list.

The word **expected** is important. The skip list does not guarantee logarithmic search time for every possible random construction. Its worst-case search time remains

$$
O(n).
$$

The logarithmic result describes its average behaviour over the random promotion process.

---

#### **A Library Metaphor**

Imagine a very large library whose books are arranged by catalogue number. Suppose a librarian needs to find book number $847{,}230$.

The complete catalogue lists every book in increasing numerical order. The librarian could begin at the first entry and inspect every catalogue number until reaching $847{,}230$. The ordering ensures that the librarian knows whether to continue, but this method may still require examining hundreds of thousands of entries.

To accelerate the search, the library creates several directory maps.

* The most detailed directory contains every shelf.
* A higher-level directory marks only selected shelves.
* Another directory marks only selected aisles.
* The sparsest directory may mark only a few major sections of the library.

The librarian begins with the sparsest directory. As long as the next checkpoint remains below catalogue number $847{,}230$, the librarian moves to it. When the next checkpoint would go beyond the desired number, the librarian switches to a more detailed directory.

For example, the librarian might follow checkpoints corresponding to

$$
100{,}000 \rightarrow 500{,}000 \rightarrow 800{,}000.
$$

If the next major checkpoint is $900{,}000$, it would pass the target. The librarian therefore consults a more detailed directory beginning from $800{,}000$. The process continues until the exact shelf and book are found.

The sparse directories provide speed, while the complete directory provides precision.

The correspondence is:

| Skip-list concept   | Library equivalent                            |
| ------------------- | --------------------------------------------- |
| Node                | Book or catalogue position                    |
| Key                 | Catalogue number                              |
| Base level $L_0$    | Complete catalogue                            |
| Upper levels        | Progressively sparser directories             |
| Horizontal movement | Moving to the next checkpoint                 |
| Vertical movement   | Consulting a more detailed directory          |
| Promotion           | Selecting a book or shelf as a checkpoint     |
| Overshooting        | The next checkpoint exceeds the target number |

The probabilistic element means that checkpoints are not chosen according to a perfectly regular rule. Two libraries containing the same books could select different shelves as checkpoints. Their directory structures would differ, although both would have the same expected density at each level.

---

#### **Why Total Order Matters**

A skip list relies on the fact that its keys belong to a **totally ordered set**.

A set is totally ordered when any two elements can be compared. For two different elements $a$ and $b$, either

$$
a<b
$$

or

$$
b<a.
$$

This property provides a unique direction for the search. If the next value is smaller than the target, we continue moving right. If it is larger, we descend to a more precise level.

Numbers have a natural total order. Words can also be ordered alphabetically. Dates can be ordered chronologically.

Multidimensional vectors, however, do not have an equally natural order that represents their geometry.

For example, consider the two-dimensional vectors

$$
(2,10)
$$

and

$$
(3,1).
$$

It is not geometrically meaningful to say that one vector naturally comes before the other in the same way that $2<3$.

We can impose an artificial total order on vectors. One common choice is called **lexicographic order**.

For two vectors

$$
x=(x_1,x_2)
$$

and

$$
y=(y_1,y_2),
$$

we say

$$
x<_{\mathrm{lex}}y
$$

when either

$$
x_1<y_1,
$$

or when the first coordinates are equal and

$$
x_2<y_2.
$$

In other words, we compare the first coordinates. Only if they are equal do we compare the second coordinates. In higher dimensions, we continue coordinate by coordinate until the first unequal pair is found.

For example,

$$
(1,100)<_{\mathrm{lex}}(2,0)
$$

because $1<2$. The second coordinates do not matter once the first coordinates differ.

This is similar to dictionary ordering. The words “apple” and “banana” are ordered according to their first different letter.

Lexicographic order gives vectors a valid total order, but it does not preserve geometric proximity. Two vectors that are close in lexicographic order may be far apart in space, while two geometrically close vectors may be separated by many other vectors in the lexicographic ordering.

For instance,

$$
(1,100)
$$

and

$$
(2,0)
$$

are consecutive under some lexicographic arrangements, but their Euclidean distance is approximately

$$
\sqrt{(2-1)^2+(0-100)^2}
========================

\sqrt{10001},
$$

which is close to $100$.

Therefore, an imposed order such as lexicographic order is not sufficient for nearest-neighbour search. In multiple dimensions, we are usually interested not in which vector comes before another, but in which vector is closest to a query.

This requires a distance or similarity function, such as Euclidean distance, cosine similarity, or inner-product similarity.

---

#### **From Skip Lists to HNSW**

The skip list provides the central intuition behind HNSW:

* use sparse upper levels for long-range movement;
* use dense lower levels for precise local search;
* begin with large steps;
* gradually move toward smaller steps.

However, the navigation rule changes.

In a skip list, movement is guided by total order. The algorithm knows whether the target lies to the right because keys can be compared using $<$ and $>$.

In HNSW, there is no useful one-dimensional order. Instead, movement is guided by distance. At every step, the algorithm selects neighbouring vectors that appear closer to the query.

Therefore:

> A skip list navigates through an ordered sequence, while HNSW navigates through a metric or similarity space.

The skip list asks:

> Is the next key still smaller than the target?

HNSW instead asks:

> Does this neighbouring vector bring us closer to the query?

Both structures use a hierarchy of increasingly dense layers. The difference is that skip-list shortcuts follow a linear order, whereas HNSW connections form a proximity graph.

There is also an important parallel in their probabilistic construction. In both structures, randomization influences the level assigned to each element. Changing the level probability changes the density of the hierarchy, the number of elements in upper layers, memory consumption, and search behaviour.

This randomness makes the structure efficient to construct, but it also means that the resulting hierarchy is not unique or completely deterministic.










#### **1. The Skip List**

Imagine a long sorted linked list. Every element knows only its immediate successor. If you are searching for a value, the ordering immediately tells you whether you should continue moving forward or stop—you never need to move backward—but you still have to advance one node at a time.

For a short list this is acceptable. For millions of elements, however, this becomes painfully inefficient because reaching a distant item may require traversing almost every intermediate node.

The obvious idea is to skip ahead instead of visiting every element. But skipping introduces a new problem: if the jumps are too large, you may leap over the element you are searching for with no easy way to recover. What we need is a mechanism that allows large jumps when we are far away from the target, yet gradually switches to smaller, more precise steps as we get closer.

A skip list achieves exactly this by organizing the same ordered sequence into multiple layers. The bottom layer contains every element and guarantees that the target can always be reached exactly. Higher layers contain progressively fewer elements, acting as express lanes that let the search cover large portions of the list in a handful of jumps before descending to finer-grained layers for the final search.

#### **Probabilistic Construction (The Coin Flip)**
A skip list is built iteratively, layer by layer. 
1. **The Base Layer ($L_0$):** Every element inserted into the skip list exists in a strictly sorted base layer, $L_0$. 
2. **Promotion:** When a new node is inserted into $L_0$, the algorithm flips a biased or fair coin (with probability of heads, $p$). 
3. **Height Determination:** If the coin lands heads, the node is promoted to layer $L_1$. The algorithm flips again. If heads, it is promoted to $L_2$. This continues until a tail is flipped. 

Because the probability of flipping $k$ consecutive heads is $p^k$, the probability of a node reaching layer $k$ decays exponentially. For $p = 1/2$, half the nodes reach $L_1$, a quarter reach $L_2$, and so on. This creates "express lanes": the top layers have exponentially fewer nodes, allowing a search algorithm to skip massive sections of the list.

#### **Search Mechanics and Complexity**
To search for a target value $T$:
1. Start at the highest layer of the "Head" node.
2. Look at the next node on the current layer $k$. 
3. If its value is $\le T$, traverse horizontally to that node.
4. If its value is $> T$ (or if it is a Null pointer), drop down vertically to layer $k-1$ and repeat.

The expected maximum height of a skip list with $n$ elements is $\approx \log_{1/p} n$. The expected number of horizontal traversals per layer before dropping down is $1/p$. Therefore, the total search time is the product of the height and the horizontal steps per level:
$$ \text{Expected Steps} = \left(\log_{1/p} n\right) \times \left(\frac{1}{p}\right) $$
This yields an overall time complexity of $O(\log n)$.


#### **2. Navigable Small Worlds (NSW)**

Skip lists are perfect for scalar values (1D) that can be sorted left-to-right. However, in modern machine learning, data points are complex vectors in high-dimensional space. We cannot easily sort them in a straight line. Instead, we organize them into a graph. 

### Small World Properties
A "Small World" graph has two defining properties:
1. **High Clustering:** Local neighborhoods form dense communities (if A is connected to B and C, B and C are likely connected).
2. **Short Path Length:** Despite local clustering, long-range "highways" exist, meaning the number of hops between any two random nodes is surprisingly small.

### Greedy Routing
A graph is "navigable" if an algorithm can find the shortest path using only local information. In NSW, this is done via **Greedy Routing**.
To find the nearest neighbor to a target vector $T$:
1. Start at a random node in the graph.
2. Calculate the distance (e.g., Euclidean or Cosine) between $T$ and all immediate neighbors of the current node.
3. Move to the neighbor that is mathematically closest to $T$.
4. Repeat until you reach a **local minimum**—a node where *none* of its neighbors are closer to $T$ than it is itself.

### The Problem: Local Minima
A flat NSW graph suffers from two issues:
1. Starting randomly means crossing the graph via many small hops is computationally expensive.
2. Greedy routing can get trapped in **local minima** (geometric "cul-de-sacs"), failing to find the true global nearest neighbor.



## 3. The Synthesis: HNSW

HNSW solves the local minima and efficiency problems of flat NSW by introducing the probabilistic hierarchy of the Skip List. 

Instead of a single graph, HNSW creates multiple NSW graphs stacked on top of each other. The top layers are extremely sparse (acting as the skip list's express lanes), while the bottom layer contains every single vector.

### The Mathematics of Hierarchy
When a new vector $v$ is inserted, its maximum layer $l$ is determined by a continuous geometric distribution, conceptually identical to the skip list's coin flip:
$$ l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor $$
* $\text{uniform}(0, 1)$ generates a random float between 0 and 1.
* $m_L$ is a scaling factor (typically $1 / \ln(M)$) that normalizes the decay.

Taking the natural log of a uniform distribution guarantees that the population of layers decays exponentially. If $L_0$ has $N$ nodes, $L_k$ will have approximately $N \cdot e^{-k/m_L}$ nodes.

### The Insertion Algorithm (Dropping the Anchor)
The multi-scale layers construct themselves dynamically as nodes are inserted:
1. **Freefall:** Starting from the global top layer ($L_{max}$), the algorithm performs a greedy search to find the closest node to $v$, dropping down layer by layer without making any connections, until it reaches the node's assigned maximum layer $l$.
2. **Wiring:** Once at layer $l$, it performs a greedy search strictly within that layer to find the $M$ nearest neighbors and creates bidirectional edges. 
3. **Descent:** It drops to layer $l-1$, using the neighbors found in layer $l$ as the new starting points, finds the $M$ nearest neighbors in $l-1$, and connects them. This repeats until $v$ is wired into $L_0$.

Because top layers are sparse, the edges formed there naturally bridge massive geometric distances. The bottom layers are dense, forming tight local clusters. During a search, you start at the top, taking massive leaps across the vector space (avoiding local minima entirely), and drop down through the layers to zero in on the precise local neighborhood.

### The Secret Engine: The Spatial Diversity Heuristic
If HNSW simply connected $v$ to the $M$ absolute closest nodes mathematically, it would fail. In high dimensions, the $M$ closest nodes are often clumped together in a dense cluster. Connecting only to them creates redundant edges pointing in the exact same geometric direction, destroying global navigability.

HNSW solves this using an **Edge Selection Heuristic** inspired by Relative Neighborhood Graphs. When deciding whether to connect $v$ to a candidate neighbor $C$, it applies the triangle inequality test:
* It evaluates candidates from closest to furthest.
* A candidate $C_x$ is only selected if its distance to $v$ is **strictly less than** its distance to any neighbor $C_{selected}$ already added to $v$'s connection list.
$$ \text{Keep } C_x \text{ if } dist(C_x, v) < dist(C_x, C_{selected}) \text{ for all } C_{selected} $$

**The Intuition:** If $C_x$ is closer to an already-connected neighbor than it is to $v$, it means $C_x$ is in the same spatial cluster. Adding a direct edge to it is redundant. By skipping it, the algorithm forces the $M$ edges to spread out across diverse geometric angles, ensuring that every node acts as an efficient multi-directional intersection.

#### Advantages

* **Best-in-class recall/latency trade-off:** For most real-world embedding distributions, HNSW simply outperforms IVF on the recall-vs-speed frontier. This is why it's the default choice in most modern vector databases (Qdrant, Weaviate, Milvus, and FAISS's `IndexHNSWFlat`).
* **No hard partitioning assumption:** Because it navigates via graph proximity rather than fixed clusters, HNSW doesn't suffer from IVF's "unlucky centroid" boundary problem in the same structural way.
* **Naturally incremental:** New vectors can be inserted into the existing graph without a full retraining pass, unlike IVF's centroid-dependent structure.

#### Flaws

* **Memory hunger:** Storing the adjacency lists for every node, at every layer, for a graph with potentially dozens of connections per node, often costs *more* RAM than storing the raw vectors themselves. For billion-scale datasets, this is frequently the binding constraint, not compute.
* **Slow to build:** Constructing a high-quality graph (high `efConstruction`) over millions of vectors is computationally expensive and doesn't parallelize as cleanly as IVF's cluster assignment.
* **Deletion is awkward:** Because nodes are woven into a graph via bidirectional edges, removing a vector cleanly means also repairing every edge that pointed to it. In practice, most implementations avoid this entirely and use tombstoning (marking a node dead without removing it), which means deleted vectors quietly continue to cost memory and traversal time until a full rebuild.

**The Math Trade-off:** HNSW drops search complexity to $O(\log N)$, offering blistering query speeds and elite recall. However, storing the adjacency lists for the complex graph connections requires an immense memory footprint (RAM), and unlike IVF — where you can rebuild centroids relatively cheaply — an HNSW graph is expensive enough to construct that engineers are often reluctant to rebuild it often, even as the underlying data distribution shifts.


### **4. PQ (Product Quantization)**

IVF and HNSW both answer the same question: *which vectors should I even bother comparing against?* Product Quantization answers a completely different one: *once I've decided to compare against a vector, how cheaply can I store and score it?* IVF and HNSW optimize *how* we search; PQ optimizes *what* we store. It is, at its core, a lossy compression technique for high-dimensional vectors.

1. **Vector Splitting:** A large, memory-heavy vector $x \in \mathbb{R}^d$ is chopped into $m$ smaller sub-vectors, each with $d/m$ dimensions.

$$x = [x^{(1)}, x^{(2)}, \dots, x^{(m)}]$$

2. **Sub-space Quantization:** For each of the $m$ sub-spaces, the system runs clustering (usually $k$-means) to find $k^*$ sub-centroids. Typically, $k^* = 256$, meaning each sub-centroid can be represented by an 8-bit integer (1 byte).
3. **Encoding:** The original sub-vectors are replaced by the ID (the 1-byte code) of their nearest sub-centroid. A massive 768-dimensional array of 32-bit floats is mathematically approximated as a tiny string of $m$ bytes.

$$x \approx [c_{i_1}^{(1)}, c_{i_2}^{(2)}, \dots, c_{i_m}^{(m)}]$$

4. **Asymmetric Distance Computation (ADC):** At query time, the query vector $q$ is *not* compressed. Instead, $q$ is split into $m$ parts. The system pre-calculates the distances between $q$'s sub-vectors and all possible 256 sub-centroids, storing them in a small lookup table. The total distance is simply the sum of these pre-calculated distances:

$$D(q, x) \approx \sum_{j=1}^{m} D(q^{(j)}, c_{i_j}^{(j)})$$

We'll formalize all four of these steps rigorously — codebooks, the full ADC derivation, memory arithmetic — in Section 6. For now, the intuition is enough to understand the trade-off.

#### Advantages

* **Dramatic memory savings:** Compressing a 3,072-byte float vector down to 64 bytes (a $48\times$ reduction, worked out in full in Section 6) is the difference between a dataset fitting in RAM and not fitting at all. For billion-scale corpora, this isn't a nice-to-have — it's often the only thing that makes the problem tractable on commodity hardware.
* **Cheap distance computation:** Because ADC replaces floating-point arithmetic with table lookups, scoring a compressed vector is extremely fast — you're doing $m$ array reads and a sum, not $d$ multiplications.
* **Composable:** PQ doesn't compete with IVF or HNSW; it *combines* with them. IVF-PQ is a standard production pattern precisely because the two solve orthogonal problems — IVF narrows down *which* vectors to check, PQ shrinks the cost of storing and checking each one.

#### Flaws

* **It is lossy, unavoidably:** Compression means information loss by definition. The distance you compute at query time is an *approximation* of the true distance, not the real thing — recall will always be strictly worse than an uncompressed index, no matter how well you tune it.
* **Codebooks need representative training data:** The $k$-means codebooks are only as good as the data they were trained on. If your live traffic drifts away from the distribution the codebooks were fit on, quantization error grows and recall silently degrades — the same staleness problem IVF has, but here it corrupts the *scoring*, not just the *routing*.
* **A tuning parameter you can get wrong in both directions:** Too few sub-vectors ($m$ small) and you don't save much memory; too many and each sub-space becomes so low-dimensional that the $k$-means clustering within it stops being meaningful, and quantization error balloons.

**The Math Trade-off:** PQ drastically reduces memory consumption (often by 90% or more) and replaces heavy floating-point arithmetic with lightning-fast $O(m)$ table lookups. The trade-off is a mathematically guaranteed drop in recall due to the lossy compression of the vectors. In massive production systems, it is frequently combined with IVF (as **IVF-PQ**) to achieve scale that would otherwise be impossible on limited hardware — a combination worth remembering, since it's what you'll actually deploy far more often than either technique alone. (A further refinement, Optimized Product Quantization (OPQ), rotates the vector space before splitting it into sub-vectors specifically to make the $m$ subspaces more independent — but that's a rabbit hole for another article.)


---

Three archetypes, three different bets: IVF bets on partitioning, HNSW bets on graph traversal, and PQ bets on compression instead of pruning. None of them is strictly "better" in the abstract — which is exactly the question the next section is actually about.










## **4. The Engineering Point of View: Trade-offs & The "No-Index" Regime**

In engineering, there is no such thing as a "better" index—there are only different profiles of trade-offs. Building an index is not a default architectural choice; it must be justified by data volume and latency requirements.

### The Trade-off Matrix

When selecting or tuning an index using popular libraries like **FAISS** (Facebook AI Similarity Search), **HNSWlib**, **Annoy**, or embedded engines like **LanceDB**, you are forced to balance four competing vectors:

```
                  [1] Search Latency (QPS)
                             ▲
                             │
 [4] Memory Consumption ◄────┼────► [2] Recall Accuracy
                             │
                             ▼
                  [3] Build/Update Time

```

1. **Search Latency (Queries Per Second):** How fast can the index resolve a query?
2. **Recall Accuracy:** What percentage of the actual, true mathematical top-$k$ nearest neighbors did the index successfully find?
3. **Build Time & Dynamic Updates:** How long does it take to construct or re-index the data? Can the index handle real-time streaming updates (like HNSW), or does it require periodic, heavy batch-rebuilding (like IVF)?
4. **Memory Consumption:** Does the index fit entirely into RAM? (HNSW can require up to 1.5x to 2x the raw data size in memory, whereas IVF-PQ can compress memory footprints by 85%).

### The "No-Index" Regime: When Exhaustive Search Wins

There are many architectural scenarios where you should **not** use an index at all. Running a flat, brute-force search (known as a `Flat` index in FAISS) is highly optimal under the following constraints:

* **Small Dataset Thresholds:** If your RAG pipeline contains fewer than 10,000 to 50,000 documents (e.g., searching within a single company's product documentation or a single user's email history), modern CPUs and GPUs utilizing SIMD (Single Instruction, Multiple Data) parallelism can execute a flat matrix multiplication faster than a CPU can traverse a complex HNSW graph.
* **100% Recall Requirement:** If your application cannot tolerate missing relevant data context (e.g., medical diagnostics, legal discovery, compliance auditing), ANN indexes are disqualified due to their inherent probabilistic loss of recall.
* **High-Frequency Writes:** If your application is writing new documents or updating vectors continuously, the CPU overhead of constantly modifying graph structures or recalculating centroids can bring your ingestion pipeline to a complete halt. Flat indexes handle dynamic append-only writes with zero overhead.


## **5. Concrete Engineering Implementations with FAISS**

The following code templates demonstrate how to implement the discussed index archetypes using **FAISS** (Facebook AI Similarity Search) in Python. We will use a standard dense vector dimension ($d = 768$, typical for models like `text-embedding-3-small` or `BERT`) and simulate a dataset of 100,000 documents.

```python
import numpy as np
import faiss
import time

# 1. Setup Mock Data (100k database vectors, 10 query vectors)
d = 768                                      # Dimensionality
n_base = 100000                              # Database size
n_query = 10                                 # Number of queries

np.random.seed(42)
xb = np.random.random((n_base, d)).astype('float32')
xq = np.random.random((n_query, d)).astype('float32')

# Define target k for top-k retrieval
k = 5

```

### Type A: Flat (Brute-Force) Index

The baseline index. It performs no compression and executes an exhaustive $O(N \cdot d)$ calculation.

```python
print("--- Building IndexFlatL2 ---")
index_flat = faiss.IndexFlatL2(d)            # Exact L2 distance index
print(f"Is trained: {index_flat.is_trained}") # True by default (no training needed)

index_flat.add(xb)                           # Add vectors directly

# Search
start_time = time.time()
distances, indices = index_flat.search(xq, k) 
print(f"Flat Search Time: {(time.time() - start_time) * 1000:.3f} ms")

```

### Type B: Inverted File (IVF-Flat) Index

Uses $k$-means to cluster the space into Voronoi cells. It requires explicit training on the data distribution before vectors can be mapped.

```python
print("\n--- Building IndexIVFFlat ---")
nlist = 100                                  # Number of Voronoi cells (clusters)
quantizer = faiss.IndexFlatL2(d)             # Coarse quantizer used to assign vectors to cells

indexvf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

print(f"Is trained before: {indexvf.is_trained}")
indexvf.train(xb)                          # Critical: Must train to find cluster centroids
print(f"Is trained after: {indexvf.is_trained}")

indexvf.add(xb)

# Adjust search-time parameters
indexvf.nprobe = 10                        # Look into the 10 closest clusters at query time

start_time = time.time()
distances, indices = indexvf.search(xq, k)
print(f"IVF Search Time: {(time.time() - start_time) * 1000:.3f} ms")

```

### Type C: HNSW (Hierarchical Navigable Small World) Index

A graph-based index offering high recall and ultra-low search latency, at the cost of high RAM utilization.

```python
print("\n--- Building IndexHNSWFlat ---")
M = 32                                       # Number of bi-directional links per node

index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.hnsw.efConstruction = 64          # Higher = more accurate graph building, slower ingest
index_hnsw.hnsw.efSearch = 32                # Higher = more accurate search, higher latency

index_hnsw.add(xb)                           # No training phase needed for HNSW

start_time = time.time()
distances, indices = index_hnsw.search(xq, k)
print(f"HNSW Search Time: {(time.time() - start_time) * 1000:.3f} ms")

```

### Type D: Composite Production Index (IVF + PQ via Index Factory)

For massive production scaling, you can combine methodologies using FAISS’s `index_factory`. This creates an IVF coarse quantizer coupled with Product Quantization bytes compression.

```python
print("\n--- Building Composite IVF+PQ via Index Factory ---")
# Factory string syntax: "IVF[centroids],[fine-quantizer]"
# "PQ64" means split the 768-dim vector into 64 sub-vectors of 12 dimensions each.
factory_string = "IVF256,PQ64"

index_composite = faiss.index_factory(d, factory_string)

print(f"Training composite index...")
index_composite.train(xb)
index_composite.add(xb)

# Set runtime cluster probe depth
faiss.extract_indexvf(index_composite).nprobe = 16

start_time = time.time()
distances, indices = index_composite.search(xq, k)
print(f"Composite IVF+PQ Search Time: {(time.time() - start_time) * 1000:.3f} ms")
print(f"Total Vectors Indexed: {index_composite.ntotal}")

```

---

## 6. Mathematical Breakdown of Product Quantization (PQ)

Product Quantization (PQ) is a lossy compression framework that enables memory reductions of up to 95% while natively supporting distance computations over compressed code domains.

### Step 1: Subspace Decomposition

Let a high-dimensional database vector $x \in \mathbb{R}^d$ be partitioned into $m$ orthogonal, lower-dimensional sub-vectors $x^1, x^2, \dots, x^m \in \mathbb{R}^{d^*}$, where:

$$d^* = \frac{d}{m}$$

The total vector space can be mathematically defined as the Cartesian product of these lower-dimensional subspaces:

$$\mathbb{R}^d = \mathbb{R}^{d^*} \times \mathbb{R}^{d^*} \times \dots \times \mathbb{R}^{d^*}$$

Visually, a 768-dimensional vector divided by $m=64$ yields 64 structural slices, each containing exactly 12 scalar fields.

### Step 2: Codebook Generation via $k$-means

For each independent subspace $i \in \{1, \dots, m\}$, a dedicated $k$-means clustering algorithm is executed on a training slice of the database:

$$\mathcal{X}^i = \{x_1^i, x_2^i, \dots, x_N^i\}$$

The optimization objective minimizes the intra-cluster sum of squared errors to resolve a discrete set of $k^*$ centroids:

$$\min_{C^i} \sum_{x^i \in \mathcal{X}^i} \min_{c \in C^i} \Vert{}x^i - c\Vert{}^2$$

where $C^i = \{c_1^i, c_2^i, \dots, c_{k^*}^i\}$ is the sub-codebook for subspace $i$.

In standard configurations, $k^*$ is set to 256 ($2^8$), meaning each sub-centroid index can be represented using exactly **1 byte (8 bits)** of information. The global codebook $\mathcal{C}$ is the total composition:

$$\mathcal{C} = C^1 \times C^2 \times \dots \times C^m$$

### Step 3: Quantization Mapping

The structural quantization function $q(x)$ maps the raw vector $x$ to a compressed tuple of integer assignments by finding the nearest sub-centroid in each subspace:

$$q(x) = \big( q_1(x^1), q_2(x^2), \dots, q_m(x^m) \big)$$

where:

$$q_i(x^i) = \arg\min_{j \in \{1, \dots, k^*\}} \Vert{}x^i - c_j^i\Vert{}$$

The vector $x \in \mathbb{R}^{768}$ (which originally required $768 \times 4 \text{ bytes} = 3072 \text{ bytes}$ of 32-bit floating-point storage) is now compactly represented as an array of $m$ bytes:

$$\text{Compressed Size} = m \times 1 \text{ byte}$$

If $m=64$, the memory footprint drops from 3,072 bytes to **64 bytes**—a $48\times$ compression factor.

### Step 4: Asymmetric Distance Computation (ADC)

To avoid decompression overhead during retrieval, PQ uses **Asymmetric Distance Computation (ADC)**. In ADC, the query vector $y$ remains unquantized at full floating-point precision, while the database vectors $x$ are evaluated via their quantized codes $q(x)$.

```
 Query (y)      [ y^1 (FP32) ]    [ y^2 (FP32) ]    ...    [ y^m (FP32) ]
                      │                 │                       │
                Distance Table    Distance Table          Distance Table
                      ▼                 ▼                       ▼
 Code q(x)      [  Byte ID  ]     [  Byte ID  ]     ...    [  Byte ID  ]
                      │                 │                       │
                      ▼                 ▼                       ▼
             d(y^1, c_id^1)^2  + d(y^2, c_id^2)^2   ...  = Final Squared Dist

```

The squared Euclidean distance between the unquantized query $y$ and the compressed database vector $x$ is approximated as:

$$\tilde{d}(y, x)^2 = \Vert{}y - q(x)\Vert{}^2 = \sum_{i=1}^m \Vert{}y^i - q_i(x^i)\Vert{}^2$$

#### The In-Memory Lookup Execution:

1. **Lookup Table Construction:** Before scanning the database entries, the query engine computes an explicit $m \times k^*$ matrix. The cell at row $i$, column $j$ stores the exact squared Euclidean distance between the query sub-vector $y^i$ and the sub-centroid $c_j^i$:
$$D_{i,j} = \Vert{}y^i - c_j^i\Vert{}^2$$


2. **Scanning Phase:** For each candidate item in the database, the system loops through its $m$ tracking bytes. It reads the byte value $j = q_i(x^i)$ and fetches the distance directly from row $i$, column $j$ of the lookup table.
3. **Accumulation:** The final distance is a basic summation of these $m$ values.

By shifting the innermost loop of the retrieval engine from heavy floating-point operations to direct CPU cache lookups, ADC drastically improves throughput while running within heavily restricted memory bounds.

---

## 7. The "Garbage In, Garbage Out" Trap in RAG Pipelines

Engineers frequently fall into an optimization trap: spending weeks tweaking IVF cluster sizes, tuning HNSW hyperparameters, and optimizing memory allocation to trim retrieval latency down to single-digit milliseconds—only for the downstream LLM to output completely irrelevant garbage.

In RAG architectures, **computational efficiency does not equal semantic retrieval quality.** Your indexing engine is a mathematical processor that strictly optimizes for spatial distance based on coordinates. It has zero intrinsic understanding of human language or logic.

If your data preparation pipeline is fundamentally flawed, you are simply accelerating the retrieval of irrelevant data.

```
[Raw Document] ──> [Flawed Chunking/Tokenization] ──> [Misaligned Embedding] ──> [Perfect Index] ──> [Ultra-fast Garbage Output]

```

### The Three Weak Links

1. **Flawed Tokenization & Chunking:** If your chunking strategy cuts off text mid-sentence, splits a core concept across two separate blocks, or fails to include structural metadata (like parent headers), the resulting semantic vector loses its context. An index will happily find that vector with blazingly fast speed, but the text payload it carries will be fragmented and useless to the LLM.
2. **Embedding Model Blindness:** Embedding models encode data into a fixed representation based on how they were trained. If you use a general-purpose embedding model on highly domain-specific data (e.g., deep legal contracts, specialized medical terminology, or proprietary codebases), the model will map disparate concepts to similar spatial coordinates, or vice versa. The index will calculate distance perfectly on fundamentally flawed map coordinates.
3. **The Out-of-Distribution Query Problem:** If a user submits a query that uses vocabulary or phrasing completely outside the distribution of the embedding model's training data, the query vector will land in a distorted region of the vector space. The index will efficiently retrieve the nearest neighbors to that distorted point, but those neighbors will be semantically meaningless.

> **The Architectural Rule:** Indexing solves the *engineering problem* of scale and latency. Tokenization, chunking strategies, and embedding fine-tuning solve the *AI problem* of quality and meaning. An elite RAG system requires absolute alignment between both.
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

<!-- 
### **2. Inverted File Indexing (IVF)**

Inverted File Indexing shifts the paradigm from exhaustive search to clustered vector quantization. IVF uses **Voronoi partitioning** to divide the vector space into distinct computational regions.

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

1. **Training Phase:** The system runs a $k$-means clustering algorithm on the dataset to partition it into $k$ clusters, determining a set of centroids $C = \{c_1, c_2, \dots, c_k\}$.
2. **Ingestion Phase:** Every incoming vector $x$ is mapped to its nearest centroid $c_i$ such that the distance $D(x, c_i)$ is minimized. The index stores this as an **inverted list**—a mapping of Centroid ID $\rightarrow$ List of Vector IDs. Mathematically, it places the vector in a Voronoi cell $V_i$:

$$V_i = \{ x \in X \mid D(x, c_i) \le D(x, c_j) \text{ for all } j \neq i \}$$


3. **Query Phase:** The query vector $q$ is first compared against only the $k$ centroids. The system selects the $n$ closest centroids (a hyperparameter called `nprobe`) and executes an exhaustive flat search *only* within those specific Voronoi cells.

* **The Math Trade-off:** By partitioning the data, the search complexity drops from $O(N \cdot d)$ to approximately $O(k \cdot d + \text{nprobe} \cdot \frac{N}{k} \cdot d)$. Increasing `nprobe` improves recall by checking adjacent cells (catching edge-case vectors) but linearly increases compute time.


### **3. HNSW (Hierarchical Navigable Small World)**

While IVF relies on clusters, HNSW relies on graph theory. It builds a multi-layer proximity graph that acts like a probabilistic skip list in high-dimensional space.

1. **Graph Construction:** Vectors are inserted into multiple layers. The bottom layer ($L_0$) contains all vectors. Each vector has a mathematically defined probability of appearing in higher layers, dictated by an exponentially decaying probability distribution:

$$P(l) \propto e^{-l / m_L}$$



where $l$ is the layer number and $m_L$ is a scaling factor.
2. **The Hierarchy:** Top layers contain very few nodes connected by long-range "highways" (large distances). Bottom layers contain dense, short-range connections representing granular local neighborhoods.
3. **Greedy Routing:** When a query vector $q$ arrives, the search starts at the highest, sparsest layer. It evaluates neighbors and greedily jumps to the node mathematically closest to $q$. Once it hits a local minimum in that layer, it drops down to the exact same node in layer $l-1$ and repeats the process until it reaches the ground layer ($L_0$).

* **The Math Trade-off:** HNSW drops search complexity to $O(\log N)$, offering blistering query speeds and elite recall. However, storing the adjacency lists for the complex graph connections requires an immense memory footprint (RAM), often taking up more space than the vectors themselves.


### **4. PQ (Product Quantization)**

Unlike IVF and HNSW—which optimize *how* we search—Product Quantization optimizes *what* we store. It is a mathematical compression technique that shrinks the memory footprint of high-dimensional vectors.

1. **Vector Splitting:** A large, memory-heavy vector $x \in \mathbb{R}^d$ is chopped into $m$ smaller sub-vectors, each with $d/m$ dimensions.

$$x = [x^{(1)}, x^{(2)}, \dots, x^{(m)}]$$


2. **Sub-space Quantization:** For each of the $m$ sub-spaces, the system runs clustering (usually $k$-means) to find $k^*$ sub-centroids. Typically, $k^* = 256$, meaning each sub-centroid can be represented by an 8-bit integer (1 byte).
3. **Encoding:** The original sub-vectors are replaced by the ID (the 1-byte code) of their nearest sub-centroid. A massive 768-dimensional array of 32-bit floats is mathematically approximated as a tiny string of $m$ bytes.

$$x \approx [c_{i_1}^{(1)}, c_{i_2}^{(2)}, \dots, c_{i_m}^{(m)}]$$


4. **Asymmetric Distance Computation (ADC):** At query time, the query vector $q$ is *not* compressed. Instead, $q$ is split into $m$ parts. The system pre-calculates the distances between $q$'s sub-vectors and all possible 256 sub-centroids, storing them in a small lookup table. The total distance is simply the sum of these pre-calculated distances:

$$D(q, x) \approx \sum_{j=1}^{m} D(q^{(j)}, c_{i_j}^{(j)})$$



* **The Math Trade-off:** PQ drastically reduces memory consumption (often by 90% or more) and replaces heavy floating-point arithmetic with lightning-fast $O(m)$ table lookups. The trade-off is a mathematically guaranteed drop in recall due to the lossy compression of the vectors. In massive production systems, it is frequently combined with IVF (as **IVF-PQ**) to achieve scale that would otherwise be impossible on limited hardware.
 -->

### **2. Inverted File Indexing (IVF)**

Go back to the library metaphor from Section 1. Suppose the books are no longer placed randomly. Instead, they are grouped by subject: mathematics books in one area, physics books in another, and biology books somewhere else. A catalogue tells you which area is most relevant to your request.

IVF applies the same idea to vectors. Instead of comparing a query with every vector in the dataset, it divides the vector space into several regions. At query time, it searches only the regions that are likely to contain vectors close to the query.

These regions are usually created using $k$-means clustering. Each region is represented by a centroid.

```text
          Vector Space Divided into Voronoi Cells

          ┌─────────────────┬─────────────────┐
          │     •    •      │       •         │
          │   •   c1        │    •  c2        │
          │     •    •      │    •     •      │
          ├─────────────────┼─────────────────┤
          │       •         │      •   •      │
          │   •  c3         │   •   c4  •     │
          │    •    •       │      •   •      │
          └─────────────────┴─────────────────┘

          ci = centroid of region i
          •  = stored vector
```

The important point is that IVF has three separate stages:

1. learning the regions,
2. assigning vectors to those regions,
3. searching selected regions.

Building the index and querying the index are therefore different operations.

---

#### 2.1 Training Phase: Learning the Centroids

The first step is to run $k$-means clustering and learn $k$ centroids:

$$
C = {c_1, c_2, \dots, c_k}.
$$

Each centroid represents one region of the vector space. Together, the centroids form what is often called the **coarse quantizer**.

The word *coarse* is important. The centroids are not intended to describe every local detail of the dataset. Their purpose is only to divide the space into broad regions so that the system can quickly decide where to search.

##### Why is $k$-means often trained on a sample?

For a large dataset, running $k$-means over every vector can be expensive. Suppose the dataset contains one billion vectors. Processing all one billion vectors during every $k$-means iteration may take a considerable amount of time and require substantial memory and data movement.

However, the centroids only need to approximate the overall distribution of the vectors. If a sufficiently large and representative sample follows approximately the same distribution as the complete dataset, then its $k$-means centroids will usually identify similar dense regions.

For example, suppose the full dataset contains embeddings from three broad topics:

```text
50% technology documents
30% medical documents
20% financial documents
```

A random sample of one million vectors will probably preserve approximately the same proportions:

```text
≈ 500,000 technology vectors
≈ 300,000 medical vectors
≈ 200,000 financial vectors
```

The sample therefore gives $k$-means enough information to locate the main regions without processing the complete dataset.

The reasoning is similar to estimating the average height of a population. Measuring every person is unnecessary if a large and representative sample is available. Increasing the sample size usually makes the estimate more stable, but after some point, processing more examples produces only a small improvement.

This does not mean that any sample will work. The sample must represent the full dataset. A biased sample can produce poor centroids. For example, if medical documents are underrepresented in the training sample, the resulting index may allocate too few centroids to that part of the space.

The sample size should also be large compared with the number of centroids. If $k$ is very large but the training sample is small, some centroids may be trained using very few examples and may not represent meaningful regions.

After training, $k$-means produces the centroids only. The actual dataset vectors have not yet been inserted into the IVF index.

---

#### 2.2 Ingestion Phase: Building the Inverted Lists

Once the centroids have been learned, every database vector $x$ is compared with the centroids and assigned to its nearest one:

$$
\operatorname{assign}(x)
========================

\arg\min_i D(x,c_i).
$$

Here, $D$ is the selected distance function, such as Euclidean distance, cosine distance, or a distance derived from the inner product.

The centroid assignment divides the vector space into Voronoi cells. The cell associated with centroid $c_i$ is

$$
V_i =
\left{
x \in X
\mid
D(x,c_i) \leq D(x,c_j)
\text{ for every } j \neq i
\right}.
$$

Every vector assigned to centroid $c_i$ belongs to the corresponding cell $V_i$.

The index then creates one list for each centroid:

```text
Centroid 1 → [vector_12, vector_58, vector_91, ...]
Centroid 2 → [vector_04, vector_17, vector_73, ...]
Centroid 3 → [vector_08, vector_29, vector_44, ...]
Centroid 4 → [vector_02, vector_31, vector_87, ...]
```

These lists are called **inverted lists**.

##### Why is it called an inverted index?

In the original dataset, the natural mapping is from a vector to its cluster:

```text
vector_12 → centroid_1
vector_58 → centroid_1
vector_17 → centroid_2
```

IVF stores the reverse mapping:

```text
centroid_1 → [vector_12, vector_58, ...]
centroid_2 → [vector_17, ...]
```

The direction of the mapping has been inverted. Instead of asking:

> Which centroid does this vector belong to?

the index allows the system to ask:

> Which vectors belong to this centroid?

This organization is similar to a traditional text inverted index. In a text search engine, the index maps each term to the documents containing it:

```text
"mathematics" → [document_4, document_19, document_71]
```

In IVF, the centroid ID plays a role similar to the term:

```text
centroid_7 → [vector_14, vector_80, vector_103]
```

The centroid does not represent a literal word, but it identifies a region of the vector space.

##### Why store vector IDs?

An inverted list commonly stores vector IDs rather than duplicating the complete application records.

For example:

```text
Centroid 7 → [14, 80, 103]
```

These IDs can be used to locate the corresponding vectors and metadata:

```text
ID 14  → vector values + document identifier + metadata
ID 80  → vector values + document identifier + metadata
ID 103 → vector values + document identifier + metadata
```

This separation has several advantages:

* The original document or application record does not need to be copied into every index structure.
* IDs are smaller and cheaper to store than full documents.
* The same ID can connect the vector index to an external database.
* Metadata filtering can be applied using the associated records.
* Deleted or updated records can be tracked through stable identifiers.

The vector values themselves must still be available during search unless an additional compression method, such as Product Quantization, is used. Depending on the implementation, the vectors may be stored directly inside each inverted list or in a separate contiguous vector storage area referenced by the IDs.

##### What does the IVF index contain in memory?

A basic IVF index typically stores:

1. **The centroids**

   Approximately $k \times d$ floating-point values, where $d$ is the vector dimension.

2. **The inverted-list structure**

   One list for every centroid, containing the vectors or references assigned to it.

3. **Vector IDs**

   IDs that connect search results to the original records.

4. **The vector data**

   Either full-precision vectors, compressed vectors, or references to vectors stored elsewhere.

A simplified memory layout may look like this:

```text
Centroid table
────────────────────────────
centroid_1: [0.12, 0.43, ...]
centroid_2: [0.51, 0.08, ...]
centroid_3: [0.33, 0.77, ...]


Inverted lists
────────────────────────────
list_1:
    ID 12 → [0.11, 0.40, ...]
    ID 58 → [0.14, 0.45, ...]
    ID 91 → [0.09, 0.41, ...]

list_2:
    ID 04 → [0.49, 0.10, ...]
    ID 17 → [0.55, 0.06, ...]
```

The memory overhead added by IVF itself is usually modest: the main additional structures are the centroid table and the boundaries or offsets of the inverted lists. The vectors still occupy most of the memory when they are stored in full precision.

IVF therefore does not automatically compress the vectors. Compression is a separate concern. Methods such as IVF-PQ combine IVF partitioning with Product Quantization to reduce memory usage.

---

#### 2.3 Query Phase: Selecting and Searching Cells

Given a query vector $q$, the search begins by comparing it with all $k$ centroids:

$$
D(q,c_1), D(q,c_2), \dots, D(q,c_k).
$$

The centroids are then ranked by distance to the query.

Instead of searching all cells, the system selects the closest `nprobe` centroids. It then searches the vectors stored in their inverted lists.

```text
Query q
   │
   ├── Compare q with all centroids
   │
   ├── Select the closest nprobe centroids
   │
   ├── Read their inverted lists
   │
   └── Compare q with vectors in those lists
```

The final step is a **flat search over the selected candidates**.

Here, flat search has the same meaning as in the previous Flat Indexing section: the query is compared directly with every candidate vector using the chosen similarity or distance measure.

For example:

* Euclidean distance,
* cosine similarity,
* inner product.

The difference is that Flat Indexing compares the query with all $N$ vectors, whereas IVF performs the same direct comparison only on vectors found in the selected cells.

Suppose the dataset contains one million vectors divided evenly across 1,000 cells. Each cell contains approximately 1,000 vectors.

If `nprobe = 5`, IVF searches approximately:

```text
5 cells × 1,000 vectors = 5,000 vectors
```

instead of all one million vectors.

The centroid search still requires 1,000 comparisons, but this is much smaller than comparing the query with the entire dataset.

---

#### 2.4 The Boundary Problem and the Role of `nprobe`

Searching only the closest cell may miss the true nearest neighbor.

Consider two neighboring cells:

```text
                 Cell A                  Cell B

             cA •                         • cB

                    q • | • x
                        |
                  cell boundary
```

* `q` is the query.
* `x` is the true nearest database vector.
* `q` belongs to Cell A because it is slightly closer to centroid `cA`.
* `x` belongs to Cell B because it is slightly closer to centroid `cB`.

Although `q` and `x` are very close to each other, they are stored in different cells.

If IVF searches only Cell A, vector `x` is never examined.

A numerical example makes this clearer:

```text
Distance between q and x  = 0.03

Distance from q to cA     = 0.40
Distance from q to cB     = 0.42

Distance from x to cA     = 0.41
Distance from x to cB     = 0.39
```

The query is assigned to Cell A because:

$$
D(q,c_A) < D(q,c_B).
$$

The vector is assigned to Cell B because:

$$
D(x,c_B) < D(x,c_A).
$$

However:

$$
D(q,x) = 0.03,
$$

so $x$ may still be the true nearest neighbor of $q$.

This happens because cell membership is determined by distance to the centroids, not by pairwise distance between every query and every stored vector.

A more useful visualization is:

```text
                     Cell A        Cell B

                         cA        cB
                          •        •
                           \      /
                            \    /
                         q • | • x
                             |
                        Voronoi boundary
```

The boundary separates points according to which centroid is closer. It does not guarantee that all nearest-neighbor relationships remain inside a single cell.

This is why IVF uses `nprobe`.

If `nprobe = 1`, only the closest centroid's list is searched.

If `nprobe = 2`, the lists associated with the two closest centroids are searched. In the previous example, searching both Cell A and Cell B allows the system to find $x$.

```text
nprobe = 1
Search Cell A only
Possible result: x is missed

nprobe = 2
Search Cell A and Cell B
Possible result: x is found
```

Increasing `nprobe` reduces the probability of missing neighbors near cell boundaries, but it also increases the number of candidate vectors that must be compared with the query.

|    `nprobe` | Search behaviour             | Expected recall                  | Expected latency |
| ----------: | ---------------------------- | -------------------------------- | ---------------- |
|           1 | Search only the closest cell | Lowest                           | Lowest           |
| Small value | Search several nearby cells  | Higher                           | Moderate         |
| Large value | Search many cells            | High                             | Higher           |
|         $k$ | Search every cell            | Same candidates as Flat Indexing | Highest          |

When `nprobe = k`, every inverted list is searched. At that point, IVF compares the query with every vector, so its candidate-search stage becomes equivalent to Flat Indexing. The index still performs the additional centroid comparison, so IVF provides no search advantage in this configuration.

There is no universal best value for `nprobe`. It depends on:

* the number of centroids,
* the size of the dataset,
* how balanced the cells are,
* the embedding distribution,
* the required recall,
* the acceptable query latency.

It should therefore be selected using measurements on a representative validation set rather than chosen only from a general rule.

---

#### 2.5 Computational Cost

For Flat Indexing, comparing one query with $N$ vectors of dimension $d$ costs approximately:

$$
O(Nd).
$$

For IVF, the query first compares itself with the $k$ centroids:

$$
O(kd).
$$

It then scans the vectors contained in the selected cells.

If the vectors are distributed evenly, each cell contains approximately:

$$
\frac{N}{k}
$$

vectors.

Searching `nprobe` cells therefore examines approximately:

$$
\text{nprobe}\cdot\frac{N}{k}
$$

vectors.

The approximate search cost becomes:

$$
O\left(
kd +
\text{nprobe}\cdot\frac{N}{k}\cdot d
\right).
$$

This estimate assumes that the cells have similar sizes. In practice, this assumption may be false. If some inverted lists are much larger than others, query cost depends on which cells are selected.

A more accurate expression is:

$$
O\left(
kd +
d\sum_{i \in P(q)} |V_i|
\right),
$$

where $P(q)$ is the set of probed cells and $|V_i|$ is the number of vectors stored in cell $i$.

This form shows an important limitation: two queries using the same `nprobe` can have different latency if one query selects small cells and the other selects large cells.

---

#### 2.6 Advantages

##### Reduced candidate set

The main advantage of IVF is that it avoids comparing the query with every vector. For a large dataset, searching a small number of relevant cells can greatly reduce the number of distance computations.

##### Adjustable recall and latency

The value of `nprobe` can be changed at query time. Increasing it searches more cells and usually improves recall. Decreasing it searches fewer cells and usually reduces latency.

The index does not need to be rebuilt when `nprobe` changes.

##### Simple storage structure

A basic IVF index consists mainly of centroids and lists of assigned vectors. It does not require storing graph edges between vectors, as graph-based methods do.

##### Relatively simple construction

After training the centroids, each vector can be inserted by finding its nearest centroid and appending it to the corresponding inverted list.

This construction is generally simpler than building and maintaining a large nearest-neighbor graph.

##### Compatible with compression

IVF is often combined with vector-compression methods. For example, IVF-PQ first uses IVF to select candidate cells and then uses Product Quantization to store and compare compressed vectors.

---

#### 2.7 Limitations

##### Approximate results

When only a subset of cells is searched, IVF may fail to return the true nearest neighbors. A relevant vector may be stored in a cell that was not selected.

The index therefore offers approximate nearest-neighbor search unless all cells are searched.

This does not mean IVF is unsuitable whenever accuracy matters. It means that its recall must be measured for the specific application. In many systems, a small probability of missing a neighbor is acceptable. In others, an exact method or a verification stage may be required.

##### Sensitivity to cell boundaries

Vectors close to one another may be assigned to different cells if they lie on opposite sides of a Voronoi boundary.

This is one of the main sources of recall loss when `nprobe` is small.

Increasing `nprobe` reduces this problem but also increases the amount of computation.

##### Uneven inverted-list sizes

Standard $k$-means minimizes the sum of squared distances between vectors and their assigned centroids. It does not explicitly attempt to create cells containing equal numbers of vectors.

As a result, the index may contain:

```text
Cell 1 → 100 vectors
Cell 2 → 950 vectors
Cell 3 → 24,000 vectors
Cell 4 → 310 vectors
```

A query probing Cell 3 will require many more vector comparisons than a query probing Cell 1.

Large cells can therefore create:

* slower queries,
* unpredictable latency,
* reduced speedup over Flat Indexing,
* more work for frequently accessed regions.

This problem is common when the data distribution contains dense and sparse regions.

##### Dependence on the number of centroids

The number of centroids, often called `nlist`, strongly affects the index.

If `nlist` is too small:

* each cell contains many vectors,
* the candidate set remains large,
* search may not be much faster than Flat Indexing.

If `nlist` is too large:

* centroid search becomes more expensive,
* more training data is required,
* many cells may contain very few vectors,
* nearby vectors may be split across more boundaries,
* a larger `nprobe` may be needed to preserve recall.

The values of `nlist` and `nprobe` must therefore be selected together.

##### Dependence on centroid quality

Poorly trained centroids produce poor partitions.

This may happen when:

* the training sample is too small,
* the sample does not represent the complete dataset,
* $k$-means converges to a poor local solution,
* the vectors were not normalized correctly for the intended metric,
* the data contains structures that $k$-means does not represent well.

Since $k$-means depends on initialization, different training runs may produce different centroids. In practice, methods such as repeated initialization or $k$-means++ are often used to obtain more stable results.

##### Distribution drift

The centroids describe the distribution observed during training.

If the data changes significantly over time, the original partition may become less suitable. For example:

* new document domains may be introduced,
* the embedding model may change,
* one topic may grow much faster than others,
* the language distribution may shift,
* new types of users or queries may appear.

New vectors can still be assigned to the existing centroids, but the cells may become increasingly unbalanced or fail to represent the new distribution well.

At some point, retraining the centroids and rebuilding the index may be necessary.

Changing the embedding model is especially important. Vectors produced by a different embedding model generally live in a different vector space and should not be inserted into an index trained using the previous representation.

##### Update and deletion management

Adding a vector is relatively simple: assign it to a centroid and append it to the corresponding list.

Deletion can be less direct. Some implementations mark an ID as deleted rather than immediately removing and compacting the stored vectors. Repeated deletions may therefore leave unused entries and require periodic maintenance.

Updates may also require removing the old vector and inserting the new vector, possibly into a different cell.

##### Centroid search can become expensive

The query must normally be compared with all $k$ centroids. When $k$ is moderate, this cost is small. When $k$ becomes very large, the centroid search itself may become significant.

Large-scale systems may use hierarchical clustering, tree structures, graph search, or another approximate method to search the centroids.

##### Performance depends on the distance metric

The clustering and search metric should be compatible.

For Euclidean nearest-neighbor search, standard $k$-means naturally uses squared Euclidean distance.

For cosine similarity, vectors are often normalized so that cosine similarity can be related to Euclidean distance on the unit sphere.

If centroid training uses one geometry while candidate ranking uses a substantially different one, the selected cells may not contain the best candidates.

##### More parameters must be tuned

Flat Indexing has little structural tuning. IVF introduces several design choices:

* number of centroids,
* training-sample size,
* number of $k$-means iterations,
* centroid initialization,
* `nprobe`,
* vector normalization,
* distance metric,
* retraining frequency.

These parameters affect recall, construction time, memory usage, and latency. A poorly configured IVF index can be both slower and less accurate than expected.

---

#### 2.8 Summary

IVF reduces search cost by organizing vectors around learned centroids.

During training, $k$-means learns the centroids. During ingestion, each vector is assigned to its nearest centroid and stored in that centroid's inverted list. During querying, the system selects the closest centroids and performs a flat distance search over the vectors contained in their lists.

```text
Training:
dataset sample → k-means → centroids

Ingestion:
vector → nearest centroid → inverted list

Query:
query → closest centroids → selected lists → flat search
```

Its effectiveness depends on whether the centroid partition provides small and useful candidate sets. A small `nprobe` gives faster search but may miss neighbors. A larger `nprobe` improves recall by searching more cells but moves the cost closer to Flat Indexing.

IVF is therefore not a replacement for distance comparison. It is a method for reducing the number of vectors on which that comparison is performed.


### **3. HNSW (Hierarchical Navigable Small World)**

Where IVF thinks in terms of *regions*, HNSW thinks in terms of *paths*. It abandons clustering entirely and instead builds a multi-layer proximity graph that behaves like a probabilistic skip list stretched across high-dimensional space.

If you haven't met skip lists before: imagine a sorted linked list, except every so often a node also gets a pointer that skips far ahead — an "express lane." You start on the express lane, cover most of the distance in a few big hops, then drop down to the regular lane for the final fine-grained steps. HNSW does exactly this, except the "lanes" are layers of a graph and the "distance" being minimized is vector similarity rather than sorted order.

1. **Graph Construction:** Vectors are inserted into multiple layers. The bottom layer ($L_0$) contains all vectors. Each vector has a mathematically defined probability of appearing in higher layers, dictated by an exponentially decaying probability distribution:

$$P(l) \propto e^{-l / m_L}$$

where $l$ is the layer number and $m_L$ is a scaling factor. In practice this means most vectors live only at $L_0$, a shrinking fraction climbs each layer up, and only a handful of "hub" vectors ever make it to the very top — exactly the long-tail structure you'd want for an express-lane system.

2. **The Hierarchy:** Top layers contain very few nodes connected by long-range "highways" (large distances). Bottom layers contain dense, short-range connections representing granular local neighborhoods.

3. **Greedy Routing:** When a query vector $q$ arrives, the search starts at the highest, sparsest layer. It evaluates neighbors and greedily jumps to the node mathematically closest to $q$. Once it hits a local minimum in that layer, it drops down to the exact same node in layer $l-1$ and repeats the process until it reaches the ground layer ($L_0$).

#### The Knobs That Actually Matter: M, efConstruction, efSearch

Three parameters govern almost everything about how an HNSW index behaves, and it's worth knowing what each one is actually doing:

* **`M`** — the maximum number of neighbor connections each node keeps per layer. Higher $M$ means a denser, more richly connected graph — better recall, but more memory and slower construction.
* **`efConstruction`** — how exhaustively the algorithm searches for good neighbors *while building* the graph. Higher values produce a higher-quality graph at the cost of a much slower index build.
* **`efSearch`** — the same idea, but at query time: how many candidates the greedy router keeps in its exploration frontier before settling on an answer. This is your recall/latency dial, playing a role analogous to `nprobe` in IVF.

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
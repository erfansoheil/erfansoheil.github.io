---
layout: post
title: "Indexing in RAG pipeline - Part III"
---


## **4. PQ (Product Quantization)**

IVF and HNSW both answer the same question: *which vectors should I even bother comparing against?* Product Quantization answers a completely different one: *once I've decided to compare against a vector, how cheaply can I store and score it?* IVF and HNSW optimize *how* we search; PQ optimizes *what* we store. It is, at its core, a lossy compression technique for high-dimensional vectors.

1. **Vector Splitting:** A large, memory-heavy vector $x \in \mathbb{R}^d$ is chopped into $m$ smaller sub-vectors, each with $d/m$ dimensions.

$$x = [x^{(1)}, x^{(2)}, \dots, x^{(m)}]$$

2. **Subspace Quantization:** For each of the $m$ subspaces, the system runs clustering (usually $k$-means) to find $k'$ centroids. Typically, $k'=256$, so the **index** of a selected centroid can be represented by an 8-bit integer (1 byte).

3. **Encoding:** The original sub-vectors are replaced by the ID (the 1-byte code) of their nearest centroid. A 768-dimensional array of 32-bit floats can therefore be represented approximately by a compact code of $m$ bytes.

$$x \approx [c_{i_1}^{(1)}, c_{i_2}^{(2)}, \dots, c_{i_m}^{(m)}]$$

4. **Asymmetric Distance Computation (ADC):** At query time, the query vector $q$ is *not* compressed. Instead, $q$ is split into $m$ parts. The system pre-calculates the distances between $q$'s sub-vectors and all possible 256 centroids, storing them in a small lookup table. The total distance is simply the sum of these pre-calculated distances:

$$D(q, x) \approx \sum_{j=1}^{m} D(q^{(j)}, c_{i_j}^{(j)})$$

We'll formalize all four of these steps rigorously — codebooks, the full ADC derivation, memory arithmetic — in Section 7. For now, the intuition is enough to understand the trade-off.

#### Advantages

* **Dramatic memory savings:** Compressing a 3,072-byte float vector down to a 64-byte code gives a $48\times$ reduction in the per-vector representation size, worked out in full in Section 7. At very large scale, this reduction can determine whether the vector collection fits within the available memory budget.
* **Cheap approximate distance computation:** ADC replaces repeated full-dimensional distance calculations with $m$ lookup-table reads and an accumulation, reducing the amount of arithmetic required per candidate.
* **Composable:** PQ doesn't compete with IVF or HNSW; it *combines* with them. IVF-PQ is a standard production pattern precisely because the two solve orthogonal problems — IVF narrows down *which* vectors to check, PQ shrinks the cost of storing and checking each one.

#### Flaws

* **It is lossy:** PQ introduces quantization error, so the distance computed at query time is an approximation of the original distance. This approximation can change the nearest-neighbor ranking and therefore reduce recall compared with exact search.
* **Codebooks need representative training data:** The $k$-means codebooks are only as good as the vectors they were trained on. If the distribution of indexed vectors drifts substantially away from that training distribution, quantization error can increase and retrieval quality can degrade. Here the approximation affects the *scoring* of candidates rather than only their coarse routing.
* **Choosing $m$ creates a memory–accuracy trade-off:** with 8-bit subquantizers, each vector requires approximately $m$ bytes. A smaller $m$ provides stronger compression, but each subquantizer must represent a higher-dimensional subspace, which generally increases quantization error. A larger $m$ uses more bytes and requires more lookup operations, but usually provides a more accurate representation of the original vector.

**The Math Trade-off:** PQ can drastically reduce memory consumption and replace many full-dimensional distance computations with $O(m)$ lookup-and-accumulate operations. The price is quantization error: compressed vectors no longer preserve the original distances exactly, which can reduce nearest-neighbor recall. In large-scale systems, PQ is frequently combined with IVF as **IVF-PQ**, because IVF reduces the candidate set while PQ reduces the storage and scoring cost of each candidate. A further refinement, Optimized Product Quantization (OPQ), learns a rotation of the vector space before the subspace split to reduce quantization error.



So far, we have investigated three archetypes: IVF bets on **partitioning**, HNSW bets on **graph traversal**, and PQ bets on **compression** instead of pruning. None of them is strictly "better" in the abstract — which is exactly the question the next section is actually about.







<iframe src="./asset/pq.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>



## **5. Indexing or Non-Indexing: This is the Question**

There is no universally "best" indexing method. There are only **different trade-offs**. The choice between exact search and an ANN index should be justified by data volume, latency, throughput, recall, memory, and update requirements.


For example, when selecting or tuning an index using popular libraries like **FAISS** (Facebook AI Similarity Search), **HNSWlib**, **Annoy**, or embedded engines like **LanceDB**, you are forced to balance four competing directions:

```
                  [1] Search Performance
                             ▲
                             │
 [4] Memory Consumption ◄────┼────► [2] Recall Accuracy
                             │
                             ▼
                  [3] Build/Update Time

```

1. **Search Performance:** How low is query latency, and how many queries per second can the system sustain?
2. **ANN Recall:** What fraction of the exact mathematical top-$k$ neighbors does the approximate index recover? A standard definition is $\mathrm{Recall@}k = \|ANN_k(q) \cap Exact_k(q)\|/k$. This measures index approximation quality, not semantic relevance to the user's question.
3. **Build Time & Dynamic Updates:** How expensive is index construction, and how efficiently can new vectors be inserted or existing data updated? HNSW must maintain graph connections during insertion, while a trained IVF index can assign new vectors to its existing centroids. Significant distribution drift may eventually make IVF retraining desirable.
4. **Memory Consumption:** How much memory does the index require beyond the original vectors? HNSW stores graph connections in addition to vector data, while PQ-based indexes reduce vector-storage cost by replacing full-precision vectors with compact codes.

### The "No-ANN-Index" Choice: When Flat Search Wins

There are many architectural scenarios where you do not need an **approximate** index at all. An exhaustive `Flat` index in FAISS compares the query against every stored vector and returns the exact top-$k$ neighbors under the chosen distance function. It can be preferable under the following conditions:

* **Small datasets or few queries:** For sufficiently small datasets, or when only a limited number of searches will be performed, exhaustive search can be preferable because it avoids ANN approximation and index-construction overhead. The crossover point depends on dataset size, vector dimension, hardware, query batching, and latency/throughput requirements rather than a universal vector-count threshold.
* **Exact nearest-neighbor requirement:** If the application requires the exact top-$k$ vectors under the chosen distance function, exhaustive Flat search is the appropriate baseline. This guarantees exact vector-space nearest neighbors, but it does **not** guarantee that those neighbors are semantically relevant to the user's question.
* **Frequent updates:** Flat storage has very little index-maintenance overhead because inserting a vector does not require updating graph connections or retraining a clustering model. HNSW insertion requires graph maintenance, while a trained IVF index assigns new vectors to existing centroids. The appropriate choice therefore depends on both update frequency and query workload.


## **6. Concrete Engineering Implementations with FAISS**

The following code templates demonstrate how to implement the discussed index archetypes using **FAISS** (Facebook AI Similarity Search) in Python. We use $d=768$ as a convenient example dimension, matching models such as BERT-base, and simulate a dataset of 100,000 vectors. These random vectors are used only to demonstrate the FAISS APIs; they should not be used to draw conclusions about retrieval quality on real embedding distributions.

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

### **Type A: Flat (Brute-Force) Index**

The baseline index. It performs no compression and executes an exhaustive $O(N \cdot d)$ calculation.

```python
print("--- Building IndexFlatL2 ---")
build_start = time.time()
index_flat = faiss.IndexFlatL2(d)            # Exact L2 distance index
print(f"Is trained: {index_flat.is_trained}") # True by default (no training needed)
index_flat.add(xb)                            # Add vectors directly
flat_build_ms = (time.time() - build_start) * 1000

search_start = time.time()
D_flat, I_flat = index_flat.search(xq, k)
flat_search_ms = (time.time() - search_start) * 1000

print(f"Flat Build Time: {flat_build_ms:.3f} ms")
print(f"Flat Search Time: {flat_search_ms:.3f} ms")

```

### **Type B: Inverted File (IVF-Flat) Index**

Uses $k$-means to cluster the space into Voronoi cells. It requires explicit training on the data distribution before vectors can be mapped.

```python
print("\n--- Building IndexIVFFlat ---")
nlist = 100                                  # Number of Voronoi cells (clusters)
quantizer = faiss.IndexFlatL2(d)             # Coarse quantizer used to assign vectors to cells

index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

build_start = time.time()
print(f"Is trained before: {index_ivf.is_trained}")
index_ivf.train(xb)                          # Learn the coarse centroids
index_ivf.add(xb)
ivf_build_ms = (time.time() - build_start) * 1000
print(f"Is trained after: {index_ivf.is_trained}")

# Adjust search-time parameters
index_ivf.nprobe = 10                        # Probe the 10 closest coarse cells

search_start = time.time()
D_ivf, I_ivf = index_ivf.search(xq, k)
ivf_search_ms = (time.time() - search_start) * 1000

print(f"IVF Build Time: {ivf_build_ms:.3f} ms")
print(f"IVF Search Time: {ivf_search_ms:.3f} ms")

```

### **Type C: HNSW (Hierarchical Navigable Small World) Index**

A graph-based index that can offer high recall and low search latency, at the cost of additional memory for graph connections.

```python
print("\n--- Building IndexHNSWFlat ---")
M = 32                                       # Number of bi-directional links per node

index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.hnsw.efConstruction = 64          # Higher = more accurate graph building, slower ingest
index_hnsw.hnsw.efSearch = 32                # Higher = more accurate search, higher latency

build_start = time.time()
index_hnsw.add(xb)                           # No training phase needed for HNSW
hnsw_build_ms = (time.time() - build_start) * 1000

search_start = time.time()
D_hnsw, I_hnsw = index_hnsw.search(xq, k)
hnsw_search_ms = (time.time() - search_start) * 1000

print(f"HNSW Build Time: {hnsw_build_ms:.3f} ms")
print(f"HNSW Search Time: {hnsw_search_ms:.3f} ms")

```

### **Type D: Composite Production Index (IVF + PQ via Index Factory)**

For larger-scale search, IVF and PQ can be combined. IVF first assigns vectors to coarse cells, and PQ then compresses the **residual information relative to the coarse centroid**. In FAISS this family is commonly referred to as **IVF-PQ / IVFADC**.

```python
print("\n--- Building Composite IVF+PQ via Index Factory ---")
# Factory string syntax: "IVF[centroids],[fine-quantizer]"
# "PQ64" means split the 768-dim vector into 64 sub-vectors of 12 dimensions each.
factory_string = "IVF256,PQ64"

index_composite = faiss.index_factory(d, factory_string)

build_start = time.time()
print("Training composite index...")
index_composite.train(xb)
index_composite.add(xb)
ivfpq_build_ms = (time.time() - build_start) * 1000

# Set runtime cluster probe depth
faiss.extract_index_ivf(index_composite).nprobe = 16

search_start = time.time()
D_ivfpq, I_ivfpq = index_composite.search(xq, k)
ivfpq_search_ms = (time.time() - search_start) * 1000

print(f"IVF-PQ Build Time: {ivfpq_build_ms:.3f} ms")
print(f"IVF-PQ Search Time: {ivfpq_search_ms:.3f} ms")
print(f"Total Vectors Indexed: {index_composite.ntotal}")

```

### **Comparing Approximate Results with the Exact Baseline**

`IndexFlatL2` returns the exact top-$k$ neighbors under L2 distance, so its output can be used as ground truth for measuring **ANN Recall@$k$**. For one query,

$$
\mathrm{Recall@}k = \frac{|ANN_k(q) \cap Exact_k(q)|}{k}.
$$

For a batch of queries, we average this value across all queries.

```python
def recall_at_k(I_true, I_approx, k):
    scores = []
    for true_row, approx_row in zip(I_true, I_approx):
        overlap = len(set(true_row[:k]) & set(approx_row[:k]))
        scores.append(overlap / k)
    return float(np.mean(scores))

ivf_recall = recall_at_k(I_flat, I_ivf, k)
hnsw_recall = recall_at_k(I_flat, I_hnsw, k)
ivfpq_recall = recall_at_k(I_flat, I_ivfpq, k)

print("\nIndex      Build ms    Search ms    Recall@k")
print(f"Flat       {flat_build_ms:8.3f}    {flat_search_ms:9.3f}    1.000")
print(f"IVF        {ivf_build_ms:8.3f}    {ivf_search_ms:9.3f}    {ivf_recall:.3f}")
print(f"HNSW       {hnsw_build_ms:8.3f}    {hnsw_search_ms:9.3f}    {hnsw_recall:.3f}")
print(f"IVF-PQ     {ivfpq_build_ms:8.3f}    {ivfpq_search_ms:9.3f}    {ivfpq_recall:.3f}")
```

This comparison is more informative than reporting latency alone. A real benchmark should also measure **memory usage** (for example, process RSS or serialized index size) and use enough queries for stable latency/throughput statistics. Parameters such as `nprobe` for IVF and `efSearch` for HNSW should be swept to produce recall-versus-latency curves rather than judged from a single configuration.

---

## **7. Mathematical Breakdown of Product Quantization (PQ)**

Product Quantization (PQ) is a lossy compression framework that can greatly reduce per-vector storage while still supporting efficient approximate distance computation over compact codes.

### **Step 1: Subspace Decomposition**

Let a high-dimensional database vector $x \in \mathbb{R}^d$ be partitioned into $m$ **disjoint** lower-dimensional sub-vectors $x^1, x^2, \dots, x^m \in \mathbb{R}^{d^*}$, where:

$$d^* = \frac{d}{m}$$

The total vector space can be mathematically defined as the Cartesian product of these lower-dimensional subspaces:

$$\mathbb{R}^d = \mathbb{R}^{d^*} \times \mathbb{R}^{d^*} \times \dots \times \mathbb{R}^{d^*}$$

Visually, a 768-dimensional vector divided by $m=64$ yields 64 structural slices, each containing exactly 12 scalar fields.

### **Step 2: Codebook Generation via $k$-means**

For each subspace $i \in \{1, \dots, m\}$, a dedicated $k$-means clustering algorithm is executed on a training slice of the database:

$$\mathcal{X}^i = \{x_1^i, x_2^i, \dots, x_N^i\}$$

The optimization objective minimizes the intra-cluster sum of squared errors to resolve a discrete set of $k^*$ centroids:

$$\min_{C^i} \sum_{x^i \in \mathcal{X}^i} \min_{c \in C^i} \Vert{}x^i - c\Vert{}^2$$

where $C^i = \{c_1^i, c_2^i, \dots, c_{k^*}^i\}$ is the sub-codebook for subspace $i$.

In standard configurations, $k^*$ is set to 256 ($2^8$), meaning each centroid index can be represented using exactly **1 byte (8 bits)** of information. The global codebook $\mathcal{C}$ is the total composition:

$$\mathcal{C} = C^1 \times C^2 \times \dots \times C^m$$

This Cartesian product explains the name **Product Quantization**. Instead of learning one enormous codebook directly in the full $d$-dimensional space, PQ learns $m$ smaller codebooks across coordinate subspaces. If each sub-codebook contains $k^*$ centroids, their combinations implicitly represent $(k^*)^m$ possible reconstructed vectors, while each stored vector requires only $m$ centroid IDs when $k^*=256$.

### **Step 3: Quantization Mapping**

For each sub-vector $x^i$, PQ stores the index of its nearest centroid:

$$
j_i = \arg\min_{j \in \{1, \dots, k^*\}} \Vert{}x^i-c_j^i\Vert{} .
$$

The compressed representation of $x$ is therefore the integer code

$$
\operatorname{code}(x)=(j_1,j_2,\dots,j_m).
$$

It is also useful to define the approximate reconstructed vector

$$
\hat{x}=[c_{j_1}^1,c_{j_2}^2,\dots,c_{j_m}^m].
$$

If each codebook contains $256=2^8$ centroids, every index $j_i$ requires one byte. Therefore, excluding shared codebooks and vector IDs, the per-vector code size is

$$
\text{Compressed Size}=m\times1\text{ byte}.
$$

For a 768-dimensional FP32 vector, the original storage is $768\times4=3072$ bytes. If $m=64$, the PQ code uses **64 bytes**, a $48\times$ reduction in per-vector representation size.

### **Step 4: Asymmetric Distance Computation (ADC)**

To avoid reconstructing every database vector during retrieval, PQ uses **Asymmetric Distance Computation (ADC)**. The query vector $y$ remains unquantized at full floating-point precision, while each database vector is represented by its compact code $(j_1,\dots,j_m)$.

```
 Query (y)      [ y^1 (FP32) ]    [ y^2 (FP32) ]    ...    [ y^m (FP32) ]
                      │                 │                       │
                Distance Table    Distance Table          Distance Table
                      ▼                 ▼                       ▼
 PQ code        [  Byte ID  ]     [  Byte ID  ]     ...    [  Byte ID  ]
                      │                 │                       │
                      ▼                 ▼                       ▼
             d(y^1, c_id^1)^2  + d(y^2, c_id^2)^2   ...  = Final Squared Dist

```

The squared Euclidean distance between the unquantized query $y$ and the compressed database vector $x$ is approximated as:

$$\tilde{d}(y,x)^2 = \Vert{}y-\hat{x}\Vert{}^2 = \sum_{i=1}^m \Vert{}y^i-c_{j_i}^i\Vert{}^2$$

#### **The In-Memory Lookup Execution:**

1. **Lookup Table Construction:** Before scanning the database entries, the query engine computes an explicit $m \times k^*$ matrix. The cell at row $i$, column $j$ stores the exact squared Euclidean distance between the query sub-vector $y^i$ and the centroid $c_j^i$:
$$D_{i,j} = \Vert{}y^i - c_j^i\Vert{}^2$$


2. **Scanning Phase:** For each candidate item, the system reads its stored centroid index $j_i$ for each subspace and fetches $D_{i,j_i}$ from the lookup table.
3. **Accumulation:** The final distance is a basic summation of these $m$ values.

ADC replaces repeated full-dimensional vector arithmetic with compact lookup-table reads and accumulation, which can substantially reduce scoring cost and memory bandwidth requirements.


## **8. The "Garbage In, Garbage Out" Trap in RAG Pipelines**

Engineers frequently fall into an optimization trap: spending weeks tweaking IVF cluster sizes, tuning HNSW hyperparameters, and optimizing memory allocation to trim retrieval latency down to single-digit milliseconds—only for the downstream LLM to output completely irrelevant garbage.

In RAG architectures, **computational efficiency does not equal semantic retrieval quality.** Your indexing engine is a mathematical processor that strictly optimizes for spatial distance based on coordinates. It has zero intrinsic understanding of human language or logic.

If your data preparation pipeline is fundamentally flawed, you are simply accelerating the retrieval of irrelevant data.

```
[Raw Document] ──> [Poor Chunking] ──> [Misaligned Embedding] ──> [Perfect Index] ──> [Irrelevant Retrieved Context]

```

### **The Three Weak Links**

1. **Poor Chunking and Context Boundaries:** If the chunking strategy cuts text at poor boundaries, separates a core concept across multiple chunks, or drops useful structural metadata such as parent headers, the resulting embedding may represent incomplete context. The index can retrieve that vector efficiently, but the associated text may still be fragmented or unhelpful to the LLM.
2. **Embedding Model Mismatch:** Embedding models encode data into a fixed representation based on how they were trained. If you use a general-purpose embedding model on highly domain-specific data (e.g., deep legal contracts, specialized medical terminology, or proprietary codebases), the model will map disparate concepts to similar spatial coordinates, or vice versa. The index will calculate distance perfectly on fundamentally flawed map coordinates.
3. **Query–Corpus Mismatch:** Users may express a concept differently from the indexed documents—for example through different terminology, abbreviations, languages, or domain-specific phrasing. Even when the relevant information exists in the corpus, the embedding model may place the query and the relevant chunk farther apart than expected, reducing retrieval quality.

Indexing should therefore be viewed primarily as a **candidate-generation stage**, not the entire retrieval system. A practical RAG pipeline may retrieve a relatively broad set of candidates using ANN search and then apply metadata filtering or a more expensive reranker to select the chunks that are actually most relevant to the query.

`Query → ANN Retrieval → Candidate Set → Filtering / Reranking → Top-k Context → LLM`

> **The Architectural Rule:** Indexing primarily solves the engineering problem of searching large vector collections efficiently. Chunking, embedding quality, metadata, filtering, and reranking determine whether the retrieved candidates are actually useful. A fast index cannot compensate for a poor representation or retrieval pipeline.

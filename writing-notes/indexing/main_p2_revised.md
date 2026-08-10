---
layout: post
title: "Indexing in RAG pipeline - Part II"
---


## **3. HNSW (Hierarchical Navigable Small World)**

The Hierarchical Navigable Small World (HNSW) algorithm is currently one of the most efficient and widely used structures for approximate nearest neighbor (ANN) search in high-dimensional spaces. To deeply understand how HNSW works, we must deconstruct it into its two foundational concepts: the probabilistic hierarchy of a **Skip List** (which relies on a total order) and the decentralized graph routing of **Navigable Small Worlds (NSW)** for high-dimensional vector spaces. 



HNSW adapts two key ideas from these structures: **hierarchical sparsification**, which enables coarse long-range movement, and **proximity-graph navigation**, which gives the search a geometric direction toward the query. With these ideas, the number of distance evaluations required by HNSW is designed to grow approximately logarithmically with dataset size under typical conditions.


### **3.1. The Skip List**

Imagine a long sorted linked list. Every element knows only its immediate successor. If we are searching for a target value, the ordering tells us whether to continue moving forward or stop. We do not need to search in arbitrary directions, but we still have to advance one node at a time.

For a short list, this is acceptable. For millions of elements, however, it becomes inefficient. Reaching a value near the end of the list may require visiting almost every element that comes before it. Therefore, searching an ordinary linked list has a worst-case time complexity of $O(n)$.

The natural idea is to skip over some elements instead of visiting every node. However, a single system of large jumps is not sufficient. If a jump takes us beyond the target, a singly linked list does not provide an efficient way to move backward and recover. We need large jumps when we are far from the target, followed by progressively smaller and more precise steps as we approach it.

A skip list provides exactly this mechanism by representing the same ordered collection at multiple levels. The bottom level, $L_0$, contains every element. Each higher level contains only a subset of the elements in the level below it.

The upper levels therefore act as express lanes. They allow the search to cross large parts of the list quickly. When the next jump would pass the target, the search descends to a denser level, where the jumps are shorter and more precise. Eventually, it reaches the complete bottom level, where the exact target can be found.

The structure can be summarized as follows:

* The bottom level provides completeness and precision.
* The upper levels provide speed.
* Searching alternates between moving right and moving down.

Here is an interactive visualization of how we traverse a Skip List. Notice how the algorithm moves right as long as the next value is less than or equal to our target. As soon as it overshoots, it drops down a layer to find a more granular path.

<iframe src="./asset/skiplist-interactive.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>


Basically, that was the whole **skip list** algorithm. However, we did not mention **how to generate** these layers. In theory, we use a probabilistic method to assign each item to specific layers. In the following, we explain this method in more detail.

#### **Probabilistic Construction: The Coin Flip**

A skip list is constructed incrementally. Whenever a new node is inserted, it is always placed in the base level $L_0$. A randomized procedure then determines whether the node should also appear in the higher levels.

Let $p$ be the probability that a node is promoted to the next level.

1. Every node is inserted into $L_0$.
2. With probability $p$, it is also promoted to $L_1$.
3. If it reaches $L_1$, it is promoted to $L_2$ with probability $p$ again.
4. This process continues until the first unsuccessful promotion.

Equivalently, we may imagine repeatedly flipping a biased coin. A successful flip promotes the node by one level, while an unsuccessful flip stops the process.

The probability that a node reaches level $k$ is $p^k.$

The number of nodes therefore decreases exponentially as we move upward. This produces sparse upper levels containing long-range shortcuts and dense lower levels containing shorter, more precise connections.


The value of $p$ directly controls the shape of the skip list. A larger value of $p$ causes more nodes to be promoted, while a smaller value causes fewer nodes to be promoted. For example, suppose that the list contains $1{,}000$ elements.

If $p=\frac{1}{2}$, the expected level sizes are approximately

$$
1000,\ 500,\ 250,\ 125,\ 62,\ldots
$$

If $p=\frac{1}{4}$, the expected level sizes are approximately

$$
1000,\ 250,\ 62,\ 15,\ 4,\ldots
$$


The probabilistic construction is one of the main strengths of a skip list because it creates a balanced hierarchy without requiring explicit rebalancing. The trade-off is that the exact structure is nondeterministic. Even when two skip lists contain the same elements and use the same probability $p$, their layer assignments may differ. Therefore, logarithmic search complexity is an **expected** property rather than a worst-case guarantee.

#### **A Mathematical POV**

From a mathematical point of view, it is useful to estimate the expected height of a skip list and its expected search complexity. Both depend on the promotion probability $p$ and the total number of items $n$. Since the probability that a node reaches level $k$ is $p^k$, the expected number of nodes at that level is approximately $np^k$. We can estimate the height $h$ by asking when the expected number of nodes in a level becomes approximately one:

$$
np^h \approx 1 \Rightarrow p^h \approx \frac{1}{n}.
$$

Taking logarithms gives

$$
h \approx \log_{1/p}n.
$$

Therefore, the expected height of a skip list grows logarithmically with the number of elements. 

In addition, at each level there are several possible horizontal movements before descending. The number of horizontal movements depends on the number of items in the level, and we know that this number depends on $p$, or more precisely on $\frac{1}{p}$.

In other words, a common intuitive approximation is

$$
\text{Expected horizontal movements per level}\approx\frac{1}{p}.
$$

Since the expected number of levels is approximately $\log_{1/p}n$, the **expected search cost** can be described as

$$
\text{Expected search cost}
\approx
\frac{1}{p}\log_{1/p}n.
$$

When $p$ is treated as a fixed constant independent of $n$, both $\frac{1}{p}$ and the logarithm base contribute only constant factors. Therefore, the expected search complexity is $O(\log n)$.

The word **expected** in **expected search cost** is important. The skip list does not **guarantee** logarithmic search time for every possible random construction. Its worst-case search time remains $O(n)$.


So far, we have discussed the **expected height** and **expected search complexity** of a skip list. However, there is an assumed fact about a skip list: its items belong to a **totally ordered set**. As we saw in the figure, all items in the skip list can be compared with each other individually. In mathematical terms, the items in the skip list come from a **totally ordered** set.

A set is totally ordered when any two elements can be compared. For two different elements $a$ and $b$, either $a<b$ or $b<a$. This property provides a **direction** for the search. If the next value is smaller than the target, we continue moving right. If it is larger, we descend to a more precise level.


Multidimensional vectors, however, do not have an equally natural order that represents their geometry. Therefore, creating a skip list for a set of multidimensional vectors is challenging. This is one of the main limitations of extending skip lists to higher dimensions, where most modern embedding-vector use cases lie.



### **3.2. Navigable Small Worlds (NSW)**

In our previous discussion, we deconstructed the Skip List—the probabilistic data structure that inspires HNSW's hierarchical organization. A Skip List requires its elements to belong to a totally ordered set. Although multidimensional vectors can be artificially ordered, such an ordering generally does not preserve their geometric neighborhood structure, making it unsuitable for nearest-neighbor search. 

In modern AI, we often deal with embeddings: dense, high-dimensional vectors representing semantic information. Although such vectors can be assigned an artificial total order, that order does not generally preserve nearest-neighbor geometry.

To solve this, HNSW replaces the 1D linked lists at each layer with a Navigable Small-World (NSW) graph. To truly understand HNSW, we must put the hierarchy aside for a moment and examine the mathematics and algorithms of a single, flat Small-World Network.

#### **Small-World Network**

A **small-world network** is a graph with two main properties:

1. Nodes are mostly connected to nearby nodes, so local neighborhoods are highly clustered.
2. Even distant nodes can usually be reached through only a few steps.

A common example is a social network: people mostly know others in their local group, but a few connections between different groups create **shortcuts** across the network.

If $d_G(u,v)$ is the shortest-path distance between two nodes, a small-world network has a small average path length while keeping strong local clustering.

The key idea is simple: **a few long-range connections can greatly shorten paths across the whole graph**.

This is the idea formalized by the Watts-Strogatz model.



#### **The Watts-Strogatz Model**

In 1998, Duncan Watts and Steven Strogatz formalized this transition using a rewiring parameter. To avoid confusing it with the skip-list promotion probability, we denote the **rewiring probability** by $\beta$.

* **Initialization:** Start with a regular ring lattice of $N$ nodes. For an even $k$, every node is connected to its $k/2$ nearest neighbors on each side.
* **Rewiring:** For each node, consider its $k/2$ edges in one direction around the ring. With probability $\beta$, rewire the far endpoint of each considered edge to another randomly selected node, while avoiding self-loops and duplicate edges.

The important point is that rewiring changes one endpoint while preserving the total number of edges. 

<iframe src="./asset/swn.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>


#### **Some Properties of the Watts-Strogatz Model**

**Role of $\beta$**

The parameter $\beta$ controls how much randomness is introduced into the graph. If $\beta=0$ no edge is rewired. The graph remains a completely regular ring lattice.

For $k\geq2$, every node $i$ is connected at least to $ (i-1)\bmod N$ and $(i+1)\bmod N.$

Therefore the graph contains the cycle

$$
0\rightarrow1\rightarrow2\rightarrow\cdots
\rightarrow N-1\rightarrow0.
$$

Since this cycle contains every node, there exists a path between every pair of vertices. Hence the graph is **path connected**.

The problem, however, is that reaching a distant node may require traversing many intermediate nodes. 
 

When $\beta$ becomes positive, some local edges are replaced by longer-range connections. A newly rewired edge can create a shortcut between distant parts of the graph.

Even a relatively small number of such shortcuts can significantly decrease the average shortest-path length.

This is the regime in which the characteristic **small-world behavior** appears:

$$
\text{high local clustering}
+
\text{small global path length}.
$$

Importantly, $\beta$ does not determine the number of edges. It only determines which edges are rewired. In fact, with a fixed $k$, changing $\beta$ does not change the number of edges.

When $\beta=1,$ every edge considered by the rewiring procedure is rewired. The resulting graph becomes highly random. 

**The Number of Edges**

Before rewiring, every node has degree $k$. Using the handshake lemma,

$$
\sum_{v\in V}\deg(v)=2|E|.
$$

Since there are $N$ vertices and every vertex initially has degree $k$, $Nk=2\|E\|$. Therefore, $\|E\|=\frac{Nk}{2}$.

Now consider one rewiring operation. Suppose an edge $(u,v)$ is removed and replaced by $(u,w).$

Then

$$
E' = \left(E\setminus\{(u,v)\}\right) \cup \{(u,w)\},
$$

and therefore

$$
|E'| = |E|-1+1 = |E|.
$$

Thus, changing $\beta$ does not change the number of edges. What changes is the **topology** of the graph.


**The Effects of $k$ and $\beta$**

The parameters $k$ and $\beta$ play fundamentally different roles. The parameter $k$ controls the **density of the graph**. Increasing $k$ gives every node more neighbors and increases the total number of edges:

$$
|E|=\frac{Nk}{2}.
$$

The parameter $\beta$, on the other hand, controls the **randomness of the graph**. Increasing $\beta$ does not create more edges. Instead, it replaces increasingly many local connections with non-local ones.

Therefore,

$$
\boxed{
k \rightarrow \text{graph density}
}
$$

while

$$
\boxed{
\beta \rightarrow \text{graph randomness}.
}
$$


**What Makes the Network "Small World"?**

The central phenomenon is the transition between two extremes.

When $\beta=0,$ the network is highly regular: **high clustering but  relatively long paths**

When $\beta$ is small but positive, a few shortcuts appear: **high clustering + short paths**. This is the **small-world** network. 

When $\beta$ approaches $1$, the graph becomes increasingly random: **short paths but weaker local regularity**

Conceptually,

$$
\boxed{
\text{Regular lattice}
\xrightarrow{\quad \beta\uparrow\quad}
\text{Small-world regime}
\xrightarrow{\quad \beta\uparrow\quad}
\text{Random graph}.
}
$$


Thus, the essential idea behind a small-world network is not simply randomness. It is the combination of **local structure and a small number of long-range shortcuts** that dramatically improve global connectivity.


A **small-world** structure gives us relatively short paths between different regions of a graph. However, this introduces another question for retrieval: **how do we actually discover those short paths?** The small-world property alone does not guarantee navigability. A short path may exist, while a search algorithm using only local information may still be unable to find it efficiently.

#### **From Small World to Navigable Small World**

A **small-world graph** is characterized by relatively short paths between nodes on average. However, this does not mean that we can easily **find** those paths using only local information.

This is the problem of **navigability**. Suppose we are at node $A$ and want to reach a query vector $q$. We do not examine the entire graph. We only look at the neighbors of our current node and ask: **which neighbor is closer to $q$?** 

This gives the basic greedy rule: if the current node is $v$, choose the neighbor $u$ that minimizes $d(u,q)$ and move there. So the search behaves roughly as: $A \rightarrow v_1 \rightarrow v_2 \rightarrow \cdots \rightarrow q$.

For this to work well, the graph must contain edges that guide us through the space. A purely random shortcut may shorten the graph theoretically, but it may not be useful for deciding where to move next. So, in order to make the graph navigable, we cannot rely only on the Watts-Strogatz rewiring probability $\beta$ to construct the graph. For vector search, we therefore need graph connections that are related to the **geometry** of the vector space, so that **distance** to the query provides useful local information about where to move next.

**Kleinberg's Idea**

Kleinberg showed this distinction clearly using a spatial small-world model. Instead of choosing every long-range connection uniformly at random, the probability of connecting two nodes depends on their distance.

In a $d$-dimensional lattice, the navigable case occurs when the probability of a long-range connection roughly follows $P(u,v)\propto 1/d(u,v)^d$. 

This creates connections at different distance scales: many short connections, fewer medium-distance connections, and some long-range connections.

This is useful for greedy search because the search can make large movements when it is far from the target and increasingly smaller movements as it gets closer.

The important remark is therefore: **a small world needs short paths; a navigable small world needs short paths that can be discovered using local information.**

Kleinberg's model gives us an important theoretical lesson: navigability depends not only on having long-range edges, but also on how those edges are distributed across distance scales. Practical vector-search structures such as NSW do not explicitly construct their graphs using Kleinberg's probability rule. Instead, they obtain navigability through incremental, proximity-based graph construction.

**Navigable Small World for Vector Search**

For vector search, every node represents a vector and the graph is built using a distance function such as Euclidean or cosine distance.

The NSW algorithm builds this graph **incrementally**, rather than starting from a ring and randomly rewiring edges.

The basic construction is:

1. Start with a small graph.
2. Insert a new vector $v$.
3. Search the existing graph for vectors close to $v$.
4. Connect $v$ to several of those nearby vectors.
5. Keep the old connections and repeat for the next vector.

The important part is that the graph is based on **proximity**. Nearby vectors tend to become connected, which creates useful local structure. However, something interesting happens because the graph grows over time. Early in construction, the graph contains only a few points.  Two points connected during an early stage may later cease to be close neighbors relative to newly inserted points. Their original edge can nevertheless remain, creating a longer-range connection in the final graph.

As a result, the graph naturally contains a mixture of edge lengths: short local edges and some longer edges connecting different regions of the space. Malkov et al. describe NSW as preserving older links produced during the incremental approximation of the proximity graph, which contributes to its small-world navigation behavior. So NSW does **not** explicitly calculate Kleinberg's probability $P(u,v)$. Instead, navigability emerges from two simple ideas: **graph growth** and **proximity-based connections**.

**Searching an NSW Graph**

Suppose the query vector is $q$ and we start from some entry node $v$.

At each step:

1. Look at the neighbors of $v$.
2. Compute their distances to $q$.
3. Move toward neighbors that are closer to $q$.
4. Repeat until no sufficiently better candidate remains.

In the simplified greedy version, if $N(v)$ is the neighborhood of $v$, we choose $u^*=\arg\min_{u\in N(v)} d(u,q)$. If $d(u^*,q)<d(v,q)$, move from $v$ to $u^*$. This illustrates the navigation principle, but practical NSW search uses a candidate set rather than following only one path.

For example: $A \rightarrow C \rightarrow F \rightarrow H \rightarrow q$. The long-range edges help the search move quickly between distant regions, while the short-range edges help refine the search once it reaches the correct neighborhood.

A purely greedy algorithm can become trapped at a **local minimum**: a node whose neighbors are all farther from $q$, even though a better node exists somewhere else in the graph. For this reason, practical NSW search does not normally follow only one path. It maintains several promising candidates and explores them before deciding where to continue. This makes the search more robust against local minima.

The main idea can therefore be summarized as: **long edges provide exploration, short edges provide refinement, and proximity-based connections give greedy search a meaningful direction.**

**What Happened to the Rewiring Probability $\beta$?**

At this point, the rewiring probability $\beta$ from the Watts-Strogatz model is no longer part of the practical NSW construction. In the classical small-world model, $\beta$ controls how many local edges are replaced by random long-range shortcuts. In NSW, we instead build the graph directly from the geometry of the vectors. Connections are created using distance-based search and neighbor selection.

So the transition is roughly:

$ \text{Watts-Strogatz: local edges + random rewiring controlled by }\beta $

to

$ \text{NSW: proximity-based incremental connections} $


#### **Limitation of Navigable Small World**

A fundamental limitation of a flat NSW graph is that **different navigation scales are mixed inside the same graph**. Short local edges and longer-range edges coexist without an explicit hierarchy. The search must therefore use the same graph both to travel toward the correct region and to refine the result once it gets there. As the graph grows, this can require exploring an increasing number of candidates. HNSW addresses this problem by explicitly separating these distance scales: sparse upper layers support coarse long-range navigation, while denser lower layers support local refinement.

NSW makes graph search much more efficient than comparing the query with every vector, but the remaining work can still become expensive when the dataset and vector dimension are large.

Suppose the database contains millions of vectors, each with dimension $d=512$. Every time the search visits a node, it must compute the distance between the query and several candidate vectors. For Euclidean distance, one comparison costs roughly $O(d)$ operations. Cosine similarity also requires roughly $O(d)$ operations once vector norms are handled. If the search evaluates $S$ candidate vectors, the distance-computation cost is approximately $O(Sd)$.

Therefore, even if NSW greatly reduces $S$ compared with brute-force search over all $N$ vectors, the cost can still become large when both $S$ and $d$ are large. This creates the motivation for **Hierarchical Navigable Small World (HNSW)**: instead of searching through one large graph at the same resolution, HNSW introduces multiple layers so that the search can first move quickly through a sparse upper graph and then progressively refine the search in denser lower layers.


Therefore, practical NSW does not use the Watts-Strogatz rewiring probability $\beta$. The important quantities instead become the graph connectivity, search breadth, neighbor-selection strategy, and distance measure.

<iframe src="./asset/nsw.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>

#### Bridging the Gap: Why We Still Need the Skip List (HNSW)

We now have both pieces. The skip list gave us a way to build layers on top of a data structure, so that a few of them stay sparse and let us skip over large chunks of data quickly. NSW gave us a way to search inside a single graph: move to whichever neighbor is closest to the query, and repeat until nothing gets closer.

Conceptually, HNSW combines these two ideas: the probabilistic hierarchy of a skip list and the proximity-based graph navigation of NSW.

$$
\text{HNSW} = \text{Skip List (hierarchy \& layer assignment)} + \text{NSW (distance-based routing within a layer)}.
$$

The layers still get sparser as you go up, same as before. The only thing that changes inside a layer is how we move: instead of comparing numbers and going left or right, we compare vectors and move to whichever neighbor is closer to the query.

**Deciding which layers a vector belongs to**

In a skip list, a node reaches level $k$ with probability $p^k$, which creates exponentially smaller populations as we move upward. HNSW uses the same general idea, but samples the maximum layer directly from an exponentially decaying distribution.

When inserting a vector, its maximum layer $l$ is sampled as

$$
l = \left\lfloor -\ln(u) \cdot m_L \right\rfloor, \qquad u \sim \text{Uniform}(0,1),
$$

where $m_L$ controls how quickly the layer populations decrease. A common choice in the original HNSW formulation is

$$
m_L = \frac{1}{\ln(M)},
$$

where $M$ is the main graph-connectivity parameter. From the sampling rule, the probability that a vector reaches at least level $k$ is

$$
P(L \geq k) = e^{-k/m_L}.
$$

Substituting $m_L=1/\ln(M)$ gives

$$
P(L \geq k) = M^{-k} = \left(\frac{1}{M}\right)^k.
$$

This reveals the connection with the skip list. In a skip list, $P(L\geq k)=p^k$; in this HNSW parameterization, the analogous promotion probability is approximately $p=1/M$, **not $M$ itself**.

If the index contains $n$ vectors, the expected number that reach level $k$ is approximately $nM^{-k}$. Setting this quantity to one gives

$$
nM^{-h} \approx 1,
$$

so the hierarchy height scales roughly as

$$
h \approx \log_M n.
$$

**Inserting a new vector**

Say we're inserting a vector $v$. Here's what happens, step by step:

1. **Pick how high $v$ goes.** Use the formula above to draw a layer $l$. $v$ will be added to every layer from the bottom up to $l$.

2. **Walk down to roughly the right area.** The graph keeps track of one starting node, called the entry point, which sits at the current highest layer. Starting from there, at each layer above $l$, we just move to whichever single neighbor is closest to $v$, and drop down a layer once nothing gets closer. This part is cheap because these top layers barely have any nodes in them. All it does is get us into the right neighborhood before we start actually connecting $v$ to anything.

3. **Search properly at each layer $v$ belongs to.** From layer $l$ down to the bottom layer, we run the same kind of search as NSW, but instead of keeping only the single best node, we keep a list of the `efConstruction` best candidates found so far. So at each of these layers, we end up with a decent-sized pool of nearby candidates.

4. **Choose neighbors from that pool.** From the candidate set, we select up to $M$ neighbors for the new vector. Instead of simply choosing the $M$ closest candidates, HNSW uses a heuristic that tries to preserve **geometric diversity**. A candidate may be rejected when it is closer to an already selected neighbor than it is to the new vector. This prevents all selected connections from concentrating in essentially the same local direction and helps preserve useful routes through different regions of the vector space. The bottom layer may use a larger maximum degree than upper layers, depending on the implementation.

5. **Connect both directions, then check for overflow.** When $v$ connects to a neighbor $u$, $u$ also connects back to $v$. If this pushes $u$ over the maximum degree allowed at that layer, the neighbor-selection heuristic is applied again to prune the list.

6. **Update the entry point, if needed.** If $v$'s layer $l$ turns out to be higher than anything currently in the graph, $v$ becomes the new entry point.

Steps 2 and 3 are really just the skip list's "move across, then drop a level" idea, except "move across" now means "move to the closer neighbor."

**Searching for a query**

Looking up a query $q$ works almost the same way:

1. Start at the entry point, at the top layer.
2. At each layer, keep moving to whichever neighbor is closest to $q$. Once nothing gets closer, drop to the layer below.
3. Once we reach the bottom layer, switch to maintaining a broader candidate set controlled by `efSearch`, analogous to `efConstruction` during insertion. The final top-$k$ results are selected from this candidate set.

The top layers do the same job the skip list's higher levels did: they let us cover a lot of ground in very few steps. The bottom layer does what NSW alone would do: slow down and look carefully at what's actually nearby. We need both parts for the same reason the skip list did.

A flat NSW graph already contains long-range connections, but all navigation scales coexist inside the same graph. HNSW makes this structure explicit: sparse upper layers are responsible mainly for long-range navigation, while dense lower layers provide local refinement.

**Why this beats a flat NSW graph**

The benefit of HNSW should not be interpreted as a strict $O(M\log n)$ bound. The upper hierarchy contains increasingly sparse graphs, allowing the search to move between distant regions using relatively few distance evaluations. Once the search reaches the bottom layer, however, it explores a broader candidate set controlled by `efSearch`.

Therefore, the actual query cost depends on several factors: the number of vectors $n$, connectivity parameter $M$, `efSearch`, vector dimension $d$, data geometry, and the desired recall. HNSW is designed to exhibit approximately logarithmic scaling with dataset size under typical conditions, but $O(Md\log n)$ should not be treated as a general worst-case complexity formula.

Compared with flat NSW, the key advantage is **scale separation**: long-range navigation is concentrated in sparse upper layers, while detailed neighborhood exploration happens mainly near the bottom.

**Advantages**

- **Efficient scaling in practice.** The hierarchical organization gives HNSW approximately logarithmic search scaling under typical conditions, without requiring a one-dimensional ordering of the vectors.
- **High recall is achievable in practice.** The bottom-layer exploration can be widened with `efSearch`, trading additional distance evaluations for better recall.
- **No training step needed.** Unlike IVF or PQ, there's no separate clustering pass before you can start indexing. Vectors can be added one at a time, whenever they arrive.
- **Works with any valid distance function.** HNSW can be used with many common vector distance or similarity measures, including Euclidean distance, inner product, and cosine similarity, depending on the implementation.

**Disadvantages**

- **Uses more memory.** Every vector stores a list of neighbors at every layer it belongs to, on top of the vector itself. With a large $M$ and a lot of vectors, this can end up costing more memory than the vectors themselves.
- **Slower to build.** Inserting a vector means running a real search at every layer it touches, plus the neighbor selection step. This is more expensive than, say, just assigning a vector to a cluster in IVF.
- **Hard to delete from.** The graph only works well because of a careful mix of short and long connections. Removing a node cleanly, without breaking the connections that relied on it, is genuinely difficult. 
- **Parameters need tuning.** $M$, `efConstruction`, and `efSearch` all affect the tradeoff between recall, speed, and memory, and there's no simple rule for picking them — you generally have to test on your own data.
- **Irregular memory-access pattern.** Graph traversal jumps between neighbor lists and vectors, which works well in memory but can be unfriendly to storage with high random-read latency. Disk-oriented ANN systems often need specialized graph layouts or caching strategies.

<iframe src="./asset/hnsw.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>



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

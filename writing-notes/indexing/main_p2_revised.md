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

#### **Bridging the Gap: Why We Still Need the Skip List (HNSW)**

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

**Advantages**

- **Efficient scaling in practice.** The hierarchical structure gives HNSW approximately logarithmic search scaling under typical conditions, without requiring a one-dimensional ordering of the vectors.
- **No training step needed.** Unlike IVF or PQ, there's no separate clustering pass before you can start indexing. Vectors can be added one at a time, whenever they arrive.
- **Works with any valid distance function.** HNSW can be used with many common vector distance or similarity measures, including Euclidean distance, inner product, and cosine similarity, depending on the implementation.

**Disadvantages**

- **Uses more memory.** Every vector stores a list of neighbors at every layer it belongs to, on top of the vector itself. With a large $M$ and a lot of vectors, this can end up costing more memory than the vectors themselves.
- **Slower to build.** Inserting a vector means running a real search at every layer it touches, plus the neighbor selection step. This is more expensive than, say, just assigning a vector to a cluster in IVF.

- **Parameters need tuning.** $M$, `efConstruction`, and `efSearch` all affect the tradeoff between recall, speed, and memory, and there's no simple rule for picking them. 

<iframe src="./asset/hnsw.html" width="100%" height="500px" style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>


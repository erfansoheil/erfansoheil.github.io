---
layout: post
title: "Positional Embeddings (PE)"
---

In this article, we will explore the concept of positional embeddings in Transformer architectures, break down the different methods used to calculate them, and discuss when to use each approach. Before diving into the positional aspect, however, we need to understand the foundational concept of modern **embeddings**.

## Embeddings

Every piece of text we input into a model needs to be translated into a language the machine can process, and computers only understand numbers. An **embedding model** is a tool that translates human language (like words, sentences, images, etc.) into a structured set of numbers called a vector.

What is the purpose of an embedding?
Categorization and similarity.

Suppose you have a mountain of messy, unorganized text inputs that you need to sort out. Think of it like having a massive basket filled with a chaotic mix of fruits. Ideally, you want to separate them so that similar fruits end up together.

In the real world, data isn't quite as simple or tangible as a physical basket of fruit. To sort text, we use embedding models to map words into a mathematical space based on their semantic similarities. This ensures that words with related meanings sit close to each other—similar to grouping all the citrus fruits in one corner and berries in another.

The figure below is AI-generated and illustrates and how an embedding model plots words as vectors based on their characteristics, automatically clustering similar concepts together.

![Embedding mechanism](./assets/images/embedding_mechanism.png)


While (token) embeddings are excellent at capturing semantic meaning, they have one critical limitation: they contain no information about word order.
Consider "The cat chased the dog" and "The dog chased the cat". Same words but opposite meanings. Yet to an embedding model, these sentences are identical. Order is invisible.
This is where positional embeddings come in. The idea is simple: before we feed our embeddings into any further processing, we enrich them with information about where each word sits in the sentence.

> **Note:** A method called the **attention mechanism** without positional iformation (which we will explore later) is permutation-invariant, meaning it cannot tell which word came first. Positional embeddings solve this by making position part of the representation itself.

## Positional Embedding Methods (PE methods)

### Absolute positional embeddings

As the name suggests, in this method we assign an absolute positional vector to each token. The model receives information about the absolute position of every token in the sequence (1st, 2nd, 3rd, and so on) by adding a position-specific vector to the token embedding before it enters the network. The positional information is injected once at the input layer, while the attention mechanism itself remains position-agnostic and operates on the combined representation of token embeddings and positional encodings.

> **Note:** A positional encoding vector contains only information about a token's position in the sequence. It does not encode the token's meaning. The final input representation is obtained by combining the token embedding and the positional encoding.


#### Sinusoidal Positional Encoding

Suppose the token $T$ is located at position $n$ in a tokenized sequence $S$. The sinusoidal positional encoding of $T$ is a vector of dimension $d$, where $d$ is the embedding dimension.


For each $0 \le i < d$,

$$
PE(n, i) =
\begin{cases}
\sin\left(\frac{n}{\omega^{i/d}}\right), & \text{if } i \text{ is even} \\
\cos\left(\frac{n}{\omega^{(i-1)/d}}\right) & \text{if } i \text{ is odd}
\end{cases},
$$

where $\omega \in \mathbb{R}$. In the original Transformer architecture, $\omega=10000$.

There are several notes on this topic for example [here](https://www.byhand.ai/p/pytorchexcel-sinusoidal-positional) and [here](https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/).

Let us mention some remarks on senusoidal positional encoding. We start we simple dot product concept. 


##### 0. From Inner Products to Norms to Distances

Before discussing positional encodings, recall some properties of inner products.

Given a vector $v \in \mathbb{R}^d$, its **norm** (length) is defined from the inner product with itself:

$$\|v\| = \sqrt{\langle v, v \rangle} = \sqrt{v \cdot v}$$

A norm gives us a **distance** (metric) between any two vectors $u, v$:

$$d(u, v) = \|u - v\| = \sqrt{\langle u-v,\, u-v\rangle}$$

Expanding:

$$\|u-v\|^2 = \|u\|^2 + \|v\|^2 - 2\langle u, v\rangle$$

If $\|u\|$ and $\|v\|$ are **constant**, then the distance between $u$ and $v$ is determined entirely by their inner product $\langle u, v \rangle$. This is exactly the situation we'll find with positional encodings.


##### 1. Setup

For model dimension $d$ (even), and frequencies

$$\omega_i = \frac{1}{\omega^{i/d}}, \qquad i = 0, 1, \dots, d-1$$

the positional encoding for position $p$ is:

$$PE(p) = \big(\sin(p\,\omega_0),\ \cos(p\,\omega_0),\ \sin(p\,\omega_1),\ \cos(p\,\omega_1),\ \dots\big) \in \mathbb{R}^d$$


##### 2. Every Positional Vector Has the Same Constant Norm

$$\|PE(p)\|^2 = \langle PE(p), PE(p)\rangle = \sum_{i=0}^{d/2-1} \Big[\sin^2(p\,\omega_i) + \cos^2(p\,\omega_i)\Big]$$

By the Pythagorean identity, **each bracket equals 1**. So,

$$\|PE(p)\|^2 = \frac{d}{2} \quad\Longrightarrow\quad \|PE(p)\| = \sqrt{\frac{d}{2}} =: r$$

**$r$ depends only on $d$**, not on the position $p$, and not on the frequencies $\omega_i$.

> Geometrically: every $PE(p)$, for every $p$, lies on the **same sphere of radius $r$** in $\mathbb{R}^d$,(In mathematics we denote this space as $\mathbb{S}^d$). Varying $p$ moves the vector *around* the sphere, never off it.


##### 3. Inner Product Between Two Different Positions

For arbitrary positions $p_1, p_2$:

$$\langle PE(p_1), PE(p_2)\rangle = \sum_{i=0}^{d/2-1} \Big[\sin(p_1\omega_i)\sin(p_2\omega_i) + \cos(p_1\omega_i)\cos(p_2\omega_i)\Big]$$

Each bracket is the cosine subtraction identity $\cos(A)\cos(B)+\sin(A)\sin(B) = \cos(A-B)$, with $A=p_1\omega_i$, $B=p_2\omega_i$:

$$\boxed{\ \langle PE(p_1), PE(p_2)\rangle = \sum_{i=0}^{d/2-1} \cos\big((p_1-p_2)\,\omega_i\big)\ }$$

Two key properties:

- **Bounded**: each term is a cosine $\in [-1,1]$, so the sum is bounded in $[-d/2,\, d/2]$.
- **Depends only on $\Delta p = p_1-p_2$**: the individual positions $p_1, p_2$ vanish, and only their *difference* remains. Two pairs with the same offset give **identical** inner products.


##### 4. Combining Norm + Inner Product: A Built-In Notion of Distance

From Section 0:

$$\|PE(p_1) - PE(p_2)\|^2 = \|PE(p_1)\|^2 + \|PE(p_2)\|^2 - 2\langle PE(p_1), PE(p_2)\rangle$$

Both norms equal $r^2 = d/2$ (Section 2), so:

$$\|PE(p_1) - PE(p_2)\|^2 = d - 2\sum_{i=0}^{d/2-1} \cos\big((p_1-p_2)\,\omega_i\big)$$

From Section 3 we know that the  last term in above equation is bounded in $[-d,\, d]$, therefore $\|PE(p_1) - PE(p_2)\|^2$ is bounded in $[0,\, 2d]$.
This squared distance is **purely a function of $\Delta p$**. So even though sinusoidal PE is *constructed* as an absolute encoding (one fixed vector per position), it induces a **relative geometric structure**: positions close together ($\Delta p$ small) yield vectors close together in space, regardless of where they sit in the sequence.


##### 5. Bounded Distance and Pure Angular Information

From Section 4, $\|PE(p_1)-PE(p_2)\|^2 \in [0, 2d]$. Dividing through by $2d$:

$$\frac{\|PE(p_1)-PE(p_2)\|^2}{2d} \in [0, 1]$$

So after rescaling by the constant $2d$, the squared distance between *any* two positional vectors lies in the unit interval regardless of $p_1, p_2, d$, or the $\omega_i$).

Recall every $PE(p)$ has the **same constant norm** $r=\sqrt{d/2}$ (Section 2). Two vectors of equal, fixed length sitting on a common sphere can only differ in **where they point** — their separation is entirely a question of the angle $\theta$ between them, via the standard identity

$$\|PE(p_1)-PE(p_2)\|^2 = 2r^2(1-\cos\theta) = d(1-\cos\theta)$$

Where $\theta$ is the angle between $PE(p_1)$ and $PE(p_2)$. 
Dividing by $2d$ gives $\frac{1-\cos\theta}{2} \in [0,1]$, a direct, monotonic reparametrization of the angle $\theta$ itself.

**Conclusion:** since the norm $r$ is fixed and the distance is just a rescaled function of $\theta$, *all* positional information (all notion of "closeness" between two positions) is encoded purely in the **angle/phase relationship** between the two vectors, never in magnitude. 
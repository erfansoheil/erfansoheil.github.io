---
layout: post
title: "Temperature in Large Language Models (LLMs)"
---

The goal of thsi arcticle is to explain what is Temperature in LLMs and how it changes the behaviour of the output of the LLMs and discuss about why the ouptus of the LLMs are nondeterministic. 

Before we dive into the concept of **temperature**, we first need to understand the *Softmax* function and how it shapes the model's choices.



## The Softmax Function and Its Properties

Suppose $n \in \mathbb{N}$ represents our vocabulary size, and $v \in \mathbb{R}^n$ is a vector of raw scores (called logits). We can represent $v$ as an $n$-tuple:

$$v = (v_1, v_2, \dots, v_n), \quad v_i \in \mathbb{R}$$

The **Softmax** function $S:\mathbb{R}^n \to (0,1)^n$ normalizes these scores into values between 0 and 1:

$$S(v_i) = \frac{e^{v_i}}{\sum_{j=1}^n e^{v_j}}$$

One immediate observation is that the Softmax function outputs a valid probability distribution—the elements are all positive and sum up to 1. It transforms any arbitrary vector of real numbers into a probability vector.

![Embedding mechanism](./assets/images/decoder_llm.png)


### The Token Generation Process in LLMs

In a decoder-only LLM, the main objective is to predict the next token (word or sub-word) in a sequence.

Right before the final output stage, the model's last linear layer produces a vector of unnormalized scores (logits), where each element corresponds to a specific word in our vocabulary. If we wanted the model to pick only the single absolute best token, we would theoretically want an output vector that contains a `1` at the index of that best word and a `0` everywhere else.

Mathematically, this hard selection is represented by the **Argmax** function:

$$\text{Argmax}: \mathbb{R}^n \to \{0,1\}^n$$

$$\text{Argmax}_i(v) = \begin{cases} 1 & \text{if } v_i = \max\{v_1, \dots, v_n\} \\ 0 & \text{otherwise} \end{cases}$$

However, during training, the model needs to learn via gradient descent. The **Argmax** function is a step function; its derivative is zero almost everywhere, and it is completely discontinuous at the boundaries. Because it is **not differentiable**, we cannot backpropagate errors through it to update the model's weights.

This is exactly why we use **Softmax**. It acts as a "soft", continuous, and fully differentiable **approximation** of Argmax, turning raw scores into a smooth probability map that allows the network to train while still identifying the most likely next tokens.

 > Note: Despite its name, Softmax is not actually a soft approximation of the maximum function. Instead, it is a soft approximation of the Argmax function, which is why many researchers prefer the more accurate term *Softargmax*.

### Soft (smooth) approximation

What does it mean when we say **Softmax** is a soft **approximation** of Argmax? 

Here are two ways to say that a sequence of functions $f_1, f_2, \ldots$ (each mapping $\mathbb{R}^n \to [0,1]^n$) gets closer to a target function $f$.

- **Pointwise convergence:** Fix one input $x$. As $k$ grows, the output $f_k(x)$ approaches $f(x)$.

- **Uniform convergence:** The approximation is close *everywhere at once*. For large enough $k$, $f_k(x) \approx f(x)$ for every input $x \in \mathbb{R}^n$ — not just for one $x$ at a time.

> Note:  Uniform convergence is the stronger notion: uniform $\Rightarrow$ pointwise, but not the other way around.

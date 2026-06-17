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


| Transformer Decoder Layer                                                                                                                          | Geometric Softmax Mapping                                                                                                         |
| :--------------------------------------------------------------------------------------------------------------------------------------------------:| :---------------------------------------------------------------------------------------------------------------------------------:|
| <img src="./assets/images/decoder_llm.png" height="320" />                                                                                         | <img src="./assets/images/output_1.png" height="320" />                                                                           |
| **Model Architecture:** The final layers of a decoder LLM, highlighting the **Linear $\rightarrow$ Softmax** pipeline that prepares token outputs. | **Geometric Intuition:** A 3D visualization showing raw logits (blue) being compressed into a bounded  probability simplex (red). |
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


Now suppose for each $T \in \mathbb{R}$ we define 

$$  
S_T:\mathbb{R}^n \to (0,1)^n \\

S_T(v_i) = \frac{e^{v_i/T}}{\sum_{j=1}^n e^{v_j/T}}
$$

In $S_T$ in the above equation is called **The Softmax Function with Temperature $T$**


Let $M = \max_{j} v_j$, be the maximum value in the vector $v$. We want to prove that as $T \to 0^+$, the softmax distribution pointwise converges to the argmax distribution:


$$\lim_{T \to 0^+} S_T (v)_i = \text{Argmax}(v)_i$$

Let's evaluate the limit of the $i$-th component, $S_T(v_i)$:


$$\lim_{T \to 0^+} S_T(v_i) = \lim_{T \to 0^+} \frac{e^{v_i/T}}{\sum_{j=1}^n e^{v_j/T}}$$

We multiply both the numerator and the denominator by $e^{-M/T}$:

$$S_T(v_i) = \frac{e^{v_i/T} \cdot e^{-M/T}}{\left(\sum_{j=1}^n e^{v_j/T}\right) \cdot e^{-M/T}}$$

Using the exponent rule $e^a \cdot e^b = e^{a+b}$, we can rewrite this as:


$$S_T(v_i) = \frac{e^{(v_j - M)/T}}{\sum_{j=1}^n e^{(v_j - M)/T}}$$

In the denominator there is one index, like $k$ such that $v_k = M. So, $(v_k - M) = 0$. Therefore, $e^{v_k-M} = e^{0/T} = e^0 = 1$. For other indices, know $v_j < M$, so $(v_j - M)$ is a strictly negative constant. Let $c_j = v_j - M < 0$. As $T \to 0^+$, the exponent $\frac{c_j}{T} \to -\infty$. Consequently, $e^{(v_j - M)/T} \to 0$.

Thus, the limit of the denominator as $T \to 0^+$ is:


$$\lim_{T \to 0^+} \sum_{j=1}^n e^{(v_j - M)/T} = 1+0 = 1$$

Now, we evaluate the limit of the entire fraction by looking at the numerator for the two possible cases for $v_j$:

**Case 1: $v_j$ is not the maximum ($v_j < M$)**

If $v_j < M$, then $(v_j - M) < 0$.


$$\lim_{T \to 0^+} e^{(v_j - M)/T} = 0  \Longrightarrow \lim_{T \to 0^+} S_T(v_i)= \frac{0}{1} = 0$$

**Case 2: $v_j$ is the maximum ($v_j = M$)**

If $v_j = M$, then $(v_j - M) = 0$.


$$\lim_{T \to 0^+} e^{(v_j - M)/T} = e^0 = 1  \Longrightarrow \lim_{T \to 0^+} S_T(v_i)= \frac{1}{1}= 1 $$

Combining both cases, we get:
$$ \lim_{T \to 0^+} S_T(v_i)= \begin{cases}
1 & \text{if } v_j = \max(v) \\
0 & \text{otherwise}
\end{cases} $$

The above argument shows that the softmax function converges pointwise to the argmax function.
> **Note:** However, this convergence is not uniform. In fact, since all of the $S_T$ functions are continuous, if they uniformly converged to a limit function $S$, then $S$ must be continuous too. However, the Argmax function is not continuous.

 In the context of LLMs, this means that if we reduce the temperature to exactly 0 (or almost 0), the model's outputs—the new tokens—will become deterministic. Normally, models like GPT run at a higher default temperature, which is why you can input the same query twice and get different answers (same meaning, but different grammar or vocabulary). Lowering the temperature to 0 eliminates this variance.
 On the other hand, when we increase the temperature ($T \to +\infty$), the limit of the functions $S_T$ converges to the uniform distribution $S(v)_i=\frac{1}{n}$. This means that the model assigns the exact same weight to every word in the vocabulary, resulting in completely random and chaotic outputs.

First let us sprove the second claim : By increasing the temperature he functions $S_T$ converges to the uniform distribution $S(v)_i=\frac{1}{n}$.

Here is the mathematical proof.


**The Uniform Distribution:**
A discrete uniform distribution over $n$ items means each item has an equal probability of occurring. The probability for each item $i$ is exactly the "mean" or equal share of the total probability mass ($1$):

$$U(\mathbf{x})_i = \frac{1}{n}$$


For each $1 \leq j \leq n$,

$$\lim_{T \to \infty} e^{v_j/T} =e^{\lim_{T \to \infty} v_j/T} =  e^0 = 1$$

since  $v_j$ is a constant real number.

Now, we apply this to both the numerator and the denominator of our softmax function.

**For the numerator:**


$$\lim_{T \to \infty} e^{v_i/T} = 1$$

**For the denominator:**
Because the limit of a finite sum is the sum of the limits, we get:


$$\lim_{T \to \infty} \sum_{j=1}^n e^{v_j/T} = \sum_{j=1}^n \left( \lim_{T \to \infty} e^{v_j/T} \right) = \sum_{j=1}^n 1$$

Summing the number $1$ exactly $n$ times simply gives us $n$:


$$\sum_{j=1}^n 1 = n$$


Putting the numerator and denominator back together, the limit of the entire fraction is:

$$\lim_{T \to \infty} \sigma(\mathbf{x}/T)_i = \frac{1}{n}$$

This proves that as the temperature approaches infinity, the softmax function completely ignores the original input values $v_i$. Every single class gets assigned the exact same probability of $\frac{1}{n}$, giving you a perfectly uniform distribution.

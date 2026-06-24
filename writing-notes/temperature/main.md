---
layout: post
title: "Temperature in Large Language Models (LLMs)"
---

The goal of this article is to explain what temperature is in LLMs and how it changes the behaviour of the output of the LLMs.

Before diving into the math, it helps to have a clear picture of what an LLM is actually doing at inference time. At its core, a decoder-only language model is a next-token predictor: given a sequence of tokens  (words or sub-words), it tries to predict which token should come next.  To do this, the model's final linear layer produces a raw score, called  a **logit**, for every token in its vocabulary, which can easily be in  the tens of thousands. These logits are then converted into probabilities, from which the model samples its next output. The mechanism that performs  this conversion, and the parameter that controls *how* it samples, are 
exactly what this article is about. 

Before we dive into the concept of **temperature**, we first need to understand the *Softmax* function and how it shapes the model's choices.



## The Softmax Function and Its Properties

Suppose $n \in \mathbb{N}$ represents our vocabulary size, and $v \in \mathbb{R}^n$ is a vector of raw scores (called logits). We can represent $v$ as an $n$-tuple:

$$v = (v_1, v_2, \ldots, v_n)$$

with each component $v_i \in \mathbb{R}$.

The **Softmax** function $S:\mathbb{R}^n \to (0,1)^n$ normalizes these scores into values between 0 and 1:

$$S(v_i) = \frac{e^{v_i}}{\sum_{j=1}^n e^{v_j}}$$

One immediate observation is that the Softmax function outputs a valid probability distribution—the elements are all positive and sum up to 1. It transforms any arbitrary vector of real numbers into a probability vector.

<div class="figure-pair">
  <figure>
    <img src="./assets/images/decoder_llm.png" alt="Transformer decoder layer" />
    <figcaption><strong>Model Architecture:</strong> The final layers of a decoder LLM, highlighting the <strong>Linear → Softmax</strong> pipeline that prepares token outputs.</figcaption>
  </figure>
  <figure>
    <img src="./assets/images/output_1.png" alt="Geometric softmax mapping" />
    <figcaption><strong>Geometric Intuition:</strong> A 3D visualization showing raw logits (blue) being compressed into a bounded probability simplex (red).</figcaption>
  </figure>
</div>

<style>
.figure-pair {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}
.figure-pair figure {
  margin: 0;
  text-align: center;
}
.figure-pair img {
  display: block;
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  margin: 0 auto;
}
.figure-pair figcaption {
  margin-top: 0.75rem;
  font-size: 0.95rem;
  line-height: 1.5;
  text-align: left;
}
@media (max-width: 768px) {
  .figure-pair {
    grid-template-columns: 1fr;
  }
}
</style>
### The Token Generation Process in LLMs

In a decoder-only LLM, the main objective is to predict the next token (word or sub-word) in a sequence.

Right before the final output stage, the model's last linear layer produces a vector of unnormalized scores (logits), where each element corresponds to a specific word in our vocabulary. If we wanted the model to pick only the single absolute best token, we would theoretically want an output vector that contains a `1` at the index of that best word and a `0` everywhere else.

Mathematically, this hard selection is represented by the **Argmax** function:

$$\text{Argmax}: \mathbb{R}^n \to \{0,1\}^n$$

$$\text{Argmax}_i(v) = \begin{cases} 1 & \text{if } v_i = \max\{v_1, \dots, v_n\} \\ 0 & \text{otherwise} \end{cases}$$

However, during training, the model needs to learn via gradient descent. The **Argmax** function is a step function; its derivative is zero almost everywhere, and it is completely discontinuous at the boundaries. Because it is **not differentiable**, we cannot backpropagate errors through it to update the model's weights.

This is exactly why we use **Softmax**. It acts as a "soft", continuous, and fully differentiable **approximation** of Argmax, turning raw scores into a smooth probability map that allows the network to train while still identifying the most likely next tokens.


 *Note: Despite its name, Softmax is not actually a soft approximation of the maximum function. Instead, it is a soft approximation of the Argmax function, which is why many researchers prefer the more accurate term *Softargmax*.*

### Soft (smooth) approximation

What does it mean when we say **Softmax** is a soft **approximation** of Argmax? 

Here are two ways to say that a sequence of functions $f_1, f_2, \ldots$ (each mapping $\mathbb{R}^n \to [0,1]^n$) gets closer to a target function $f$.

- **Pointwise convergence:** Fix one input $x$. As $k$ grows, the output $f_k(x)$ approaches $f(x)$.

- **Uniform convergence:** The approximation is close *everywhere at once*. For large enough $k$, $f_k(x) \approx f(x)$ for every input $x \in \mathbb{R}^n$ — not just for one $x$ at a time.

*Note: Uniform convergence is the stronger notion: uniform $\Rightarrow$ pointwise, but not the other way around.*


Now suppose for each $T \in \mathbb{R}$ we define 

$$  
S_T:\mathbb{R}^n \to (0,1)^n 
$$
$$
S_T(v_i) = \frac{e^{v_i/T}}{\sum_{j=1}^n e^{v_j/T}}
$$

$S_T$ in the above equation is called the **softmax function with temperature $T$**.

***Training vs. Inference:** It is worth being precise about *when* temperature plays a role. During **training**, the standard Softmax (implicitly with $T = 1$) is used to maintain differentiability and enable gradient-based learning. During **inference**, however, temperature becomes a tunable knob that controls the shape of the output distribution *after* the model's weights are frozen. Unless stated otherwise, every reference to temperature in this article refers to the **inference-time** setting.*

Let $M = \max_{j} v_j$ be the maximum value in the vector $v$. We want to prove that as $T \to 0^+$, the softmax distribution pointwise converges to the argmax distribution:


$$\lim_{T \to 0^+} S_T (v)_i = \text{Argmax}(v)_i$$

Let's evaluate the limit of the $i$-th component, $S_T(v_i)$:


$$\lim_{T \to 0^+} S_T(v_i) = \lim_{T \to 0^+} \frac{e^{v_i/T}}{\sum_{j=1}^n e^{v_j/T}}$$

We multiply both the numerator and the denominator by $e^{-M/T}$:

$$S_T(v_i) = \frac{e^{v_i/T} \cdot e^{-M/T}}{\left(\sum_{j=1}^n e^{v_j/T}\right) \cdot e^{-M/T}}$$

Using the exponent rule $e^a \cdot e^b = e^{a+b}$, we can rewrite this as:


$$S_T(v_i) = \frac{e^{(v_i - M)/T}}{\sum_{j=1}^n e^{(v_j - M)/T}}$$

In the denominator there is one index, like $k$, such that $v_k = M$. So, $(v_k - M) = 0$. Therefore, $e^{v_k-M} = e^{0/T} = e^0 = 1$. For other indices, we know $v_j < M$, so $(v_j - M)$ is a strictly negative constant. Let $c_j = v_j - M < 0$. As $T \to 0^+$, the exponent $\frac{c_j}{T} \to -\infty$. Consequently, $e^{(v_j - M)/T} \to 0$.

Thus, the limit of the denominator as $T \to 0^+$ is:


$$\lim_{T \to 0^+} \sum_{j=1}^n e^{(v_j - M)/T} = 1+0 = 1$$

Now, we evaluate the limit of the entire fraction by looking at the numerator for the two possible cases for $v_i$:

**Case 1: $v_i$ is not the maximum ($v_i < M$)**

If $v_i < M$, then $(v_i - M) < 0$.


$$\lim_{T \to 0^+} e^{(v_i - M)/T} = 0  \Longrightarrow \lim_{T \to 0^+} S_T(v_i)= \frac{0}{1} = 0$$

**Case 2: $v_i$ is the maximum ($v_i = M$)**

If $v_i = M$, then $(v_i - M) = 0$.


$$\lim_{T \to 0^+} e^{(v_i - M)/T} = e^0 = 1  \Longrightarrow \lim_{T \to 0^+} S_T(v_i)= \frac{1}{1}= 1 $$

Combining both cases, we get:
$$ \lim_{T \to 0^+} S_T(v_i)= \begin{cases}
1 & \text{if } v_i = \max(v) \\
0 & \text{otherwise}
\end{cases} $$

The above argument shows that the softmax function converges pointwise to the argmax function.
 ***Note:** However, this convergence is not uniform. In fact, since all of the $S_T$ functions are continuous, if they uniformly converged to a limit function $S$, then $S$ must be continuous too. However, the Argmax function is not continuous. In practical terms, this means that the  "sharpening" effect of lowering the temperature is **input-dependent**. For a logit vector where one score strongly dominates the others, even a  moderate temperature reduction will push the distribution close to deterministic. But for a logit vector where scores are nearly tied, you  may need to lower the temperature much further to achieve the same level of concentration. There is no single threshold temperature that produces the same behavior across all inputs. Simply put: *context always matters*.*

 In the context of LLMs, this means that if we reduce the temperature to exactly 0 (or almost 0), the model's outputs—the new tokens—will become deterministic. Normally, models like GPT run at a higher default temperature, which is why you can input the same query twice and get different answers (same meaning, but different grammar or vocabulary). Lowering the temperature to 0 eliminates this variance.
 On the other hand, when we increase the temperature ($T \to +\infty$), the limit of the functions $S_T$ converges to the uniform distribution $S(v)_i=\frac{1}{n}$. This means that the model assigns the exact same weight to every word in the vocabulary, resulting in completely random and chaotic outputs.

First let us prove the second claim: by increasing the temperature, the functions $S_T$ converge to the uniform distribution $S(v)_i=\frac{1}{n}$.

Here is the mathematical proof.


**The Uniform Distribution:**
A discrete uniform distribution over $n$ items means each item has an equal probability of occurring. The probability for each item $i$ is exactly the "mean" or equal share of the total probability mass ($1$):

$$U(\mathbf{x})_i = \frac{1}{n}$$


For each $1 \leq j \leq n$,

$$\lim_{T \to \infty} e^{v_j/T} =e^{\lim_{T \to \infty} v_j/T} =  e^0 = 1$$

since $v_j$ is a constant real number.

Now, we apply this to both the numerator and the denominator of our softmax function.

**For the numerator:**


$$\lim_{T \to \infty} e^{v_i/T} = 1$$

**For the denominator:**
Because the limit of a finite sum is the sum of the limits, we get:


$$\lim_{T \to \infty} \sum_{j=1}^n e^{v_j/T} = \sum_{j=1}^n \left( \lim_{T \to \infty} e^{v_j/T} \right) = \sum_{j=1}^n 1$$

Summing the number $1$ exactly $n$ times simply gives us $n$:


$$\sum_{j=1}^n 1 = n$$


Putting the numerator and denominator back together, the limit of the entire fraction is:

$$\lim_{T \to \infty} S_T(v)_i = \frac{1}{n}$$

This proves that as the temperature approaches infinity, the softmax function completely ignores the original input values $v_i$. Every single class gets assigned the exact same probability of $\frac{1}{n}$, giving you a perfectly uniform distribution.


To make these theoretical claims concrete, the following short demonstration shows what temperature actually looks like in practice on a real language  model. The same prompt is submitted multiple times at progressively different 
temperature values 

<video controls autoplay muted loop width="100%" style="border-radius: 8px;">
  <source src="./assets/videos/temperature_vid.webm" type="video/webm">
<source src="./assets/videos/temperature_vid.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


This short demonstration illustrates the effect of the **temperature** parameter on the output of a locally hosted language model. In text generation, temperature is applied during the sampling stage after softmax, controlling how strongly the model favors high-probability tokens. Lower temperatures generally produce more deterministic and conservative outputs, while higher temperatures increase variability and can lead to more unexpected generations.

For the experiment, I use **Ollama** to run `llama3.2:3b` locally on my own machine. The command `/set parameter temperature ...` is used to modify the generation temperature, and `/clear` is used before each run to remove the previous conversation history and make the comparison cleaner. I also fix the `seed` to improve reproducibility, although the role of seed will be discussed separately in a later blog post.

Since the experiment is performed with a relatively small 3B model running locally, the results should be interpreted as a practical illustration rather than a general benchmark. The goal is to provide an intuitive example of how temperature influences language model behavior.


To truly grasp how temperature warps the model's output, it helps to look at it geometrically. Because the Softmax function outputs probabilities that sum to 1, any 3-dimensional output (for a vocabulary of 3 words) must live on a flat, triangular surface called a probability simplex. This triangle connects the absolute certainty points: $(1,0,0)$, $(0,1,0)$, and $(0,0,1)$.

Let's fix a raw logit vector to observe how temperature moves the output probability across this space. In the following interactive plots, our model has output the raw scores²: 

$$v = [2.5, 1.0, -0.5]$$ 

#### Interactive: Convergence to Argmax
<iframe src="./assets/argmax_3d.html" width="80%" height="400px" frameborder="0" scrolling="no"></iframe>

In this first visualization, we lower the temperature from $T = 2.0$ down to near zero.

The Space: The $P_1$, $P_2$, and $P_3$ axes represent the probability assigned to each of our three logits. The gray outline is the simplex boundary.

The Trajectory: Follow the colored line. As $T$ decreases, the output distribution is aggressively pulled toward the corner vertex $(1, 0, 0)$. The model becomes 100% confident in $v_1$ (the maximum logit) and assigns 0% probability to the others.

The Color: The color gradient of the points maps to the temperature. Lighter/yellower points represent higher temperatures, while the darker/purple points show the temperature approaching $0$, where the point finally rests at the Argmax vertex.

#### Interactive: Convergence to Uniform Distribution
<iframe src="./assets/uniform_3d.html" width="80%" height="400px" frameborder="0" scrolling="no"></iframe>

Now, let's see what happens when we heat things up. Using the exact same initial logit vector $v = [2.5, 1.0, -0.5]$, we increase the temperature from $T = 1.0$ up to $T = 50.0$.

The Trajectory: Instead of moving toward a corner, the rising temperature "washes out" the differences between the raw scores. The point is pulled directly into the dead center of the triangle, the coordinate $(1/3, 1/3, 1/3)$.

The Meaning: At this center point, the model completely ignores the fact that $v_1$ was much larger than $v_3$. It assigns an equal $33.3\%$ probability to all three tokens, making the output completely random.

The Color: Again, the color gradient maps the temperature. The dark points represent the starting temperature, and as the color shifts to yellow, $T$ is approaching infinity, dragging the distribution to perfect uniformity.
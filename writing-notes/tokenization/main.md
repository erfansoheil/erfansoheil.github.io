---
layout: post
title: "Tpkenization VS Token Representation"
---

The idea of this article came to my mind when I was readig about LLM archtiectures and was trying to understand the difference between **Token**, **Tokenization**, **Token IDs** and **Token Represenations**. At first all of these look the same to me and it took some time for me to categorize them in my mind. This articel is a try to create a mental map for someone who wants to go deep in these topics. 

In this artcile I first discuss about *Token* and *Tokenization* process in LLMs. Then we disucss about *Token Representations* process. Before that lets recall the trasdformers architecture since it is our building block of this article. 

![Transformers Architecture](./assets/images/trans_arch.png)


In the above figure you can see the **Input** and **Output** embeddings moduls. The whole process of tokenization happens before embedding process (later we see that thsi step called *Token Representation*)

## First Part: Tokenization


Throughout this article, the word **token** is used frequently. A token is a **discrete unit of information processed by a model**. In domain of language models, a token usually represents a word, part of a word, a punctuation mark, or another piece of text. In other domains, however, tokens may represent image patches, audio segments, time-series windows, and so on.

Before a model can process an input, it must first convert it into a sequence of tokens. The tokens themselves are not directly understood by the neural network. Instead, they must be converted into numbers using the model's Vocabulary—which is a fixed master list of every token the model is trained to recognize.

The total number of unique items in this list is the Vocabulary Size. Each token is mapped to its exact index number in this vocabulary, known as a Token ID. These token IDs are then transformed into meaningful vectors through an **embedding layer**, producing numerical representations that the model can process.

As a result, the pipeline looks as follows:

<center>Input → Tokens → Token IDs → Embeddings (Vectors)</center>

The process of converting an input into tokens is called **tokenization**. Lets make an example of the fist two steps the above process, meaning :

<center>Input → Tokens → Token IDs </center>


<!-- , while the process of converting token IDs into vectors is called **embedding**. -->

**Exmple :**

Let’s see how a simple sentence moves through the first two steps of the pipeline using a standard subword tokenizer (like the one used by GPT models).

- Input (Raw Text): > "Tokenization is fun!"

- Tokens (Discrete Units): The tokenizer chops the text into pieces. Notice how "Tokenization" is broken into smaller subword units:
$[\text{Token}, \text{ization}, \text{is}, \text{fun}, \text{!}]$

- Token IDs (Unique Integers): Each token is looked up in the model's pre-defined vocabulary dictionary and replaced by its corresponding mathematical ID:
$[30121, 1634, 318, 1257, 0]$



**Why do we need Token IDs after Tokenization?**

It is tempting to think that once we split a sentence into string-based tokens like `["Token", "ization","is","fun","!"]`, the hard part is over. However, computers and neural networks cannot perform mathematical operations on raw text strings. They operate strictly on matrices and vectors of floating-point numbers. 

Token IDs act as the bridge. By assigning every unique token in our vocabulary a fixed, unique integer index (e.g., `"token"` $\rightarrow$ `2534`), we create a structured lookup system that the neural network can interact with mathematically.

Think of this lookup table as a simple, two-way dictionary that handles two distinct phases:

* **Token to ID (Encoding):** When processing your input text, the system looks up each string token to find its corresponding index number.
  $$\text{"Token"} \longrightarrow \text{Lookup Table} \longrightarrow 30121$$

* **ID to Token (Decoding):** When the model generates a response, it outputs a raw number. This number is run backward through the exact same table to turn it back into a readable word for humans.
  $$30121 \longrightarrow \text{Lookup Table} \longrightarrow \text{"Token"}$$

<!-- **Are Token IDs Fixed or Learned?**

A common point of confusion is whether these IDs change as the model learns. **Token IDs are entirely fixed.** Once a tokenizer is trained and its vocabulary dictionary is locked in, the ID for a specific token never changes. For instance, in a BERT tokenizer, the token `"the"` will always map to the integer ID `1996`. What *does* change during the LLM's training are the **embedding vectors** associated with those IDs. The integer ID is simply a stable index used to point the model to the correct row in its embedding matrix. -->



**Where Do These Numbers Come From?**

A natural question arises: **how do we come up with these specific numbers in the first place?** 

This mapping isn't random, nor is it done manually. The process of tokenization is handled via a **Tokenizer Model** (such as Byte-Pair Encoding or WordPiece). Before the main Large Language Model (LLM) can even begin reading text, this tokenizer model must go through its own independent training phase.

When we say a tokenizer is "trained," we don't mean it uses complex neural networks. Instead, tokenizer training is a statistical optimization process. Its entire goal is to read a massive sample dataset of raw text, analyze it, and find the most efficient balance between character-level chunks and whole-word chunks to build its vocabulary list.

Modern LLMs rely on three primary algorithmic flavors to construct their vocabulary:

* **Byte-Pair Encoding (BPE)**: Starts with individual characters and iteratively merges the most frequent adjacent pairs. (Used by: GPT models, LLaMA).

* **WordPiece**: Similar to BPE, but instead of choosing the most frequent pair, it picks pairs based on a statistical likelihood that maximizes predictability. (Used by: BERT).

* **Unigram**: Starts with a massive vocabulary of full words and iteratively removes (prunes) the least useful tokens. (Used by: T5).

In the followig we dezcribe these three methods individually. 


**Method 1: Byte-Pair Encoding (BPE)**

**Byte-Pair Encoding (BPE)** is a bottom-up subword tokenization method. It starts from a small base alphabet, usually characters or bytes, and gradually builds larger units by merging frequent adjacent pairs.

BPE was not originally designed for language models. It comes from **data compression**: the idea was to replace frequently repeated symbol pairs with a new symbol, making the sequence shorter. Modern NLP adapted the same idea for tokenization. Instead of only compressing text, BPE also produces a vocabulary of useful subword units.

**Intuition**

The core idea is:

$$
\text{frequent pair} \quad \Rightarrow \quad \text{good candidate for a new token}
$$

For example, if the pair `d` + `e` appears many times in the corpus, BPE may create a new token `de`. Later, if `de` + `r` is frequent, it may create `der`. In this way, BPE gradually builds larger subwords and sometimes full words.


**How BPE is Trained**

1. **Start from a base vocabulary**
   The corpus is first represented using a small set of atomic symbols, such as characters or bytes. In some versions, an end-of-word marker like `</w>` is added so that the algorithm can distinguish word boundaries.

2. **Count adjacent pairs**
   BPE scans the corpus and counts how often each neighboring pair of symbols appears. For example, it may count pairs such as `d e`, `e r`, `t h`, or `i n`.

3. **Choose the best immediate merge**
   The most frequent adjacent pair is selected. This is a greedy decision: BPE chooses the merge that gives the largest immediate reduction in sequence length.

   If a pair appears many times, replacing each occurrence of two symbols by one new symbol shortens the corpus:

   $$
   (a, b) \rightarrow ab
   $$

4. **Update the corpus**
   All selected occurrences of that pair are replaced by the new merged token. The corpus is now represented using this larger unit.

5. **Repeat until the vocabulary budget is reached**
   The algorithm repeats this process until it has learned the desired number of merge rules or reached the target vocabulary size.

**A Mathematical Perspective**

From a mathematical perspective, BPE is trying to find a sequence of merges:

$$
\mu = (\mu_1, \mu_2, \dots, \mu_M)
$$

that maximizes the *compression* utility of the corpus:

$$
\kappa_x(\mu)
$$

Here, $(\kappa_x(\mu))$ measures how much shorter the sequence becomes after applying the merge sequence $(\mu)$. The ideal objective would be:

$$
\mu^\star = \arg\max_{\mu} \kappa_x(\mu)
$$

However, standard BPE does not search over all possible merge sequences, as that would be computationally expensive. Instead, it uses a greedy approximation: at each step, it chooses the merge with the best immediate gain.

Algorithm 3 of the paper [*A Formal Perspective on Byte-Pair Encoding*](https://arxiv.org/abs/2306.16837) gives a useful way to understand what “optimal BPE” would mean.

* Standard BPE is greedy: at each step, it merges the most frequent adjacent pair. This gives the best immediate compression gain, but it does not guarantee the best final vocabulary. A merge that looks good now may prevent a better sequence of merges later.

* Algorithm 3 takes a different approach. Instead of committing immediately to the most frequent pair, it searches over possible merge sequences and keeps the sequence that gives the best final compression. In other words, standard BPE asks, “Which pair should I merge now?”, while Algorithm 3 asks, “Which full sequence of merges gives the shortest final representation?”

This makes Algorithm 3 an exact method for finding an optimal BPE vocabulary under the compression objective. However, it is computationally expensive, so it is mainly useful for theoretical analysis rather than for training large real-world tokenizers.

For a practical and minimal implementation of standard BPE, Andrej Karpathy’s [*minbpe*](https://github.com/karpathy/minbpe) repository is a good reference. It implements the usual greedy version of BPE: count adjacent pairs, merge the most frequent pair, update the text, and repeat. This is different from Algorithm 3 in the formal paper, which searches for an optimal merge sequence. So `minbpe` is useful for understanding how BPE is used in practice, while Algorithm 3 is useful for understanding what an optimal BPE vocabulary would mean mathematically.







**Why BPE Works Well in LLMs**

* Frequent patterns become tokens
* Rare words can still be decomposed into smaller pieces
* The model avoids relying only on a fixed word-level vocabulary

This balance makes BPE an effective and widely used tokenization method in modern language models.

 **Method 2: WordPiece**

**WordPiece** is very similar to BPE in that it starts with individual characters and iteratively merges them upward into larger units. However, instead of simply picking the most *frequent* pair, WordPiece uses a statistical calculation to pick the pair that makes the language most predictable. It is famously used by Google's BERT model.



**Method 1: Byte-Pair Encoding (BPE)**

**Byte-Pair Encoding (BPE)** is a bottom-up subword tokenization method. It starts from a small base alphabet, usually characters or bytes, and gradually builds larger units by merging frequent adjacent pairs.

BPE was not originally designed for language models. It comes from **data compression**: the idea was to replace frequently repeated symbol pairs with a new symbol, making the sequence shorter. Modern NLP adapted the same idea for tokenization. Instead of only compressing text, BPE also produces a vocabulary of useful subword units.

**Intuition**

The core idea is:

$$
\text{frequent pair} \quad \Rightarrow \quad \text{good candidate for a new token}
$$

For example, if the pair `d` + `e` appears many times in the corpus, BPE may create a new token `de`. Later, if `de` + `r` is frequent, it may create `der`. In this way, BPE gradually builds larger subwords and sometimes full words.


**How BPE is Trained**

1. **Start from a base vocabulary**
   The corpus is first represented using a small set of atomic symbols, such as characters or bytes. In some versions, an end-of-word marker like `</w>` is added so that the algorithm can distinguish word boundaries.

2. **Count adjacent pairs**
   BPE scans the corpus and counts how often each neighboring pair of symbols appears. For example, it may count pairs such as `d e`, `e r`, `t h`, or `i n`.

3. **Choose the best immediate merge**
   The most frequent adjacent pair is selected. This is a greedy decision: BPE chooses the merge that gives the largest immediate reduction in sequence length.

   If a pair appears many times, replacing each occurrence of two symbols by one new symbol shortens the corpus:

   $$
   (a, b) \rightarrow ab
   $$

4. **Update the corpus**
   All selected occurrences of that pair are replaced by the new merged token. The corpus is now represented using this larger unit.

5. **Repeat until the vocabulary budget is reached**
   The algorithm repeats this process until it has learned the desired number of merge rules or reached the target vocabulary size.

**A Mathematical Perspective**

From a mathematical perspective, BPE is trying to find a sequence of merges:

$$
\mu = (\mu_1, \mu_2, \dots, \mu_M)
$$

that maximizes the *compression* utility of the corpus:

$$
\kappa_x(\mu)
$$

Here, $\kappa_x(\mu)$ measures how much shorter the sequence becomes after applying the merge sequence $\mu$. The ideal objective would be:

$$
\mu^\star = \arg\max_{\mu} \kappa_x(\mu)
$$

However, standard BPE does not search over all possible merge sequences, as that would be computationally expensive. Instead, it uses a greedy approximation: at each step, it chooses the merge with the best immediate gain.

Algorithm 3 of the paper [*A Formal Perspective on Byte-Pair Encoding*](https://arxiv.org/abs/2306.16837) gives a useful way to understand what “optimal BPE” would mean.

* Standard BPE is greedy: at each step, it merges the most frequent adjacent pair. This gives the best immediate compression gain, but it does not guarantee the best final vocabulary. A merge that looks good now may prevent a better sequence of merges later.

* Algorithm 3 takes a different approach. Instead of committing immediately to the most frequent pair, it searches over possible merge sequences and keeps the sequence that gives the best final compression. In other words, standard BPE asks, “Which pair should I merge now?”, while Algorithm 3 asks, “Which full sequence of merges gives the shortest final representation?”

This makes Algorithm 3 an exact method for finding an optimal BPE vocabulary under the compression objective. However, it is computationally expensive, so it is mainly useful for theoretical analysis rather than for training large real-world tokenizers.

For a practical and minimal implementation of standard BPE, Andrej Karpathy’s [*minbpe*](https://github.com/karpathy/minbpe) repository is a good reference. It implements the usual greedy version of BPE: count adjacent pairs, merge the most frequent pair, update the text, and repeat. This is different from Algorithm 3 in the formal paper, which searches for an optimal merge sequence. So `minbpe` is useful for understanding how BPE is used in practice, while Algorithm 3 is useful for understanding what an optimal BPE vocabulary would mean mathematically.

**Why BPE Works Well in LLMs**

* Frequent patterns become tokens
* Rare words can still be decomposed into smaller pieces
* The model avoids relying only on a fixed word-level vocabulary

This balance makes BPE an effective and widely used tokenization method in modern language models.


**Method 2: WordPiece**

**WordPiece** is another bottom-up subword tokenization method, heavily utilized by models like BERT and DistilBERT. Structurally it is similar with BPE. Meaning, both of thel starting from a base alphabet and iteratively expanding the vocabulary. However then merging strategy is grounded in probability and information theory rather than raw frequency.

**Intuition**

Instead of merging the most *frequent* adjacent pair, WordPiece merges the pair that **maximizes the likelihood of the training data** according to a unigram language model. 

The core idea can be expressed as:

$$
\text{highest relative mutual information} \quad \Rightarrow \quad \text{good candidate for a new token}
$$

For example, if the individual characters `p` and `h` are highly frequent on their own (appearing in many words like `up`, `hope`, `hot`, `pet`), but they rarely appear together, a simple frequency-based method might still merge them if the corpus is large enough. WordPiece, however, evaluates how much *more likely* they are to appear together than we would expect by chance. It creates a new token `ph` only if the combination carries high predictive value.

To manage word boundaries, WordPiece uses a special prefix—usually **`##`**—to mark subwords that must be attached to a preceding token (e.g., `pre`, `##train`, `##ing`).

**How WordPiece is Trained**

1. **Initialize the vocabulary**
   The vocabulary is seeded with all basic characters, punctuation marks, and special symbols present in the corpus. Suffix characters are duplicated with the `##` prefix to handle inside-word pieces.

2. **Build a unigram language model**
   The algorithm treats the current vocabulary as independent units and estimates a language model over the text. 

3. **Score all adjacent pairs**
   WordPiece evaluates every neighboring pair of symbols $(A, B)$ in the text using a scoring metric derived from their individual and joint frequencies:

   $$
   \text{Score}(A, B) = \frac{\text{Count}(AB)}{\text{Count}(A) \times \text{Count}(B)}
   $$

4. **Select and merge the highest-scoring pair**
   The pair that yields the maximum score is chosen and added to the vocabulary as a new merged token $AB$. 

5. **Repeat until the vocabulary budget is reached**
   The text is updated, frequencies are re-computed, and the process repeats until the predefined vocabulary size is filled.

**A Mathematical Perspective**

Mathematically, WordPiece optimizes a probabilistic objective. Given a text corpus $X = (x_1, x_2, \dots, x_N)$ and a vocabulary $V$, a unigram language model assumes that the probability of the corpus is the product of the probabilities of its constituent tokens:

$$
P(X) = \prod_{i=1}^{N} P(t_i)
$$

Where each token $t_i \in V$, and its probability is estimated via maximum likelihood: $P(t) = \frac{\text{Count}(t)}{\sum_{t' \in V} \text{Count}(t')}$.

When WordPiece considers merging two adjacent tokens $A$ and $B$ into a new token $AB$, it evaluates how this merge alters the total log-likelihood of the corpus. The change in log-likelihood is directly proportional to the pointwise mutual information (PMI) of the two tokens:

$$
\log \left( \frac{P(AB)}{P(A)P(B)} \right) = \log P(AB) - \log P(A) - \log P(B)
$$

By picking the pair that maximizes $\frac{\text{Count}(AB)}{\text{Count}(A) \times \text{Count}(B)}$, WordPiece greedily selects the merge rule that provides the greatest immediate increase (or smallest decrease) to the corpus likelihood.

---

### Concrete Tokenization Examples

To illustrate how these trained vocabularies behave in practice, let us examine how both BPE and WordPiece process a sample sentence once training is complete.

Assume both tokenizers have been trained on an English corpus containing a mix of common and rare words, resulting in the following simplified vocabularies:
* **BPE Vocabulary:** `["the", "cat", "walk", "ed", "strang", "ely", "s", "a", "b", "c", ...]` (plus the end-of-word marker `</w>`)
* **WordPiece Vocabulary:** `["the", "cat", "walk", "##ed", "strang", "##ely", "##e", "a", "b", "c", ...]`

#### 1. BPE Tokenization Example
BPE tokenizes text in a **bottom-up** fashion by looking up its learned *merge rules* in the exact chronological order they were discovered during training. 

Input word: `"strangely"`
1. The word is split into individual characters with an end-of-word marker: `s  t  r  a  n  g  e  l  y  </w>`
2. The tokenizer scans its merge rules list. The earliest rule matching any pair here is the one that created `strang`. The sequence becomes: `strang  e  l  y  </w>`
3. The next highest-priority merge rule in the vocabulary is for `ely`. The sequence becomes: `strang  ely  </w>`
4. No further valid merge rules apply.
* **Final BPE Tokens:** `["strang", "ely</w>"]` (often rendered simply as `["strang", "ely"]` with space indicators like `Ġstrang`, `ely` depending on the exact implementation).

#### 2. WordPiece Tokenization Example
WordPiece tokenizes text in a **top-down, greedy** fashion using an algorithm called **MaxMatch** (Maximum Matching). Instead of executing merge rules chronologically, it scans an isolated word from left to right, hunting for the longest string starting from the current position that exists in the vocabulary.

Input word: `"strangely"`
1. It looks at the whole string `"strangely"`. It is not in the vocabulary.
2. It chops letters off the end until it finds a match: `"strangely"` $\rightarrow$ `"strangel"` $\rightarrow$ `"strange"` $\rightarrow$ **`"strang"`** (Match found!).
3. The remaining substring is `"ely"`. Because it is a continuation of a word, the tokenizer looks for pieces prefixed with `##`.
4. It looks for `"##ely"`. (Match found!).
* **Final WordPiece Tokens:** `["strang", "##ely"]`

---

### Why BPE Lacks an Unknown Token, But WordPiece Has One

In practice, WordPiece relies on an explicit **`[UNK]`** (unknown) token to handle unexpected characters, whereas modern implementations of BPE (such as those used by GPT-4 or LLaMA) guarantee a **0% out-of-vocabulary rate** without needing an unknown fallback. This difference stems from how they construct their initial base vocabularies:

#### WordPiece and `[UNK]`
WordPiece initializes its base alphabet using characters (Unicode code points) discovered within its training corpus. If a user inputs text during inference that contains a completely novel character—such as an emoji, a specialized mathematical symbol, or a character from a foreign alphabet that was entirely absent from the training data—WordPiece cannot decompose it. 

When the MaxMatch algorithm encounters a character or substring for which no individual base character exists in the vocabulary, the entire lookup routine fails. Because it cannot output a valid sequence of tokens, it maps the unrecognized token to a catch-all sequence: `[UNK]`.

#### BPE and Byte-Level Representation
Modern NLP implementations of BPE circumvent this limitation by shifting the base vocabulary down from the character level to the **byte level**. 

Instead of initializing the alphabet with thousands of unique Unicode characters, Byte-Level BPE (BBPE) initializes its base vocabulary with exactly **256 fundamental tokens**, representing every possible value of a raw byte ($0\text{x}00$ through $0\text{x}\text{FF}$). Any text string, no matter how rare or bizarre the character, can be encoded into a sequence of UTF-8 bytes. Because all 256 bytes are permanently locked into the base vocabulary, the tokenizer can *always* fall back to individual byte tokens if it encounters a character it has never seen before. It is mathematically impossible to produce a sequence that cannot be decomposed, rendering an `[UNK]` token obsolete.

---

### Standard WordPiece vs. Fast WordPiece

When exploring WordPiece implementations, a distinction is frequently made between **Standard WordPiece** and **Fast WordPiece** (often associated with Google's linear-time implementations and Hugging Face's Rust-based `tokenizers` library). This distinction centers purely on the execution efficiency of the tokenization step rather than the underlying vocabulary rules.

#### Standard WordPiece ($\mathcal{O}(n^2)$ or $\mathcal{O}(nm)$ Complexity)
The traditional MaxMatch algorithm used in standard WordPiece requires nested iterations over the input text:
* It sets a pointer at the beginning of the word and scans forward to check if the substring matches a vocabulary entry.
* If a match fails, it backtracks, shortens the candidate substring by one character, and tries again.

If a word has a length of $n$ and the maximum token length in the vocabulary is $m$, this backtracking search results in a worst-case time complexity of $\mathcal{O}(n^2)$ or $\mathcal{O}(nm)$ per word. When processing billions of tokens during large-scale pre-training or high-throughput real-time inference, this computational overhead becomes a noticeable bottleneck.

#### Fast WordPiece ($\mathcal{O}(n)$ Complexity)
Fast WordPiece eliminates backtracking entirely by modeling the vocabulary as a specialized data structure known as a **Trie** (a prefix tree), augmented with advanced search mechanisms inspired by the **Aho-Corasick** string-matching algorithm.

* **Failure Links & Failure Pops:** In Fast WordPiece, every node in the vocabulary trie contains precomputed "failure links". If the tokenizer is stepping through the tree matching characters (e.g., tracking `s` $\rightarrow$ `t` $\rightarrow$ `r` $\rightarrow$ `a`) and hits a character that fails to match an edge, it does not reset and backtrack to the beginning of the word. Instead, it immediately follows a failure link to another precomputed node in the trie where a valid sub-match exists, emitting the tokens along the way ("failure pops").
* **Single-Pass End-to-End Processing:** While standard tokenizers use a two-step approach—first splitting an entire sentence into raw words using whitespaces/punctuation (pre-tokenization) and then passing each word individually to the subword loop—Fast WordPiece handles both simultaneously. It streams the entire raw sentence text through the trie in a **single linear pass**.

As a result, Fast WordPiece processes text in **strict $\mathcal{O}(n)$ time complexity** relative to the sentence length $n$, executing up to 5 to 8 times faster than traditional implementations without altering the final tokenized output.

### **The Golden Rule: Once Learned, It is Frozen**

This brings us to a crucial rule: **Token IDs are entirely fixed and never change during the LLM's life.** 

Once the tokenizer algorithm finishes its training phase and locks in its vocabulary table, the ID for a specific token is set in stone. The token `"pug"` will *always* map to the integer ID `4`. Whether the model is being trained to understand language or is generating text for a user, this lookup table remains completely frozen.

**The Multi-Step Role of the Tokenizer**

While we often use "tokenization" to refer to the whole text-to-ID pipeline, the tokenizer object in modern NLP libraries (like Hugging Face) actually handles several distinct steps in sequence:

1. **Normalization:** Cleaning the text (e.g., stripping whitespace, handling casing, removing accents).
2. **Pre-tokenization:** Splitting the raw text into rough boundaries, usually by spaces or punctuation.
3. **Model Tokenization:** Applying a core algorithmic rule (like BPE) to split words into subwords.
4. **Post-Processing:** Injecting model-specific special tokens (such as `[CLS]`, `[SEP]`, or `<|endoftext|>`).
5. **ID Mapping:** Converting the final array of tokens into their corresponding Token IDs.




### Beyond Words and Characters: Advanced Tokenization
While word, subword, and character tokenization are standard for text, the frontier of AI relies on alternative methods:

* **Byte-level Tokenization:** Instead of characters, text is broken down into raw UTF-8 bytes (ranging from 0 to 255). Used by modern models like GPT-4, this guarantees the model can process *any* language, symbol, or emoji without ever generating an "Unknown Token" (`[UNK]`) error.
* **Visual Patching:** In Vision-Language models, images are sliced into a grid of uniform pixel squares (e.g., $16 \times 16$ patches). Each patch is projected linearly and treated exactly like a text token.
* **Structural/Graph Tokenization:** Used in highly specialized models for code parsing (ASTs) or biology (DNA sequencing and SMILES strings for molecular chemistry), where structure dictates token boundaries more than spacing does.

---

### The One-Pass Lifecycle: Training vs. Inference
To see how these concepts fit together, let's trace a single pass from text to embedding vector, highlighting how the mechanics differ when a model is predicting versus when it is learning.

#### During Inference (Forward Pass)
1. **Input:** The user types `"Deep learning"`.
2. **Tokenizer Loop:** The tokenizer splits the text into tokens and looks up their integers in its dictionary, producing `[2534, 4671]`.
3. **Index Retrieval:** These IDs are passed to the model's Embedding Layer. The layer treats the IDs as row indices. If an ID is `2534`, it extracts row `2534` from its internal weight matrix.
4. **Output:** The model receives a sequence of static, floating-point vectors ready for Transformer computation.

#### During Training (Forward + Backward Pass)
1. **Tokenizer Loop:** The process is identical to inference. The tokenizer text-to-ID mapping is static and **does not update** during model training.
2. **Index Retrieval:** The model grabs the vectors for `[2534, 4671]`.
3. **Optimization:** The model passes these vectors forward, makes a prediction, computes the error (loss), and calculates mathematical gradients via backpropagation.
4. **Weight Update:** The numerical values *inside* the rows `2534` and `4671` of the embedding matrix are slightly modified. Over millions of steps, these vectors shift positions in vector space to accurately capture the semantic meaning of "Deep learning".

---

### Engineering Distinction: Tokenizers vs. Embeddings in Frameworks
From a software engineering perspective, it is critical to realize that **tokenizing and embedding are separate architectural modules.**

In frameworks like Hugging Face `transformers`:
* **The Tokenizer (`AutoTokenizer`)** is a **CPU-bound** pre-processing utility. It handles raw Python strings and transforms them into standard integer tensors. It operates entirely outside of the neural network and contains no learnable weights.
* **The Model (`AutoModel`)** houses the embedding layer (`nn.Embedding`), which is a **GPU/TPU-bound** neural network component. It is a matrix of trainable weights that converts those integer tensors into continuous vector representations.

In the following figure, some LLM tokenization methods are mentioned.
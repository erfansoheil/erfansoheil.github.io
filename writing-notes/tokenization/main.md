---
layout: post
title: "Tpkenization VS Token Representation"
---

The idea of this article came to my mind when I was readig about LLM archtiectures and was trying to understand the difference between **Token**, **Tokenization**, **Token IDs** and **Token Represenations**. At first all of these look the same to me and it took some time for me to categorize them in my mind. This articel is a try to create a mental map for someone who wants to go deep in these topics. 

In this artcile I first discuss about *Token* and *Tokenization* process in LLMs. Then we disucss about *Token Representations* process. Before that lets recall the trasdformers architecture since it is our building block of thsi article. 

![Transformers Architecture](./assets/images/tr_arch.png)

## First Part: Tokenization


Throughout this article, the word **token** is used frequently. A token is a discrete unit of information processed by a model. In domain of language models, a token usually represents a word, part of a word, a punctuation mark, or another piece of text. In other domains, however, tokens may represent image patches, audio segments, time-series windows, and so on.

Before a model can process an input, it must first convert it into a sequence of tokens. The tokens themselves are not directly understood by the neural network. Instead, each token is mapped to a unique integer called a **token ID**. These token IDs are then transformed into vectors through an **embedding layer**, producing numerical representations that the model can process.

As a result, the pipeline looks as follows:

<center>Input → Tokens → Token IDs → Embeddings (Vectors)</center>

The process of converting an input into tokens is called **tokenization**, while the process of converting token IDs into vectors is called **embedding**.

---

### Why do we need Token IDs after Tokenization?
It is tempting to think that once we split a sentence into string-based tokens like `["deep", "learning"]`, the hard part is over. However, computers and neural networks cannot perform mathematical operations on raw text strings. They operate strictly on matrices and vectors of floating-point numbers. 

Token IDs act as the bridge. By assigning every unique token in our vocabulary a fixed, unique integer index (e.g., `"deep"` $\rightarrow$ `2534`), we create a structured lookup system that the neural network can interact with mathematically.

### Are Token IDs Fixed or Learned?
A common point of confusion is whether these IDs change as the model learns. **Token IDs are entirely fixed.** Once a tokenizer is trained and its vocabulary dictionary is locked in, the ID for a specific token never changes. For instance, in a BERT tokenizer, the token `"the"` will always map to the integer ID `1996`. What *does* change during the LLM's training are the **embedding vectors** associated with those IDs. The integer ID is simply a stable index used to point the model to the correct row in its embedding matrix.

### The Multi-Step Role of the Tokenizer
While we often use "tokenization" to refer to the whole text-to-ID pipeline, the tokenizer object in modern NLP libraries (like Hugging Face) actually handles several distinct steps in sequence:

1. **Normalization:** Cleaning the text (e.g., stripping whitespace, handling casing, removing accents).
2. **Pre-tokenization:** Splitting the raw text into rough boundaries, usually by spaces or punctuation.
3. **Model Tokenization:** Applying a core algorithmic rule (like BPE) to split words into subwords.
4. **Post-Processing:** Injecting model-specific special tokens (such as `[CLS]`, `[SEP]`, or `<|endoftext|>`).
5. **ID Mapping:** Converting the final array of tokens into their corresponding Token IDs.

---

### How is a Tokenizer Trained?
Unlike the primary Language Model, which learns grammar and facts via backpropagation, a tokenizer is trained using algorithmic statistics on a large corpus of raw text. The goal is to build an optimal vocabulary of a target size (e.g., 32,000 or 50,257 tokens). 

Take **Byte-Pair Encoding (BPE)**, one of the most popular tokenization algorithms, as an example:

* **Initialization:** The vocabulary starts with individual characters (the alphabet, numbers, and basic punctuation). Every word in the training text is split into characters.
* **Iterative Merging:** The algorithm scans the text to find the most frequently occurring pair of adjacent tokens (e.g., if `e` and `s` appear next to each other more than any other pair, they are merged into a new token: `es`).
* **Growing the Vocab:** This single new token `es` is added to the vocabulary. The algorithm repeats this counting and merging process over and over until the vocabulary reaches its pre-defined capacity.

Other methods like **WordPiece** or **Unigram** use slightly different statistical criteria (like maximizing data likelihood rather than raw frequency), but the underlying philosophy remains data-driven vocabulary building.

---

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
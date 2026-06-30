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

**Byte-Pair Encoding (BPE)** is a bottom-up tokenization algorithm. It begins by looking at text at the absolute lowest level (characters) and systematically builds whole words and subwords based on frequency. It is the core tokenizer architecture behind models like GPT-2, GPT-4, LLaMA, and Mistral.

#### **How BPE is Trained (Step-by-Step)**

1. **Tokenize into Characters:** The training algorithm splits every single word in the text dataset into its individual characters. A special end-of-word symbol (like `</w>`) is added to remember where whole words naturally end.
2. **Count Neighboring Pairs:** The algorithm counts every instance of two characters appearing right next to each other. For example, across millions of words, how many times does `d` sit right next to `e`?
3. **Merge the Champion Pair:** The single most frequent pair of adjacent characters is merged into a brand-new, combined token. If `d` + `e` has the highest count, a new token `de` is born.
4. **Iterate and Expand:** The algorithm rescans the entire dataset, treating `de` as a single unit now, and counts the next most frequent adjacent pair (which could be `de` + `r`). This loop repeats until the target vocabulary size is filled.

 **Method 2: WordPiece**

**WordPiece** is very similar to BPE in that it starts with individual characters and iteratively merges them upward into larger units. However, instead of simply picking the most *frequent* pair, WordPiece uses a statistical calculation to pick the pair that makes the language most predictable. It is famously used by Google's BERT model.



**How WordPiece is Trained (Step-by-Step)**

1. **Initialize Base Vocabulary:** Just like BPE, it starts by extracting all basic characters and punctuation marks from the training text corpus.
2. **Build a Candidate Scoring Matrix:** The algorithm looks at all adjacent pairs, but instead of counting pure frequency, it scores them using a formula:
   $$\text{Score} = \frac{\text{Frequency of Pair }(A, B)}{\text{Frequency of } A \times \text{Frequency of } B}$$
   *Why this matters:* If the letter `z` and `q` are rare, but whenever they appear they are *always* next to each other, this score will be incredibly high, meaning they belong together as a single token.
3. **Merge the Highest-Scoring Pair:** The pair that maximizes this statistical likelihood is merged into a new subword unit.
4. **Iterate:** The corpus is updated, and the scoring system runs again until the user-defined vocabulary size is reached.

 **Method 3: Unigram Language Modeling**

Unlike BPE and WordPiece, which start small (characters) and build up, **Unigram** goes in reverse. It is a top-down algorithm that starts with a massive, over-inflated vocabulary list and systematically weeds out the least useful tokens. It is widely used in Google's T5 model and SentencePiece implementations.

**How Unigram is Trained (Step-by-Step)**

1. **Initialize a Massive Vocabulary:** The algorithm extracts all full words and highly common substrings from the training text, creating a massive initial vocabulary list that is much larger than the desired final size.
2. **Train a Probabilistic Model:** It estimates the probability of every single token occurring in the text dataset assuming all tokens are independent (a unigram model).
3. **Calculate "Loss" for Every Token:** The algorithm simulates what would happen if it deleted a specific token from the dictionary. If removing a token (like `"ing"`) forces the tokenizer to chop words into clumsy, inefficient pieces, that token has a high "loss value" (meaning it is highly useful). If removing a token makes almost no difference, its loss value is low.
4. **Prune the Bottom 10-20%:** It ranks all tokens by their usefulness and permanently drops the bottom percentage of tokens with the lowest scores.
5. **Repeat Until Trimmed:** Steps 2 through 4 are repeated. The vocabulary shrinks iteration by iteration until it is trimmed down exactly to the target vocabulary size.


**A Concrete Training Example**

Imagine our training text dataset is tiny, consisting only of these three words repeated multiple times: `"hug"`, `"pug"`, and `"pun"`.

1. **Base Vocab Created:** `["h", "u", "g", "p", "n"]`
2. **First Scan & Merge:** The algorithm notices that the pair `["u"]` and `["g"]` appears frequently in "hug" and "pug". It merges them into a new token $\rightarrow$ `─> "ug"`.
3. **Second Scan & Merge:** It notices `["p"]` and the new `["ug"]` chunk are highly frequent together. It merges them into a new token $\rightarrow$ `─> "pug"`.

By the end of training, the static lookup dictionary looks like this:

| Assigned Token ID | String Token | Type |
| :--- | :--- | :--- |
| **0** | `"h"` | Base Character |
| **1** | `"u"` | Base Character |
| **2** | `"g"` | Base Character |
| **3** | `"ug"` | Subword Chunk |
| **4** | `"pug"` | Full Word |

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
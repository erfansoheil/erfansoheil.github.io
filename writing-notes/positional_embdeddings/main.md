
# Positional Embeddings

In this article, we will explore the concept of positional embeddings in Transformer architectures, break down the different methods used to calculate them, and discuss when to use each approach. Before diving into the positional aspect, however, we need to understand the foundational concept of modern **embeddings**.

## Embdeddings

Every piece of text we input into a model needs to be translated into a language the machine can process, and computers only understand numbers. An **embedding model** is a tool that translates human language (like words, sentences, images, etc) into a structured set of numbers called a vector.

What is the purpose of an embedding?
Categorization and similarity.

Suppose you have a mountain of messy, unorganized text inputs that you need to sort out. Think of it like having a massive basket filled with a chaotic mix of fruits. Ideally, you want to separate them so that similar fruits end up together.

In the real world, data isn't quite as simple or tangible as a physical basket of fruit. To sort text, we use embedding models to map words into a mathematical space based on their semantic similarities. This ensures that words with related meanings sit close to each other—similar to grouping all the citrus fruits in one corner and berries in another.

The figure below illustrates how an embedding model plots words as vectors based on their characteristics, automatically clustering similar concepts together:

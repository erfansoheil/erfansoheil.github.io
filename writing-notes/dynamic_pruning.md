---
---
# Dynamic Pruning

When we talk about **dynamic pruning**, we refer to the process of pruning and compressing a model's architecture *during* the training phase, rather than as a post-training optimization step. 

While this approach offers significant advantages, it also comes with its own set of unique limitations. Throughout this article, we will explore these trade-offs by reviewing several papers in the field. Specifically, we will focus on the sections of these papers that dedicate themselves entirely to in-training (dynamic) architectural pruning.

For each paper discussed, we will break down:

* **The Core Idea:** The intuition and concept behind the proposed method.
* **The Mathematical Foundation:** The underlying equations and theory driving the pruning mechanism.
* **Implementation Snippets:** A brief code snippet demonstrating the technique. *(Note: Where official code is publicly available, a link to the original repository will be provided).*

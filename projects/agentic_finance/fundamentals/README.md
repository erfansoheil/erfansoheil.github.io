# Financial Fundamentals Tool-Calling Model

This repository contains an initial working project for training and testing a specialized financial tool-calling model. The model converts natural-language requests about company fundamentals into a structured JSON function call.

The project is part of a larger agentic-system direction. In the intended architecture, an orchestrator agent talks with the user, decides which capability is needed, and calls specialized models or tools at the right time. This repository focuses on the first specialized model: a function-calling model for fundamentals retrieval.

## Core Idea

The model should not answer financial questions directly. Instead, it should translate a user request into a valid tool call.

Example input:

```text
Please show the free cash flow and diluted shares of Amazon from 2018 to 2023.
```

Expected output:

```json
{
  "action": "call",
  "function": "get_fundamentals",
  "arguments": {
    "queries": [
      {
        "symbols": ["Amazon"],
        "metrics": ["Free Cash Flow", "Diluted Shares"],
        "start_year": 2018,
        "end_year": 2023
      }
    ]
  }
}
```

## High-Level Workflow

```mermaid
flowchart LR
    A[Generate data] --> B[Normalize queries]
    B --> C[Validate JSONL]
    C --> D[Build chat dataset]
    D --> E[Fine-tune LoRA adapter]
    E --> F[Test inference]
```

The project includes notebooks for:

- Generating synthetic financial tool-calling examples.
- Rephrasing repetitive queries to improve linguistic diversity.
- Preparing and validating a Hugging Face compatible dataset.
- Fine-tuning a compact Qwen model with LoRA.
- Running inference against the fine-tuned adapter.

## Dataset Design

Each training example pairs a natural-language request with a deterministic JSON tool call.

Conceptual fields:

| Field | Meaning |
| --- | --- |
| `query` | User-style financial request. |
| `output_str` | JSON function call as text, used as the training target. |
| `output_json` | Parsed JSON object for validation. |
| `metadata` | Companies, metrics, years, and generation trace data. |

The dataset includes both easy and intermediate examples.

Easy examples ask for the same metrics across one or more companies. Intermediate examples use grouped company-metric relationships, where different company groups may require different metrics. This helps teach the model not to mix company and metric assignments.

## Training Approach

The current training approach uses:

- A compact Qwen base model.
- LoRA adapters instead of full fine-tuning.
- Chat-style formatting with system, user, and assistant messages.
- A system prompt that constrains the model to valid JSON function calling.
- Prompt-token masking so training focuses on the assistant JSON output.
- Grouped train/test splitting to reduce paraphrase leakage.

The fine-tuned model is intended to be called by a future orchestrator when the user request clearly requires fundamentals data.

## Repository Contents

At a high level, the repository contains:

- Data generation notebooks.
- Data normalization and validation workflow.
- Training and inference notebooks.
- JSONL datasets and Hugging Face dataset artifacts.
- A saved LoRA adapter for the current model version.

Exact artifact locations may change as the project evolves, so the most important entry points are the notebooks and the generated documentation in this repository.

## Current Status

This is an initial working project. It demonstrates the complete first pipeline from data generation to inference:

- Synthetic examples have been generated.
- Query normalization has been applied.
- Training data has been validated and formatted.
- A LoRA adapter has been trained.
- Basic inference tests are available.

The project is not yet a complete production agentic system. The orchestrator, broader tool registry, stronger evaluation, and deployment workflow are future stages.

## Future Development

Future plans will be expanded as the project direction becomes more detailed. The likely next development areas are:

- Build the orchestrator layer that decides when to call this model.
- Add stricter JSON schema validation at inference time.
- Add automated evaluation for exact-match structure, valid JSON, and company-metric preservation.
- Add more financial tools and route between them.
- Improve rejection or fallback behavior for irrelevant requests.
- Add real user query examples alongside synthetic data.
- Package the trained model and inference code into a cleaner service interface.

## Documentation

Additional project documents:

- `REPORT.md`: Full professional report on the procedure and design choices.
- `TECHNICAL_NOTES.md`: Intuition-focused technical notes on tokenization, prompts, JSON tool calling, training, inference, and hardware.


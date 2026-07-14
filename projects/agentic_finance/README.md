# Agentic Finance

Agentic Finance is the parent project for building an agentic financial assistant. The central idea is to separate conversation, reasoning, tool routing, and specialized model behavior into clear components.

The system is designed around an orchestrator agent. The orchestrator has the conversation with the user, understands the user's intent, decides which capability is needed, and then calls the correct model or tool. Some requests may be answered directly. Other requests may require structured tool calls, data retrieval, calculation, or a specialized submodel.

## Overall Vision

The target architecture is not a single model trying to do everything. It is a coordinated system where each component has a clear role.

<!-- 
```mermaid
flowchart TD
    U[User] --> O[Orchestrator Agent]
    O --> R{Decision}
    R -->|General conversation| A[Direct response]
    R -->|Financial data request| F[Specialized tool-calling model]
    R -->|External capability| T[Tool or service]
    F --> J[Structured JSON call]
    J --> D[Financial data tool]
    D --> O
    T --> O
    A --> U
    O --> U
``` -->


```mermaid

flowchart LR
    %% Define nodes with distinct shapes for better visual hierarchy
    U([User])
    O[Orchestrator Agent]
    R{Decision}
    A[Direct response]
    F[Specialized tool-calling model]
    J[Structured JSON call]
    T[Tool or service]
    D[(Financial data tool)]

    %% Initial request
    U -->|Query| O
    O --> R

    %% Branching logic
    R -->|General conversation| A
    R -->|External capability| T
    R -->|Financial data request| F

    %% Financial sub-process
    F --> J
    J --> D

    %% Feedback loops (Return paths to Orchestrator)
    D -.->|Returns data| O
    T -.->|Returns data| O

    %% Output paths to User
    A -->|Direct output| U
    O -->|Final output| U
```



The orchestrator is responsible for:

- Managing the conversation with the user.
- Deciding whether the request needs a model, a tool, or a direct answer.
- Calling specialized models when a constrained output is required.
- Calling external tools when data or execution is needed.
- Combining tool results into a final user-facing response.

Specialized models are responsible for narrow, reliable transformations. For example, a model can translate a natural-language financial request into a valid JSON function call. This makes each model easier to train, evaluate, and improve.

## Current Implementation

The first implemented subproject is the fundamentals tool-calling model. It focuses on converting user requests about company fundamentals into a structured call to a fundamentals retrieval function.

This subproject demonstrates the first complete pipeline:

- Generate synthetic financial tool-calling data.
- Normalize repetitive queries to improve language variety.
- Validate the JSONL dataset.
- Convert examples into chat-training format.
- Fine-tune a compact model with LoRA.
- Run inference with the trained adapter.

The current implementation is an initial working project. It is not yet the full production orchestrator, but it provides one specialized model that can later be called by the orchestrator.

## Repository Structure

```text
agentic_finance/
  README.md
  fundamentals/
    README.md
    REPORT.md
    TECHNICAL_NOTES.md
    notebooks and related artifacts
```

The parent README explains the overall project direction. Each subproject should contain its own README, report, technical notes, notebooks, and implementation artifacts.

## Design Principles

### 1. Separation of Responsibility

The orchestrator should not contain every specialized behavior. It should route work to the right component. Specialized models should solve bounded tasks with clear inputs and outputs.

### 2. Structured Tool Use

When a user request requires external data or execution, the system should produce structured calls rather than vague text. JSON is used as a practical function-calling format because it can be validated and executed by downstream tools.

### 3. Faithfulness Before Fluency

For financial workflows, preserving the user's exact intent is more important than producing creative language. Company names, metrics, time ranges, and associations must be preserved before any final answer is generated.

### 4. Modular Growth

New financial abilities should be added as modules: fundamentals, prices, filings, portfolio analysis, screening, and other tools can each have their own data, evaluation, and model behavior.

## Development Roadmap

The next development stages are expected to include:

- Build the orchestrator layer.
- Define routing rules between direct answers, specialized models, and tools.
- Add strict JSON schema validation before tool execution.
- Add automated evaluation for tool-call accuracy.
- Connect model-generated calls to real financial data retrieval.
- Add more specialized financial tools and submodels.
- Add stronger handling for ambiguous, unsupported, or irrelevant requests.
- Package inference behind a clean service interface.

Future plans can be expanded as the architecture becomes more detailed.

## Documentation

The fundamentals subproject currently contains the detailed report and technical notes for the first implemented model. Future subprojects should follow the same pattern so the overall system remains understandable as it grows.
# Business Context & Architectural Vision

> [!IMPORTANT]
> This document serves as the source of truth for the strategic direction of this project. All refactoring and architectural decisions MUST align with the principles outlined here.

## 1. Original Purpose
The script was originally designed to:
1.  Take a repository (as a ZIP archive).
2.  Merge it into a single text file.
3.  Send it to the Gemini model (leveraging its large context window).
4.  Receive documentation or analysis back from the model.

## 2. Current State
Currently, it operates as a **console utility** with a configuration file (`config.yaml`). It mixes core logic with CLI input/output operations.

## 3. Future Vision
The goal is to evolve this tool into an **independent, representation-agnostic module**.

-   **Architecture**: It should be capable of running as a **microservice** or **AWS Lambda** function.
-   **Interface**: It should support various interfaces, such as:
    -   CLI (current)
    -   FastAPI / HTTP Endpoint
-   **Contract**:
    -   **Input**: A ZIP file + Optional Configuration.
    -   **Output**: Generated artifacts (documentation, reports, etc.).

## 4. Refactoring Constraints
To achieve the future vision, the following constraints apply to all code changes:

-   **Separation of Concerns**: The **Core Logic** must be completely decoupled from **Input/Output** operations.
-   **No Side Effects in Core**: The core processing functions should NOT print to stdout/stderr.
    -   *Bad*: `print("Processing file...")` inside a logic function.
    -   *Good*: Return a status object or yield events that the caller (CLI or API) can decide how to handle (e.g., log to file, stream to socket, print to console).
-   **Modularity**: The code should be structured so that the processing engine can be imported and used by other Python scripts without modification.

# Guide: Data Flow - fb56082f-02fc-47db-a636-69fd6aebc3df

> How data flows through the system.

---

## Overview

Data flows through the following layers:

1. **Input**: APIs, events, commands
2. **Validation**: Data verification
3. **Processing**: Business logic
4. **Persistence**: Database, cache
5. **Output**: Responses, events


## Entry Points

- `src/main.py`
- `src/config.py`
- `src/__init__.py`
- `src/templates/doc_generator.py`
- `src/templates/__init__.py`
- `src/graph/nodes.py`
- `src/graph/__init__.py`
- `src/llm/embeddings.py`
- `src/llm/provider.py`
- `src/llm/__init__.py`

---

*Back to [Index](./00_INDEX.md)*

# 🤖 Rules for AI Agents - fb56082f-02fc-47db-a636-69fd6aebc3df

> Specific guidelines for AI agents working with this codebase.
> Read this document **BEFORE** making any code changes.

**Architecture:** Monolithic with a Service-Oriented approach  
**Main Language:** Python  
**Framework:** N/A  
**Last Updated:** February 2026

---

## 📋 Index

- [Project Context](#project-context)
- [Documentation Navigation](#documentation-navigation)
- [File Structure](#file-structure)
- [Code Patterns](#code-patterns)
- [Naming Conventions](#naming-conventions)
- [Dependencies and Imports](#dependencies-and-imports)
- [Anti-Patterns](#anti-patterns)
- [Checklist Before Committing](#checklist-before-committing)

---

## Project Context

### About the Project

This project follows a **Monolithic with a Service-Oriented approach** architecture, 
implemented primarily in **Python**.

### Detected Patterns

- No specific patterns detected

### Technology Stack

- Stack not identified


---

## Documentation Navigation

### Golden Rule
```
ALWAYS start reading in this order:
1. This file (AGENT_RULES.md) ─ You are here
2. /docs/usage/00_INDEX.md ─ To know which guide to follow
3. The specific guide for your task ─ Only ONE at a time
4. Documents from /context/ ─ Only when referenced
```

### Decision Tree

```
What do you need to do?
│
├─► Create new component/module?
│   └─► Read: /docs/usage/02_CODE_PATTERNS.md
│
├─► Understand data flow?
│   └─► Read: /docs/usage/03_DATA_FLOW.md
│
├─► Add new feature?
│   └─► Read: /docs/usage/04_ADDING_FEATURES.md
│
├─► Understand general architecture?
│   └─► Read: /docs/charts/01_ARCHITECTURE_OVERVIEW.md
│
└─► Debug or maintenance?
    └─► Read: /docs/charts/06_DEPENDENCY_GRAPH.md
```

### ⚠️ DON'T DO
- Don't load all documents at once
- Don't ignore documentation and go straight to code
- Don't modify code without checking patterns

---

## File Structure

### Modular Monolith

```
src/
├── modules/
│   ├── module-a/
│   └── module-b/
├── shared/        ─► Shared code
└── infrastructure/
```

**Rules:**
1. Modules should be as independent as possible
2. Inter-module communication via public interfaces
3. Shared contains only generic utilities


---

## Code Patterns

### General Patterns

Follow existing patterns in the code. Before implementing:

1. Look for similar implementations in the codebase
2. Maintain consistency with existing style
3. Check `/docs/context/PATTERNS.md`


---

## Naming Conventions

### Python

```python
# Files and modules: snake_case
my_module.py
my_service.py

# Classes: PascalCase
class MyService:
    pass

# Functions and variables: snake_case
def my_function():
    my_variable = 1

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Private: prefix _
def _internal_method():
    pass
```


---

## Dependencies and Imports

### Import Order

1. **Standard libraries** (built-in)
2. **External dependencies** (third-party)
3. **Internal imports** (from project)

### Python

```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
from fastapi import FastAPI
import numpy as np

# 3. Internal
from .my_module import my_function
from src.services import MyService
```


---

## Useful Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or
poetry install
# or  
pdm install

# Run tests
pytest
pytest -v  # verbose

# Type checking (if using mypy/pyright)
mypy src/

# Format code
black src/
isort src/

# Lint
ruff check src/
flake8 src/
```


---

## Anti-Patterns

> ⛔ What **NOT** to do in this project

### Avoid These Mistakes

1. **Don't duplicate code**
   - Before creating something new, check if it already exists
   - Check `/docs/context/COMPONENTS.md`

2. **Don't ignore architecture**
   - Respect layers and responsibilities
   - Don't create circular dependencies

3. **Don't hardcode values**
   - Use configurations and constants
   - Sensitive data goes in environment variables

4. **Don't make giant commits**
   - Small and focused commits
   - One feature per commit

5. **Don't ignore types**
   - If the project uses TypeScript/types, maintain typing
   - Avoid `any` or overly generic types

6. **Don't modify configuration files unnecessarily**
   - `package.json`, `pyproject.toml`, etc.
   - Only when strictly necessary


---

## Checklist Before Committing

```
□ Does the code follow patterns documented in /docs/context/PATTERNS.md?
□ Are new components in the correct folder?
□ Do imports follow project conventions?
□ Is there no duplicate code of something that already exists?
□ Were tests added/updated?
□ Was documentation updated if needed?
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `src` | N/A |

---

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

*Automatically generated by Code Analysis Agent*  
*Last updated: February 2026*

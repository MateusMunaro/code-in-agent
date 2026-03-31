# Guide: Getting Started - fb56082f-02fc-47db-a636-69fd6aebc3df

> Step by step to setup and understand the project.

---

## Prerequisites

- Python 3.10+
- pip or poetry
- Git


## Environment Setup

```bash
# Clone
git clone <repository-url>
cd <project>

# Virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Dependencies
pip install -r requirements.txt
# or: poetry install
```


## First Steps

1. Clone the repository
2. Install dependencies
3. Configure environment variables (if applicable)
4. Run the project

## Useful Commands

```bash
# Run
python main.py
# or: python -m src

# Tests
pytest

# Lint
ruff check .
```


---

*Back to [Index](./00_INDEX.md)*

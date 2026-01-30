# Code Indexer AI Agent - Python Worker

Python service for intelligent code analysis using LangGraph.

## Features

- 🔍 **Tree-sitter parsing** for accurate AST analysis
- 🧠 **LangGraph agent** with reasoning loop for deep code understanding
- 🔄 **Multi-LLM support** (OpenAI, Anthropic, Google, Ollama)
- 📊 **Dependency graph** construction
- 📝 **Documentation generation** with architecture patterns detection

## Setup

### Prerequisites

- Python 3.11+
- Redis running
- Supabase project configured

### Installation

```bash
cd agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Running

```bash
python -m src.main
```

## Architecture

```
agent/
├── src/
│   ├── main.py              # Entry point, Redis listener
│   ├── config.py            # Configuration management
│   ├── services/
│   │   ├── git_service.py   # Repository cloning
│   │   ├── parser_service.py # Tree-sitter AST parsing
│   │   └── graph_builder.py # Dependency graph construction
│   ├── llm/
│   │   ├── provider.py      # LLM factory pattern
│   │   └── embeddings.py    # Vector embeddings
│   └── graph/
│       ├── state.py         # Agent state definition
│       ├── nodes.py         # LangGraph nodes
│       └── graph.py         # Graph orchestration
└── requirements.txt
```

## LangGraph Agent Flow

```
┌─────────────────┐
│  ReadStructure  │ ← Reads folder structure, identifies files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Planning     │ ← Identifies architecture patterns
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│  Verification   │ ──► │ Read More?  │ ── Yes ──┐
└────────┬────────┘     └─────────────┘          │
         │                                        │
         │ No (confidence > 80%)                  │
         ▼                                        │
┌─────────────────┐                              │
│    Response     │ ← Generates documentation    │
└─────────────────┘                              │
         ▲                                        │
         └────────────────────────────────────────┘
```

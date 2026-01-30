# Templates de Documentação

Este módulo contém templates e geradores para criar documentação estruturada e amigável para agentes de IA.

## Estrutura

```
templates/
├── __init__.py              # Exports públicos
├── doc_structure.py         # Templates de estrutura de documentação
├── agent_guidelines.py      # Gerador de regras para agentes de IA
└── doc_generator.py         # Gerador principal de documentação
```

## Uso

### Gerar Documentação Completa (Pasta `docs/`)

```python
from src.templates import generate_documentation

# Gera estrutura completa de documentação
docs = generate_documentation(
    project_name="MeuProjeto",
    architecture_pattern="Clean Architecture",
    confidence=0.85,
    main_language="Python",
    files_read=files_list,
    patterns_detected=patterns_list,
    dependency_graph=dep_graph,
    output_format="full"  # Retorna dict de arquivos
)

# docs é um dict: {"docs/charts/00_INDEX.md": "conteúdo...", ...}
```

### Gerar Documentação Resumida (Um Arquivo)

```python
from src.templates import generate_documentation

# Gera um único arquivo markdown
documentation = generate_documentation(
    project_name="MeuProjeto",
    architecture_pattern="Clean Architecture",
    confidence=0.85,
    main_language="Python",
    files_read=files_list,
    patterns_detected=patterns_list,
    dependency_graph=dep_graph,
    output_format="summary"  # Retorna string
)
```

### Gerar Apenas Regras para Agentes

```python
from src.templates import generate_agent_guidelines

rules = generate_agent_guidelines(
    project_name="MeuProjeto",
    architecture_pattern="Clean Architecture",
    main_language="Python",
    framework="FastAPI",
    patterns_detected=patterns_list
)
```

## Estrutura de Documentação Gerada

```
docs/
├── charts/                      # 📊 Diagramas visuais (Mermaid)
│   ├── 00_INDEX.md             # Índice e navegação
│   ├── 01_ARCHITECTURE_OVERVIEW.md
│   ├── 02_CLASS_DIAGRAM.md
│   ├── 03_SEQUENCE_FLOWS.md
│   ├── 04_COMPONENT_DIAGRAM.md
│   ├── 05_DATA_FLOW.md
│   └── 06_DEPENDENCY_GRAPH.md
│
├── context/                     # 📚 Documentação de referência
│   ├── ARCHITECTURE.md
│   ├── COMPONENTS.md
│   ├── PATTERNS.md
│   └── TECH_STACK.md
│
├── usage/                       # 📖 Guias práticos
│   ├── 00_INDEX.md             # Árvore de decisão
│   ├── 01_GETTING_STARTED.md
│   ├── 02_CODE_PATTERNS.md
│   ├── 03_DATA_FLOW.md
│   └── 04_ADDING_FEATURES.md
│
├── implementations/             # 📝 Histórico (vazio inicialmente)
│
└── AGENT_RULES.md               # 🤖 Regras para agentes de IA
```

## Benefícios

1. **Navegação Eficiente**: Árvores de decisão guiam para o documento certo
2. **Economia de Contexto**: Agentes carregam apenas o necessário
3. **Diagramas Visuais**: Mermaid para entendimento rápido
4. **Regras Específicas**: Diretrizes claras para cada tipo de arquitetura
5. **Escalável**: Fácil adicionar mais documentação

## Arquiteturas Suportadas

O gerador detecta e cria regras específicas para:

- Clean Architecture
- MVC
- Hexagonal (Ports & Adapters)
- Microservices
- Monolith Modular
- Arquiteturas genéricas

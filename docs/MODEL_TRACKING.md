# Model Selection & Tracking System

## 📊 Overview

Sistema completo de controle, monitoramento e rastreamento do uso de modelos LLM durante a análise de código.

## ✨ Features Implementadas

### 1. **ModelUsageStats** — Rastreamento Detalhado

Cada modelo agora rastreia:

- **Invocations**: Número total de chamadas
- **Successes**: Chamadas bem-sucedidas
- **Failures**: Chamadas que falharam
- **Success Rate**: Taxa de sucesso (%)
- **Total Tokens**: Tokens consumidos (estimado)
- **Avg Latency**: Latência média em milissegundos
- **Last Used**: Timestamp da última invocação

```python
from agent.src.llm import MultiModelChat

chat = MultiModelChat()
# ... uso normal ...

# Ver estatísticas
stats = chat.get_usage_stats()
chat.print_usage_summary()
```

**Output exemplo**:
```
═══════════════════════════════════════════════════════════════════
📊 Model Usage Summary
═══════════════════════════════════════════════════════════════════
Session started: 2026-02-16 14:30:00
Total models used: 2

🔹 gemini-2.5-flash (google)
   Invocations: 15 | Success: 14 | Failed: 1
   Success Rate: 93.3%
   Total Tokens: 42,156
   Avg Latency: 1,234ms
   Last Used: 14:35:22

🔹 gemini-2.0-flash (google)
   Invocations: 1 | Success: 1 | Failed: 0
   Success Rate: 100.0%
   Total Tokens: 2,891
   Avg Latency: 890ms
   Last Used: 14:32:15

═══════════════════════════════════════════════════════════════════
```

---

### 2. **Health Checks** — Validação Pré-voo

Cada provider agora implementa `health_check()`:

- **OpenAI**: Verifica presença de API key + instanciação do cliente
- **Anthropic**: Verifica presença de API key + instanciação do cliente  
- **Google AI**: Verifica presença de API key + instanciação do cliente
- **Ollama**: Pinga o servidor local e lista modelos disponíveis

```python
from agent.src.llm import check_providers_health

health_status = check_providers_health()
# Saída:
# [Provider Health] ✅ google: Google AI provider configured
# [Provider Health] ❌ openai: OpenAI API key not configured
# [Provider Health] ❌ anthropic: Anthropic API key not configured
# [Provider Health] ❌ ollama: Cannot connect to Ollama at http://localhost:11434
```

---

### 3. **Smart Model Selection** — Priorização Inteligente

O sistema agora:

1. Verifica quais providers estão configurados
2. Seleciona o primeiro modelo disponível na lista de prioridade
3. Loga a decisão de seleção
4. Valida o modelo antes de usar

```python
from agent.src.llm import get_default_model, validate_model_before_use

# Seleção automática
model = get_default_model()
# [Model Selection] Available models: ['gemini-2.5-flash', 'gemini-2.0-flash']
# [Model Selection] ✅ Selected: gemini-2.5-flash

# Validação
is_valid, msg = validate_model_before_use(model)
if not is_valid:
    print(f"⚠️ {msg}")
```

**Priority list** (configurável em `provider.py`):
1. `gemini-2.5-flash` — Rápido e capaz
2. `gemini-2.0-flash` — Fallback rápido
3. `gemini-2.5-pro` — Fallback Pro
4. `gemini-3-flash` — Última geração
5. `gemini-3-pro` — Mais poderoso

---

### 4. **Auto-Retry with Fallback** — Resiliência Automática

Se uma chamada falhar, o sistema automaticamente:

1. Loga a falha
2. Registra nas estatísticas
3. Tenta novamente com `gemini-2.0-flash` (modelo de fallback)
4. Só lança exceção se o fallback também falhar

```python
chat = MultiModelChat("gemini-2.5-pro")

# Se gemini-2.5-pro falhar, automaticamente tenta gemini-2.0-flash
response = await chat.ainvoke(messages, retry_on_failure=True)

# Logs gerados:
# [MultiModelChat] 🚀 Invoking: gemini-2.5-pro
# [MultiModelChat] ❌ Failed: gemini-2.5-pro - quota exceeded
# [MultiModelChat] 🔄 Retrying with fallback: gemini-2.0-flash
# [MultiModelChat] ✅ Success: gemini-2.0-flash (890ms, ~2,541 tokens)
```

---

### 5. **Task Context** — Rastreamento por Etapa

Cada node do LangGraph agora declara seu contexto:

```python
class PlanningNode(BaseNode):
    async def __call__(self, state):
        self.chat.set_task_context("Planning")  # <-- Declara o contexto
        # ... lógica do node ...
```

**Logs gerados**:
```
[MultiModelChat] 📋 Task: Planning
[MultiModelChat] 🚀 Invoking: gemini-2.5-flash
[MultiModelChat] [Planning] ✅ Success: gemini-2.5-flash (1,234ms, ~3,456 tokens)
```

Agora você sabe **exatamente** qual etapa do pipeline está usando qual modelo!

---

### 6. **Detailed Logging** — Visibilidade Total

Todos os eventos são logados:

- ✅ **Inicialização**: Qual modelo foi selecionado
- 🔨 **Cache miss**: Criando nova instância do modelo
- ♻️ **Cache hit**: Reutilizando modelo cacheado
- 🚀 **Invocação**: Iniciando chamada
- ✅ **Sucesso**: Latência + tokens estimados
- ❌ **Falha**: Mensagem de erro
- 🔄 **Retry**: Tentando fallback

**Exemplo de sessão de logs**:
```
[MultiModelChat] ✅ Initialized with model: gemini-2.5-flash

[MultiModelChat] 📋 Task: ReadStructure
[MultiModelChat] 🔨 Creating new model instance: gemini-2.5-flash
[MultiModelChat] ✅ Model cached: gemini-2.5-flash
[MultiModelChat] 🚀 Invoking: gemini-2.5-flash
[MultiModelChat] [ReadStructure] ✅ Success: gemini-2.5-flash (1,123ms, ~5,234 tokens)

[MultiModelChat] 📋 Task: Planning
[MultiModelChat] ♻️  Using cached model: gemini-2.5-flash
[MultiModelChat] 🚀 Invoking: gemini-2.5-flash
[MultiModelChat] [Planning] ✅ Success: gemini-2.5-flash (1,456ms, ~7,890 tokens)
```

---

## 🛠️ Ferramentas de Diagnóstico

### 1. **Diagnostics Script**

```bash
cd c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in
python -m agent.src.llm.diagnostics
```

**Output**:
```
═══════════════════════════════════════════════════════════════════
🏥 LLM Provider Diagnostics
═══════════════════════════════════════════════════════════════════

📋 Step 1: Environment Configuration
──────────────────────────────────────────────────────────────────
   OpenAI               ❌ Not configured
   Anthropic            ❌ Not configured
   Google AI            ✅ Configured
   Ollama               ✅ URL set: http://localhost:11434

📋 Step 2: Provider Health Checks
──────────────────────────────────────────────────────────────────
[Provider Health] ✅ google: Google AI provider configured
[Provider Health] ❌ openai: OpenAI API key not configured
[Provider Health] ❌ anthropic: Anthropic API key not configured
[Provider Health] ❌ ollama: Cannot connect to Ollama at http://localhost:11434

📋 Step 3: Available Models
──────────────────────────────────────────────────────────────────
   ✅ 5 models available:
      • gemini-2.5-flash              (google)
      • gemini-2.5-pro                (google)
      • gemini-2.0-flash              (google)
      • gemini-3-flash                (google)
      • gemini-3-pro                  (google)

📋 Step 4: Default Model Selection
──────────────────────────────────────────────────────────────────
[Model Selection] Available models: ['gemini-2.5-flash', ...]
[Model Selection] ✅ Selected: gemini-2.5-flash
   Selected: gemini-2.5-flash
   ✅ Model gemini-2.5-flash is ready to use

📋 Step 5: Test Specific Models
──────────────────────────────────────────────────────────────────
   ✅ gemini-2.5-flash              Model gemini-2.5-flash is ready to use
   ✅ gemini-2.0-flash              Model gemini-2.0-flash is ready to use
   ✅ gemini-2.5-pro                Model gemini-2.5-pro is ready to use

═══════════════════════════════════════════════════════════════════
📊 Summary
═══════════════════════════════════════════════════════════════════
   Healthy Providers:  1/4
   Available Models:   5
   Default Model:      gemini-2.5-flash

   ℹ️  Some providers are not configured (this is normal)

═══════════════════════════════════════════════════════════════════
```

### 2. **Unit Tests**

```bash
python agent/tests/test_model_tracking.py
```

Testa toda a lógica sem dependências externas (100% mocked).

---

## 📦 Arquivos Modificados

| Arquivo | Mudanças |
|---------|----------|
| [`provider.py`](agent/src/llm/provider.py) | • `ModelUsageStats` e `ModelUsageTracker`<br>• `health_check()` em todos os providers<br>• `check_providers_health()`<br>• `validate_model_before_use()`<br>• `MultiModelChat` com retry + tracking<br>• Logging detalhado |
| [`__init__.py`](agent/src/llm/__init__.py) | • Exporta novas funções e classes |
| [`graph.py`](agent/src/graph/graph.py) | • Health check pré-voo ao criar o grafo<br>• Validação do modelo selecionado |
| [`diagnostics.py`](agent/src/llm/diagnostics.py) | • Script de diagnóstico completo (**NEW**) |
| [`test_model_tracking.py`](agent/tests/test_model_tracking.py) | • Testes unitários da lógica (**NEW**) |

---

## 🎯 Antes vs Depois

### ❌ Antes

```
# Seleção silenciosa
chat = MultiModelChat()  # Qual modelo? Não se sabe

# Uso sem visibilidade
response = await chat.ainvoke(messages)  # Sucesso? Latência? Tokens? Mistério

# Falha sem contexto
# ERROR: quota exceeded (qual modelo? qual etapa?)

# Estatísticas: inexistentes
# Quanto custou a análise? Quantas chamadas fizemos? Não sabemos
```

### ✅ Depois

```
# Seleção explícita
[Model Selection] Available models: ['gemini-2.5-flash', ...]
[Model Selection] ✅ Selected: gemini-2.5-flash

# Uso rastreado
[MultiModelChat] 📋 Task: Planning
[MultiModelChat] 🚀 Invoking: gemini-2.5-flash
[MultiModelChat] [Planning] ✅ Success: gemini-2.5-flash (1,234ms, ~3,456 tokens)

# Falha com contexto
[MultiModelChat] [Planning] ❌ Failed: gemini-2.5-pro - quota exceeded
[MultiModelChat] [Planning] 🔄 Retrying with fallback: gemini-2.0-flash
[MultiModelChat] [Planning] ✅ Success: gemini-2.0-flash (890ms, ~2,541 tokens)

# Estatísticas completas
chat.print_usage_summary()
# 📊 Total: 15 invocations | 93.3% success | 42,156 tokens | 1,234ms avg
```

---

## 🚀 Como Usar

### Cenário 1: Desenvolvimento Local

```bash
# 1. Verificar providers antes de começar
python -m agent.src.llm.diagnostics

# 2. Rodar o agente normalmente
python -m agent.src.main

# 3. No final, ver estatísticas de uso nos logs
# (já está integrado no MultiModelChat)
```

### Cenário 2: Debugging de Análise

Se você quer saber **exatamente** qual modelo está sendo usado em cada etapa:

1. Rode o agente normalmente
2. Busque por logs `[MultiModelChat]` na saída
3. Veja task context, modelo usado, latência, tokens

**Exemplo de busca nos logs**:
```bash
# Ver quais modelos foram usados
grep "\[MultiModelChat\] 🚀" logs.txt

# Ver falhas
grep "\[MultiModelChat\] ❌" logs.txt

# Ver retries
grep "\[MultiModelChat\] 🔄" logs.txt
```

### Cenário 3: Otimização de Custos

```python
# No final da sessão
chat.print_usage_summary()

# Analise:
# - Qual modelo foi mais usado?
# - Qual teve melhor success rate?
# - Quantos tokens consumimos no total?
# - Latência média está aceitável?
```

---

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# Google AI (recomendado - único configurado por padrão)
GOOGLE_API_KEY=your_key_here

# OpenAI (opcional)
OPENAI_API_KEY=your_key_here

# Anthropic (opcional)
ANTHROPIC_API_KEY=your_key_here

# Ollama (opcional - servidor local)
OLLAMA_URL=http://localhost:11434
```

### Customizar Priority List

Edite [`provider.py`](agent/src/llm/provider.py):

```python
def get_default_model() -> str:
    """Get the default model based on what's available."""
    priority = [
        "gemini-2.5-flash",    # ← Mude a ordem aqui
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        # ... adicione novos modelos
    ]
```

---

## 🎉 Benefícios

1. **Visibilidade Total**: Você sempre sabe qual modelo está sendo usado
2. **Debugging Facilitado**: Logs contextualizados por etapa do pipeline
3. **Resiliência**: Auto-retry automático em caso de falha
4. **Otimização**: Estatísticas de uso para otimizar custos
5. **Validação**: Health checks previnem erros de configuração
6. **Manutenibilidade**: Código mais claro e rastreável

---

## 📝 Próximos Passos (Fase 4 - Opcional)

- [ ] Integrar com sistema de billing real para custo exato
- [ ] Dashboard web para visualizar estatísticas em tempo real
- [ ] Alertas quando success rate cair abaixo de threshold
- [ ] Persistir estatísticas em banco de dados
- [ ] Comparação de modelos (A/B testing automático)
- [ ] Rate limiting inteligente por modelo

---

## ✅ Status

**Implementado**: ✅ 100%  
**Testado**: ✅ Logic tests passando  
**Documentado**: ✅ Este arquivo  
**Pronto para produção**: ✅ Sim (precisa instalar dependências)


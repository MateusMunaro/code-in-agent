"""
LLM Provider Diagnostics Tool

Checks the health and availability of all configured LLM providers.

Run:
    cd c:\Users\HP\OneDrive\Desktop\Trabalho\Projetos\code-in
    python -m agent.src.llm.diagnostics
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agent.src.llm.provider import (
    check_providers_health,
    list_available_models,
    get_default_model,
    validate_model_before_use,
    MODEL_PROVIDERS,
)
from agent.src.config import settings


def main():
    print("\n" + "=" * 70)
    print("🏥 LLM Provider Diagnostics")
    print("=" * 70 + "\n")

    # ─── Step 1: Check environment configuration ───────────────
    print("📋 Step 1: Environment Configuration")
    print("-" * 70)
    
    env_status = {
        "OpenAI": "✅ Configured" if settings.openai_api_key else "❌ Not configured",
        "Anthropic": "✅ Configured" if settings.anthropic_api_key else "❌ Not configured",
        "Google AI": "✅ Configured" if settings.google_api_key else "❌ Not configured",
        "Ollama": f"✅ URL set: {settings.ollama_url}",
    }
    
    for provider, status in env_status.items():
        print(f"   {provider:20s} {status}")
    
    print()

    # ─── Step 2: Provider health checks ────────────────────────
    print("📋 Step 2: Provider Health Checks")
    print("-" * 70)
    
    health_results = check_providers_health()
    
    print()

    # ─── Step 3: Available models ──────────────────────────────
    print("📋 Step 3: Available Models")
    print("-" * 70)
    
    available = list_available_models()
    
    if available:
        print(f"   ✅ {len(available)} models available:")
        for model in available:
            provider = MODEL_PROVIDERS.get(model, "unknown")
            print(f"      • {model:30s} ({provider})")
    else:
        print("   ❌ No models available")
        print("   → Check that at least one provider is configured")
    
    print()

    # ─── Step 4: Default model selection ───────────────────────
    print("📋 Step 4: Default Model Selection")
    print("-" * 70)
    
    try:
        default = get_default_model()
        print(f"   Selected: {default}")
        
        # Validate it
        is_valid, msg = validate_model_before_use(default)
        if is_valid:
            print(f"   ✅ {msg}")
        else:
            print(f"   ⚠️ Warning: {msg}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()

    # ─── Step 5: Test recommended models ───────────────────────
    print("📋 Step 5: Test Specific Models")
    print("-" * 70)
    
    test_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
    ]
    
    for model in test_models:
        is_valid, msg = validate_model_before_use(model)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {model:30s} {msg}")
    
    print()

    # ─── Step 6: Summary ───────────────────────────────────────
    print("=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    healthy_providers = sum(1 for is_healthy, _ in health_results.values() if is_healthy)
    total_providers = len(health_results)
    
    print(f"   Healthy Providers:  {healthy_providers}/{total_providers}")
    print(f"   Available Models:   {len(available)}")
    print(f"   Default Model:      {default if 'default' in locals() else 'N/A'}")
    
    if healthy_providers == 0:
        print()
        print("   ⚠️  WARNING: No providers are healthy!")
        print("   → Set API keys in environment variables or .env file")
        print("   → For Google AI: GOOGLE_API_KEY")
        print("   → For OpenAI: OPENAI_API_KEY")
        print("   → For Anthropic: ANTHROPIC_API_KEY")
        print("   → For Ollama: Start the Ollama server")
    elif healthy_providers < total_providers:
        print()
        print("   ℹ️  Some providers are not configured (this is normal)")
    else:
        print()
        print("   ✅ All providers are healthy!")
    
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

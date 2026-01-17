#!/usr/bin/env python3
"""
Test script for LLM providers (Ollama and OpenAI)

This script helps verify that your LLM setup is working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.hint_generator import HintGenerator, OllamaProvider, OpenAIProvider

def test_ollama():
    """Test Ollama provider"""
    print("\n" + "="*80)
    print("Testing Ollama Provider")
    print("="*80)
    
    try:
        provider = OllamaProvider(model="llama3:8b-instruct")
        if provider.is_available():
            print("✓ Ollama is running")
            if provider.check_model_exists():
                print(f"✓ Model 'llama3:8b-instruct' is available")
                
                # Test generation
                print("\nTesting hint generation...")
                prompt = "Generate a brief hint for solving a quadratic equation."
                system = "You are a helpful tutoring assistant."
                result = provider.generate(prompt, system_message=system, max_tokens=50)
                print(f"Generated hint: {result}")
                print("✓ Ollama provider working correctly!")
                return True
            else:
                print("✗ Model 'llama3:8b-instruct' not found")
                print("  Available models: run 'ollama list' to check")
                print("  To download: ollama pull llama3:8b-instruct")
                return False
        else:
            print("✗ Ollama is not running or not accessible")
            print("  Make sure Ollama is running: ollama serve")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_openai():
    """Test OpenAI provider"""
    print("\n" + "="*80)
    print("Testing OpenAI Provider")
    print("="*80)
    
    try:
        provider = OpenAIProvider(model="gpt-3.5-turbo")
        if provider.is_available():
            print("✓ OpenAI API key found")
            
            # Test generation
            print("\nTesting hint generation...")
            prompt = "Generate a brief hint for solving a quadratic equation."
            system = "You are a helpful tutoring assistant."
            result = provider.generate(prompt, system_message=system, max_tokens=50)
            print(f"Generated hint: {result}")
            print("✓ OpenAI provider working correctly!")
            return True
        else:
            print("✗ OpenAI API key not found")
            print("  Set OPENAI_API_KEY environment variable")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_hint_generator_auto():
    """Test HintGenerator with auto provider selection"""
    print("\n" + "="*80)
    print("Testing HintGenerator (Auto Provider Selection)")
    print("="*80)
    
    try:
        generator = HintGenerator(use_llm=True, provider="auto")
        
        if generator.use_llm and generator.llm_provider:
            provider_type = type(generator.llm_provider).__name__
            print(f"✓ Using provider: {provider_type}")
        else:
            print("✓ Using template fallback (no LLM available)")
        
        # Test hint generation
        hint = generator.generate_hint(
            problem_id="test_001",
            problem_description="Solve a quadratic equation: x^2 + 5x + 6 = 0",
            state="confused",
            features={'consecutive_failures': 3},
            attempt_num=5
        )
        
        print(f"\nGenerated hint:")
        print(f"  Detail level: {hint['detail_level']}")
        print(f"  Tone: {hint['tone']}")
        print(f"  Hint text: {hint['hint_text']}")
        print("✓ HintGenerator working correctly!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("LLM Provider Test Suite")
    print("="*80)
    
    results = []
    
    # Test Ollama
    results.append(("Ollama", test_ollama()))
    
    # Test OpenAI
    results.append(("OpenAI", test_openai()))
    
    # Test HintGenerator auto mode
    results.append(("HintGenerator (Auto)", test_hint_generator_auto()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("\nNext steps:")
        print("1. For Ollama: Make sure Ollama is running and model is downloaded")
        print("2. For OpenAI: Set OPENAI_API_KEY environment variable")
        print("3. The system will use template fallback if no LLM is available")

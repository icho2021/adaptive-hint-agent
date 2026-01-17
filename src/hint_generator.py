"""
LLM-Powered Hint Generator

Generate adaptive hints using prompt-based control, varying hint detail
and tone based on inferred learner state.

Supports multiple LLM providers:
- OpenAI API (cloud-based)
- Ollama (local Llama3 and other models)
- Template fallback (no LLM required)
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import requests for Ollama HTTP calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_message: str = "", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Generate text using the LLM
        
        Args:
            prompt: User prompt
            system_message: System message/instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Args:
            model: OpenAI model name
            api_key: API key (if None, reads from OPENAI_API_KEY env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.model = model
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it as environment variable or pass as argument.")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, system_message: str = "", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate using OpenAI API"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') is not None


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider (for Llama3 and other models)"""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Args:
            model: Ollama model name (e.g., "llama3", "llama3:8b", "llama3:8b-instruct")
            base_url: Ollama API base URL
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package not installed. Install with: pip install requests")
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
    
    def generate(self, prompt: str, system_message: str = "", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate using Ollama API"""
        # Combine system message and prompt
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available and running"""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            # Try to check if Ollama is running
            health_url = f"{self.base_url}/api/tags"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model_exists(self) -> bool:
        """Check if the specified model is available in Ollama"""
        if not self.is_available():
            return False
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if model name matches (with or without tag)
                return any(self.model in name or name.startswith(self.model) for name in model_names)
            return False
        except:
            return False


class HintGenerator:
    """Generate adaptive hints based on learner state"""
    
    def __init__(
        self,
        use_llm: bool = True,
        provider: str = "auto",  # "auto", "openai", "ollama", or "template"
        model: str = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Args:
            use_llm: Whether to use actual LLM or fallback templates
            provider: LLM provider to use:
                - "auto": Try Ollama first, then OpenAI, then fallback
                - "openai": Use OpenAI API
                - "ollama": Use local Ollama
                - "template": Use template fallback only
            model: Model name (provider-specific):
                - OpenAI: "gpt-3.5-turbo", "gpt-4", etc.
                - Ollama: "llama3", "llama3:8b", "llama3:8b-instruct", etc.
            ollama_base_url: Base URL for Ollama API
        """
        self.use_llm = use_llm
        self.provider_name = provider
        self.llm_provider: Optional[LLMProvider] = None
        
        # Determine which provider to use
        if not use_llm or provider == "template":
            self.use_llm = False
            self.llm_provider = None
        elif provider == "auto":
            # Try Ollama first, then OpenAI, then fallback
            self.llm_provider = self._try_init_provider("ollama", model, ollama_base_url)
            if not self.llm_provider:
                self.llm_provider = self._try_init_provider("openai", model, ollama_base_url)
            if not self.llm_provider:
                print("Warning: No LLM provider available. Using fallback hint generator.")
                self.use_llm = False
        elif provider == "ollama":
            # Default to llama3:latest if no model specified (most common case)
            default_model = model or "llama3:latest"
            self.llm_provider = self._try_init_provider("ollama", default_model, ollama_base_url)
            if not self.llm_provider:
                print("Warning: Ollama not available. Using fallback hint generator.")
                self.use_llm = False
        elif provider == "openai":
            self.llm_provider = self._try_init_provider("openai", model or "gpt-3.5-turbo", ollama_base_url)
            if not self.llm_provider:
                print("Warning: OpenAI not available. Using fallback hint generator.")
                self.use_llm = False
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'auto', 'openai', 'ollama', or 'template'")
        
        # Hint templates for fallback mode (always initialize)
        self.hint_templates = {
            'confused': {
                'detailed': "Let's break this down step by step. First, consider what the problem is asking for. Then, think about what information you have and what you need to find. Start with the simplest approach and build from there.",
                'moderate': "Take a moment to review what the problem is asking. Break it into smaller parts and tackle them one at a time.",
                'brief': "Try breaking the problem into smaller steps."
            },
            'progressing': {
                'detailed': "You're making good progress! Here's a more advanced approach: consider how the different parts of the problem relate to each other. You might want to explore alternative methods or check edge cases.",
                'moderate': "Great work so far! Consider exploring alternative approaches or checking your reasoning.",
                'brief': "You're on the right track! Keep going."
            },
            'neutral': {
                'detailed': "Here's a helpful hint: review the key concepts related to this problem. Make sure you understand what each part means and how they connect.",
                'moderate': "Review the main concepts and how they apply to this problem.",
                'brief': "Review the key concepts."
            }
        }
    
    def _try_init_provider(self, provider_type: str, model: Optional[str], ollama_base_url: str) -> Optional[LLMProvider]:
        """Try to initialize a provider, return None if it fails"""
        try:
            if provider_type == "ollama":
                if not REQUESTS_AVAILABLE:
                    return None
                model_name = model or "llama3:latest"
                provider = OllamaProvider(model=model_name, base_url=ollama_base_url)
                if provider.is_available():
                    # Check if model exists
                    if provider.check_model_exists():
                        print(f"Using Ollama provider with model: {model_name}")
                        return provider
                    else:
                        # Try to find any available llama3 model
                        try:
                            import requests
                            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                            if response.status_code == 200:
                                models = response.json().get("models", [])
                                model_names = [m.get("name", "") for m in models]
                                # Try to find llama3:latest or any llama3 model
                                fallback_model = None
                                for m in model_names:
                                    if "llama3" in m.lower():
                                        fallback_model = m
                                        break
                                if fallback_model:
                                    print(f"Warning: Model '{model_name}' not found. Using available model: {fallback_model}")
                                    provider = OllamaProvider(model=fallback_model, base_url=ollama_base_url)
                                    if provider.check_model_exists():
                                        print(f"Using Ollama provider with model: {fallback_model}")
                                        return provider
                        except:
                            pass
                        print(f"Warning: Ollama model '{model_name}' not found. Available models: run 'ollama list' to check.")
                        return None
                return None
            elif provider_type == "openai":
                if not OPENAI_AVAILABLE:
                    return None
                model_name = model or "gpt-3.5-turbo"
                provider = OpenAIProvider(model=model_name)
                if provider.is_available():
                    print(f"Using OpenAI provider with model: {model_name}")
                    return provider
                return None
        except Exception as e:
            # Silently fail and try next provider
            return None
        return None
    
    def _determine_hint_detail(
        self,
        state: str,
        consecutive_failures: int,
        attempt_num: int
    ) -> str:  # Returns 'brief', 'moderate', or 'detailed'
        """Determine hint detail level based on context"""
        # More confused or more failures -> more detailed hints
        if state == 'confused' or consecutive_failures >= 4:
            return 'detailed'
        elif consecutive_failures >= 2 or attempt_num > 5:
            return 'moderate'
        else:
            return 'brief'
    
    def _determine_tone(self, state: str) -> str:
        """Determine hint tone based on state"""
        if state == 'confused':
            return "supportive and encouraging"
        elif state == 'progressing':
            return "positive and challenging"
        else:
            return "neutral and informative"
    
    def _generate_llm_hint(
        self,
        problem_context: str,
        state: str,
        detail_level: str,
        tone: str,
        recent_attempts: Optional[str] = None
    ) -> str:
        """Generate hint using LLM"""
        if not self.use_llm or not self.llm_provider:
            return self._generate_fallback_hint(state, detail_level)
        
        prompt = f"""You are an adaptive tutoring system helping a learner solve a problem.

Problem context: {problem_context}

Learner state: {state}
Hint detail level: {detail_level}
Tone: {tone}
{f"Recent attempts: {recent_attempts}" if recent_attempts else ""}

Generate a helpful hint that:
1. Matches the {detail_level} detail level ({'more detailed' if detail_level == 'detailed' else 'concise' if detail_level == 'brief' else 'moderate detail'})
2. Uses a {tone} tone
3. Is appropriate for a learner who is {state}
4. Guides without giving away the answer
5. Is encouraging and supportive

Hint:"""

        try:
            system_message = "You are a helpful, adaptive tutoring assistant."
            hint_text = self.llm_provider.generate(
                prompt=prompt,
                system_message=system_message,
                max_tokens=150,
                temperature=0.7
            )
            return hint_text
        except Exception as e:
            print(f"Error calling LLM: {e}. Using fallback.")
            return self._generate_fallback_hint(state, detail_level)
    
    def _generate_fallback_hint(
        self,
        state: str,
        detail_level: str
    ) -> str:
        """Generate hint using templates (fallback when LLM unavailable)"""
        return self.hint_templates.get(state, self.hint_templates['neutral']).get(
            detail_level,
            self.hint_templates[state]['moderate']
        )
    
    def generate_hint(
        self,
        problem_id: str,
        problem_description: str = "a problem-solving task",
        state: str = "neutral",
        features: Optional[Dict] = None,
        attempt_num: int = 1,
        recent_attempts: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Generate an adaptive hint
        
        Args:
            problem_id: Identifier for the problem
            problem_description: Description of the problem
            state: Inferred learner state ('confused', 'progressing', 'neutral')
            features: Feature dictionary (for context)
            attempt_num: Current attempt number
            recent_attempts: List of recent attempt results
        
        Returns:
            Dict with 'hint_text', 'detail_level', 'tone', 'state'
        """
        if features is None:
            features = {}
        
        consecutive_failures = features.get('consecutive_failures', 0)
        detail_level = self._determine_hint_detail(state, consecutive_failures, attempt_num)
        tone = self._determine_tone(state)
        
        # Format recent attempts for context
        recent_context = None
        if recent_attempts:
            recent_context = "; ".join([
                f"Attempt {i+1}: {'correct' if a.get('is_correct') else 'incorrect'}"
                for i, a in enumerate(recent_attempts[-3:])
            ])
        
        # Generate hint
        hint_text = self._generate_llm_hint(
            problem_description,
            state,
            detail_level,
            tone,
            recent_context
        )
        
        return {
            'hint_text': hint_text,
            'detail_level': detail_level,
            'tone': tone,
            'state': state,
            'problem_id': problem_id,
            'attempt_num': attempt_num
        }


if __name__ == "__main__":
    # Test hint generator
    generator = HintGenerator(use_llm=False)  # Use fallback for testing
    
    test_cases = [
        {'state': 'confused', 'consecutive_failures': 4, 'attempt_num': 5},
        {'state': 'progressing', 'consecutive_failures': 1, 'attempt_num': 3},
        {'state': 'neutral', 'consecutive_failures': 2, 'attempt_num': 4},
    ]
    
    print("Hint Generator Test:\n")
    for i, case in enumerate(test_cases, 1):
        hint = generator.generate_hint(
            problem_id="test_problem",
            problem_description="Solve a math problem involving quadratic equations",
            state=case['state'],
            features={'consecutive_failures': case['consecutive_failures']},
            attempt_num=case['attempt_num']
        )
        print(f"Test {i} - State: {case['state']}, Failures: {case['consecutive_failures']}")
        print(f"Detail: {hint['detail_level']}, Tone: {hint['tone']}")
        print(f"Hint: {hint['hint_text']}\n")

from abc import ABC, abstractmethod
from typing import Generator, Optional
from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.core.logging import logger

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Synchronously generate text response."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        """Stream generated text tokens."""
        pass


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider using the modern google-genai SDK.
    Supports gemini-2.5-flash and other Gemini models.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.LLM_MODEL_NAME

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("GEMINI_API_KEY is not configured in environment settings.")

        self._client = None

    def _get_client(self):
        """Lazy load and initialize Gemini SDK client."""
        if not self.api_key or self.api_key == "your_api_key_here":
            raise LLMServiceError(
                "GEMINI_API_KEY is missing or unconfigured. Please add a valid API key to your .env file."
            )

        if self._client is None:
            if not HAS_GOOGLE_GENAI:
                raise LLMServiceError("google-genai SDK package is not installed.")
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                raise LLMServiceError(f"Gemini client initialization error: {str(e)}")

        return self._client

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate text using Gemini API."""
        client = self._get_client()
        try:
            config = types.GenerateContentConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
                system_instruction=system_instruction if system_instruction else None
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            if not response or not response.text:
                return "I couldn't find sufficient information about this in the uploaded documents."

            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API generation error: {str(e)}")
            raise LLMServiceError(f"Gemini LLM call failed: {str(e)}")

    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text tokens from Gemini API."""
        client = self._get_client()
        try:
            config = types.GenerateContentConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
                system_instruction=system_instruction if system_instruction else None
            )

            response_stream = client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini API streaming error: {str(e)}")
            raise LLMServiceError(f"Gemini LLM streaming failed: {str(e)}")


def get_llm_provider() -> BaseLLMProvider:
    """Factory function for acquiring LLM provider instance."""
    if settings.LLM_PROVIDER.lower() == "gemini":
        return GeminiLLMProvider()
    else:
        raise LLMServiceError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

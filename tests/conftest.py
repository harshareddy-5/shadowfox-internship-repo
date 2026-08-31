import os
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.generation.llm import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for unit tests."""

    def __init__(self, response_text: str = "This is a mock grounded answer based on context."):
        self.response_text = response_text

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        if "STRICTLY" in (system_instruction or ""):
            return "Supervised learning is a type of machine learning where a model learns from labeled training data."
        if "GROUNDEDNESS" in (system_instruction or ""):
            return "GROUNDED\nThe answer is directly supported by the context."
        return self.response_text

    def generate_stream(self, prompt: str, system_instruction: str = None):
        words = self.response_text.split()
        for w in words:
            yield w + " "


@pytest.fixture(scope="function")
def temp_data_dirs(monkeypatch):
    """Create isolated temporary directories for tests."""
    temp_dir = tempfile.mkdtemp()
    uploads_path = Path(temp_dir) / "uploads"
    indexes_path = Path(temp_dir) / "indexes"

    uploads_path.mkdir(parents=True, exist_ok=True)
    indexes_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads_path)
    monkeypatch.setattr(settings, "INDEXES_DIR", indexes_path)

    yield uploads_path, indexes_path

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_llm(monkeypatch):
    """Inject mock LLM provider into generation dependency."""
    mock = MockLLMProvider()
    monkeypatch.setattr("app.generation.llm.get_llm_provider", lambda: mock)
    monkeypatch.setattr("app.graph.nodes.get_llm_provider", lambda: mock)
    return mock


@pytest.fixture
def test_client(temp_data_dirs, mock_llm):
    """FastAPI TestClient with isolated data directories and mock LLM."""
    from app.main import app
    with TestClient(app) as client:
        yield client

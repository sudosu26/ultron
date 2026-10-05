"""Base provider interface and chat message model."""

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ChatRequest:
    """A request to send to an LLM provider."""
    messages: list[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True


class LLMProvider(Protocol):
    """Interface that every LLM provider must implement."""

    name: str

    async def stream_chat(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        """Yield response text chunks as they arrive from the provider."""
        ...

    async def validate_key(self, api_key: str) -> bool:
        """Return True if the API key is accepted by the provider."""
        ...
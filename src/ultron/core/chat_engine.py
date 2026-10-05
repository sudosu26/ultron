"""Chat engine — orchestrates messages between the UI and providers."""

from ultron.providers.base import ChatMessage, ChatRequest
from ultron.providers.gemini import GeminiProvider
from ultron.utils.logging import setup_logging

logger = setup_logging()

_SYSTEM_PROMPT = (
    "You are ULTRON, a helpful personal AI assistant. "
    "Be concise, accurate, and friendly. "
    "If you are unsure about something, say so. "
    "Never invent facts, prices, or API details."
)


class ChatEngine:
    """Manages the conversation and delegates to the active provider."""

    def __init__(self) -> None:
        self.provider = GeminiProvider()
        # The provider resolves the model dynamically. This is only a label
        # used for display; the actual model is chosen by the provider.
        self.model = "auto"
        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT)
        ]

    def add_user_message(self, text: str) -> None:
        self.messages.append(ChatMessage(role="user", content=text))

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=text))

    def build_request(self) -> ChatRequest:
        return ChatRequest(
            messages=list(self.messages),
            model=self.model,  # Provider ignores this and resolves its own.
            stream=True,
        )

    async def stream_response(self, api_key: str):
        """Yield response chunks from the provider."""
        request = self.build_request()
        full_response = ""
        async for chunk in self.provider.stream_chat(request, api_key):
            full_response += chunk
            yield chunk
        self.add_assistant_message(full_response)
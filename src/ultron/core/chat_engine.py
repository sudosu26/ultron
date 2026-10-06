"""Chat engine — orchestrates messages between the UI and providers."""

from ultron.providers.base import ChatMessage, ChatRequest
from ultron.providers.gemini import GeminiProvider
from ultron.utils.logging import setup_logging

logger = setup_logging()

_SYSTEM_PROMPT = (
    "You are ULTRON, a helpful personal AI assistant. "
    "Answer directly in 1-3 short sentences by default. "
    "Give more detail only when the user asks for it. "
    "Be accurate and friendly. "
    "If you are unsure about something, say so. "
    "Never invent facts, prices, or API details."
)


class ChatEngine:
    """Manages the in-memory conversation and delegates to the active provider.

    The database is the source of truth for persistence. The engine holds
    only the messages needed to send on the next provider call.
    """

    def __init__(self) -> None:
        self.provider = GeminiProvider()
        self.model = "auto"
        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT)
        ]

    def reset(self) -> None:
        self.messages = [ChatMessage(role="system", content=_SYSTEM_PROMPT)]

    def load_history(self, stored_messages: list[dict]) -> None:
        self.messages = [ChatMessage(role="system", content=_SYSTEM_PROMPT)]
        for m in stored_messages:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant"):
                self.messages.append(ChatMessage(role=role, content=content))

    def add_user_message(self, text: str) -> None:
        self.messages.append(ChatMessage(role="user", content=text))

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=text))

    def build_request(self) -> ChatRequest:
        return ChatRequest(
            messages=list(self.messages),
            model=self.model,
            stream=True,
        )

    async def stream_response(self, api_keys: list[str]):
        """Yield response chunks. `api_keys` is a list of provider keys."""
        request = self.build_request()
        full_response = ""
        async for chunk in self.provider.stream_chat(request, api_keys):
            full_response += chunk
            yield chunk
        self.add_assistant_message(full_response)
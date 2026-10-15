from typing import Any, Protocol

import ollama  # imported here so the rest of the package works without it

QWEN3_4B = "qwen3:4b"


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        """Return the model's reply, constrained to the JSON `schema`."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str = QWEN3_4B,
        host: str | None = None,
        num_ctx: int = 4096,
        temperature: float = 0.0,
    ) -> None:

        self.client = ollama.Client(host=host)
        self.model = model
        self.options = {"num_ctx": num_ctx, "temperature": temperature}

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema,
            options=self.options,
            think=False,
        )
        return response.message.content

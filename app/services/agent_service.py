from __future__ import annotations

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.tools import news_search, rag_search, risk_analyzer
from app.utils.config import get_settings
from app.utils.prompt_templates import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT


def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set your Gemini API key.")

    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
    )


def _build_agent():
    return create_agent(
        model=_get_llm(),
        tools=[rag_search, news_search, risk_analyzer],
        system_prompt=SYSTEM_PROMPT,
    )


def run_analysis(query: str) -> str:
    agent = _build_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{query}\n\n{OUTPUT_INSTRUCTIONS}",
                }
            ]
        }
    )

    messages = result.get("messages", [])
    if not messages:
        return "No response generated."

    last_message = messages[-1]
    content = getattr(last_message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    return str(content)

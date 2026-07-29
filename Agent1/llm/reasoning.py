"""
LLM reasoning layer — routes to Gemini, OpenAI, or Ollama.
Never sends images. Only sends structured detection JSON as text.

Provider selection (set LLM_PROVIDER in .env):
  gemini  → Google Gemini API  (GEMINI_API_KEY)
  openai  → OpenAI API         (OPENAI_API_KEY)
  ollama  → Ollama local       (LLM_BASE_URL + LLM_MODEL)
"""
import logging
from typing import Any

from backend.config import settings
from llm.prompts import SYSTEM_PROMPT, build_detection_prompt, build_question_prompt

logger = logging.getLogger(__name__)

_PLACEHOLDERS = {"", "sk-placeholder", "sk-your-key-here"}


class LLMReasoner:

    def _call_gemini(self, user_message: str) -> str:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
        )
        return response.text.strip()

    def _call_openai_compat(self, user_message: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL.strip() or None,
        )
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def _chat(self, user_message: str) -> str:
        provider = settings.LLM_PROVIDER.lower().strip()
        try:
            if provider == "gemini":
                if settings.GEMINI_API_KEY in _PLACEHOLDERS:
                    return _no_key_msg("GEMINI_API_KEY")
                return self._call_gemini(user_message)

            elif provider in ("openai", "ollama"):
                if settings.LLM_API_KEY in _PLACEHOLDERS:
                    return _no_key_msg("OPENAI_API_KEY")
                return self._call_openai_compat(user_message)

            else:
                return f"Unknown LLM_PROVIDER '{provider}'. Use: gemini | openai | ollama"

        except Exception as e:
            logger.error("LLM call failed [%s]: %s", provider, e)
            return f"LLM error: {e}"

    def explain(self, payload: dict[str, Any]) -> str:
        """Generate a natural language explanation of detection results."""
        detections = payload.get("detections", [])
        avg_conf = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0.0
        )
        return self._chat(build_detection_prompt(
            image_name=payload["image_name"],
            total_cells=payload["total_cells"],
            rbc=payload["rbc"],
            wbc=payload["wbc"],
            platelet=payload["platelet"],
            inference_time=payload["inference_time"],
            avg_confidence=avg_conf,
        ))

    def answer(self, question: str, payload: dict[str, Any]) -> str:
        """Answer a user question about a specific detection result."""
        detections = payload.get("detections", [])
        avg_conf = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0.0
        )
        return self._chat(build_question_prompt(
            question=question,
            total_cells=payload["total_cells"],
            rbc=payload["rbc"],
            wbc=payload["wbc"],
            platelet=payload["platelet"],
            avg_confidence=avg_conf,
        ))


def _no_key_msg(key_name: str) -> str:
    return (
        f"LLM explanation unavailable — set {key_name} in your .env file. "
        f"Current provider: {settings.LLM_PROVIDER}"
    )


llm_reasoner = LLMReasoner()

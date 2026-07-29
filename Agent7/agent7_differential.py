"""
Agent 7 — Differential Aggregator
Takes structured JSON outputs from Agents 1–6 and synthesises one coherent
clinical differential using the configured LLM (Gemini / OpenAI / Ollama).
No model weights required — prompt-based reasoning only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.config import settings

_PLACEHOLDERS = {"", "sk-placeholder", "sk-your-key-here"}

_SYSTEM = """You are a senior haematology AI assistant integrated into a blood smear analysis system.
You receive structured JSON outputs from multiple specialised screening agents and must synthesise them into one coherent clinical differential.

Rules:
- Cross-reference findings across agents (e.g. high neutrophils + malaria negative → bacterial infection likely).
- Flag contradictions or low-confidence findings explicitly.
- Do NOT diagnose — produce a differential with ranked possibilities and supporting evidence.
- Be concise, structured, and clinically precise.
- Always state that findings require review by a qualified clinician."""


def _build_prompt(agent_outputs: dict[str, Any]) -> str:
    return f"""Below are the structured outputs from all HemaScope screening agents for one blood smear image.
Synthesise these into a coherent clinical differential.

=== AGENT OUTPUTS ===
{json.dumps(agent_outputs, indent=2, default=str)}

Provide your response in this exact structure:
1. KEY FINDINGS — bullet list of the most significant findings across all agents
2. DIFFERENTIAL DIAGNOSIS — ranked list (most likely first) with supporting evidence from the agent data
3. CROSS-AGENT CORRELATIONS — notable agreements or contradictions between agents
4. RECOMMENDED FOLLOW-UP — specific tests or clinical actions
5. CONFIDENCE NOTE — flag any agents with low cell counts or uncertain results"""


@dataclass
class Agent7Result:
    synthesis: str
    agent_outputs: dict
    model_used: str

    def to_dict(self) -> dict:
        return {
            "synthesis":     self.synthesis,
            "agent_outputs": self.agent_outputs,
            "model_used":    self.model_used,
        }


class DifferentialAggregator:

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=f"{_SYSTEM}\n\n{prompt}",
        )
        return response.text.strip()

    def _call_openai_compat(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL.strip() or None,
        )
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    def run(self, agent_outputs: dict[str, Any]) -> Agent7Result:
        """
        agent_outputs: dict with any subset of keys:
          agent1, agent2_malaria, agent3_morphology,
          agent4_wbc, agent5_leukemia, agent6_anemia
        """
        prompt   = _build_prompt(agent_outputs)
        provider = settings.LLM_PROVIDER.lower().strip()

        try:
            if provider == "gemini":
                if settings.GEMINI_API_KEY in _PLACEHOLDERS:
                    synthesis = "LLM unavailable — set GEMINI_API_KEY in .env"
                else:
                    synthesis = self._call_gemini(prompt)
            elif provider in ("openai", "ollama"):
                if settings.LLM_API_KEY in _PLACEHOLDERS:
                    synthesis = "LLM unavailable — set OPENAI_API_KEY in .env"
                else:
                    synthesis = self._call_openai_compat(prompt)
            else:
                synthesis = f"Unknown LLM_PROVIDER '{provider}'. Use: gemini | openai | ollama"
        except Exception as e:
            synthesis = f"LLM error: {e}"

        return Agent7Result(
            synthesis=synthesis,
            agent_outputs=agent_outputs,
            model_used=settings.LLM_MODEL,
        )

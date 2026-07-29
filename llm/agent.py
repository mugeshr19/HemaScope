"""
Blood Cell Detection Agent — True Agentic Layer

The LLM is given tools and decides which to call based on the user's question.
Tools available to the agent:
  - detect_cells        : run YOLOv11 on an image
  - get_cell_counts     : return RBC / WBC / Platelet counts
  - get_confidence_stats: return avg / min / max confidence
  - compare_to_normal   : compare counts to normal clinical ranges
  - get_detection_detail: get details of a specific cell type
  - summarise_findings  : produce a full natural language summary

Provider routing (set LLM_PROVIDER in .env):
  gemini  → Google Gemini API  (GEMINI_API_KEY)
  openai  → OpenAI / Groq / any OpenAI-compatible API
  ollama  → Ollama local       (LLM_BASE_URL + LLM_MODEL)
"""
import json
import logging
from typing import Any

from backend.config import settings
from backend.services.inference_service import inference_service
from llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Normal clinical reference ranges ─────────────────────────────────────────
NORMAL_RANGES = {
    "RBC":      {"min": 4_200_000, "max": 6_100_000, "unit": "cells/µL",  "note": "4.2–6.1 million/µL"},
    "WBC":      {"min": 4_000,     "max": 11_000,     "unit": "cells/µL",  "note": "4,000–11,000/µL"},
    "Platelet": {"min": 150_000,   "max": 400_000,    "unit": "cells/µL",  "note": "150,000–400,000/µL"},
}

# ── Tool definitions sent to the LLM ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_cells",
            "description": "Run YOLOv11 blood cell detection on an image file. Returns full detection results including bounding boxes, counts, and confidence scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the blood smear image file."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_counts",
            "description": "Get the RBC, WBC, and Platelet counts from the last detection result.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_confidence_stats",
            "description": "Get average, minimum, and maximum confidence scores from the last detection.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_normal",
            "description": "Compare detected cell counts to normal clinical reference ranges. Returns whether each count is low, normal, or high.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_type": {
                        "type": "string",
                        "enum": ["RBC", "WBC", "Platelet", "all"],
                        "description": "Which cell type to compare. Use 'all' for all types."
                    }
                },
                "required": ["cell_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection_detail",
            "description": "Get detailed information about a specific cell type from the last detection, including individual bounding boxes and confidence scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_type": {
                        "type": "string",
                        "enum": ["RBC", "WBC", "Platelets"],
                        "description": "The cell type to get details for."
                    }
                },
                "required": ["cell_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_findings",
            "description": "Generate a complete natural language summary of all detection findings.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


_PLACEHOLDERS = {"", "sk-placeholder", "sk-your-key-here"}
_AGENT_SYSTEM = (
    SYSTEM_PROMPT + "\n\n"
    "You are an AI agent with tools to detect and analyse blood cells. "
    "Use the tools to answer the user's question step by step. "
    "Always call detect_cells first if an image path is provided and no detection has been run yet. "
    "Do not diagnose diseases. Only explain what the detector found."
)


class BloodCellAgent:
    """
    Agentic loop: LLM decides which tools to call, calls them,
    feeds results back, and continues until it has a final answer.
    Supports Gemini (function-calling via OpenAI-compat endpoint),
    OpenAI / Groq, and Ollama.
    """

    def __init__(self) -> None:
        self._last_payload: dict[str, Any] | None = None

    def _get_client(self):
        """Build OpenAI-compatible client lazily so import never crashes."""
        from openai import OpenAI
        provider = settings.LLM_PROVIDER.lower().strip()
        if provider == "gemini":
            # Gemini exposes an OpenAI-compatible endpoint
            return OpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        return OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL.strip() or None,
        )

    def _is_available(self) -> bool:
        provider = settings.LLM_PROVIDER.lower().strip()
        if provider == "gemini":
            return settings.GEMINI_API_KEY not in _PLACEHOLDERS
        return settings.LLM_API_KEY not in _PLACEHOLDERS

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, user_message: str, image_path: str | None = None) -> str:
        """
        Run the agent with a user message and optional image path.
        The agent will decide which tools to call to answer the question.
        """
        if not self._is_available():
            return (
                f"Agent unavailable — set the API key for provider "
                f"'{settings.LLM_PROVIDER}' in your .env file."
            )

        client = self._get_client()
        messages = [
            {"role": "system", "content": _AGENT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{user_message}"
                    + (f"\n\nImage path: {image_path}" if image_path else "")
                ),
            },
        ]

        # Agentic loop — keep calling tools until LLM gives a final answer
        for _ in range(10):  # max 10 tool calls per turn
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )

            msg = response.choices[0].message

            # No more tool calls — LLM has a final answer
            if not msg.tool_calls:
                return msg.content.strip()

            # Append assistant message with tool calls
            messages.append(msg)

            # Execute each tool call and feed results back
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(tool_name, args)
                logger.info(
                    "Tool called: %s | Args: %s | Result keys: %s",
                    tool_name, args,
                    list(result.keys()) if isinstance(result, dict) else "str",
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

        return "Agent reached maximum tool call limit. Please try a simpler question."

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> dict | str:
        tools_map = {
            "detect_cells":         self._tool_detect_cells,
            "get_cell_counts":      self._tool_get_cell_counts,
            "get_confidence_stats": self._tool_get_confidence_stats,
            "compare_to_normal":    self._tool_compare_to_normal,
            "get_detection_detail": self._tool_get_detection_detail,
            "summarise_findings":   self._tool_summarise_findings,
        }
        fn = tools_map.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return {"error": str(e)}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_detect_cells(self, image_path: str) -> dict:
        if not inference_service.is_loaded():
            inference_service.load_model()
        payload = inference_service.predict(image_path)
        self._last_payload = payload
        return {
            "status":         "success",
            "prediction_id":  payload["prediction_id"],
            "image_name":     payload["image_name"],
            "total_cells":    payload["total_cells"],
            "rbc":            payload["rbc"],
            "wbc":            payload["wbc"],
            "platelet":       payload["platelet"],
            "inference_time": payload["inference_time"],
            "annotated_path": payload["annotated_path"],
        }

    def _tool_get_cell_counts(self) -> dict:
        if not self._last_payload:
            return {"error": "No detection has been run yet. Call detect_cells first."}
        return {
            "total_cells": self._last_payload["total_cells"],
            "rbc":         self._last_payload["rbc"],
            "wbc":         self._last_payload["wbc"],
            "platelet":    self._last_payload["platelet"],
        }

    def _tool_get_confidence_stats(self) -> dict:
        if not self._last_payload:
            return {"error": "No detection has been run yet. Call detect_cells first."}
        confs = [d["confidence"] for d in self._last_payload["detections"]]
        if not confs:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        return {
            "avg":   round(sum(confs) / len(confs), 4),
            "min":   round(min(confs), 4),
            "max":   round(max(confs), 4),
            "count": len(confs),
        }

    def _tool_compare_to_normal(self, cell_type: str) -> dict:
        if not self._last_payload:
            return {"error": "No detection has been run yet. Call detect_cells first."}

        counts = {
            "RBC":      self._last_payload["rbc"],
            "WBC":      self._last_payload["wbc"],
            "Platelet": self._last_payload["platelet"],
        }

        def _assess(ct: str) -> dict:
            count = counts.get(ct, 0)
            ref   = NORMAL_RANGES[ct]
            if count < ref["min"]:
                status = "LOW"
            elif count > ref["max"]:
                status = "HIGH"
            else:
                status = "NORMAL"
            return {
                "detected":       count,
                "normal_range":   ref["note"],
                "status":         status,
                "note": (
                    "This is a raw cell count from a single image patch, "
                    "not a full blood count. Clinical interpretation required."
                )
            }

        if cell_type == "all":
            return {ct: _assess(ct) for ct in ["RBC", "WBC", "Platelet"]}
        return {cell_type: _assess(cell_type)}

    def _tool_get_detection_detail(self, cell_type: str) -> dict:
        if not self._last_payload:
            return {"error": "No detection has been run yet. Call detect_cells first."}
        cells = [
            d for d in self._last_payload["detections"]
            if d["class"] == cell_type
        ]
        return {
            "cell_type": cell_type,
            "count":     len(cells),
            "cells": [
                {
                    "cell_id":    c["cell_id"],
                    "confidence": c["confidence"],
                    "bbox":       c["bbox"],
                }
                for c in cells[:20]  # cap at 20 to avoid token overflow
            ]
        }

    def _tool_summarise_findings(self) -> dict:
        if not self._last_payload:
            return {"error": "No detection has been run yet. Call detect_cells first."}
        p = self._last_payload
        confs = [d["confidence"] for d in p["detections"]]
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0
        return {
            "image":          p["image_name"],
            "total_cells":    p["total_cells"],
            "rbc":            p["rbc"],
            "wbc":            p["wbc"],
            "platelet":       p["platelet"],
            "avg_confidence": avg_conf,
            "inference_time": p["inference_time"],
            "annotated_path": p["annotated_path"],
        }


# Singleton — instantiated lazily so import never crashes without an API key
blood_cell_agent = BloodCellAgent()

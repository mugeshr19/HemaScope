"""
Gradio UI — Blood Cell Detection Agent
Tabs: Detection | Agent Chat | Examples

Run:
    python app.py
    Open http://localhost:7860
"""
import json
import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from backend.config import settings
from backend.services.inference_service import inference_service
from llm.reasoning import llm_reasoner
from llm.agent import blood_cell_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

inference_service.load_model()


# ── Helper functions ──────────────────────────────────────────────────────────

def run_detection(image_path, conf_threshold, iou_threshold):
    if image_path is None:
        return None, None, "*No image provided.*", "No image provided.", "{}"

    settings.CONFIDENCE_THRESHOLD = conf_threshold
    settings.IOU_THRESHOLD = iou_threshold

    try:
        payload = inference_service.predict(image_path)
    except Exception as e:
        return None, None, f"*Error: {e}*", str(e), "{}"

    annotated_bgr = cv2.imread(payload["annotated_path"])
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    rows = [
        {
            "Cell ID":    d["cell_id"],
            "Type":       d["class"],
            "Confidence": f"{d['confidence']:.2%}",
            "X1": d["bbox"][0], "Y1": d["bbox"][1],
            "X2": d["bbox"][2], "Y2": d["bbox"][3],
        }
        for d in payload["detections"]
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Cell ID", "Type", "Confidence", "X1", "Y1", "X2", "Y2"]
    )

    counts_md = (
        f"| Cell | Count |\n|---|---|\n"
        f"| **Total** | {payload['total_cells']} |\n"
        f"| **RBC** | {payload['rbc']} |\n"
        f"| **WBC** | {payload['wbc']} |\n"
        f"| **Platelets** | {payload['platelet']} |\n"
        f"| **Time** | {payload['inference_time']}s |"
    )

    explanation = llm_reasoner.explain(payload)
    export = {k: v for k, v in payload.items() if k != "image_path"}
    return annotated_rgb, df, counts_md, explanation, json.dumps(export, indent=2, default=str)


def ask_llm(question, json_state):
    if not question.strip():
        return "Please enter a question."
    try:
        payload = json.loads(json_state)
    except Exception:
        return "No detection result available. Run detection first."
    if not payload:
        return "No detection result available. Run detection first."
    return llm_reasoner.answer(question, payload)


def save_json(json_state):
    if not json_state or json_state == "{}":
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    tmp.write(json_state)
    tmp.close()
    return tmp.name


def run_agent(image_path, question, history):
    if not question.strip():
        return history, ""
    try:
        answer = blood_cell_agent.run(question, image_path or None)
    except Exception as e:
        answer = f"Agent error: {e}"
    history = history + [[question, answer]]
    return history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Blood Cell Detection Agent") as demo:

    json_state    = gr.State("{}")
    agent_history = gr.State([])

    gr.Markdown(
        """
        # Blood Cell Detection Agent
        **YOLOv11 + LLM Agent** · Detects RBC, WBC, and Platelets in blood smear images.
        > For research use only. Not a medical diagnostic tool.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Detection ──────────────────────────────────────────────────
        with gr.Tab("Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Upload Blood Smear Image", height=300)
                    with gr.Row():
                        conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence Threshold")
                        iou_slider  = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU Threshold")
                    detect_btn = gr.Button("Detect Cells", variant="primary", size="lg")
                    counts_md  = gr.Markdown("*Upload an image and click Detect.*")

                with gr.Column(scale=1):
                    annotated_output  = gr.Image(label="Annotated Image", height=300)
                    download_json_btn = gr.DownloadButton(label="⬇ Download JSON", variant="secondary", size="sm")

            gr.Markdown("### Detections")
            detection_table = gr.DataFrame(
                headers=["Cell ID", "Type", "Confidence", "X1", "Y1", "X2", "Y2"],
                label="Detection Results", wrap=True,
            )

            gr.Markdown("### LLM Explanation")
            explanation_box = gr.Textbox(
                label="Natural Language Summary", lines=8, interactive=False,
                placeholder="LLM explanation will appear here after detection...",
            )

            gr.Markdown("### Ask a Follow-up Question")
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder='e.g. "How many RBCs?" or "What is the average confidence?"',
                    label="Question", scale=4,
                )
                ask_btn = gr.Button("Ask", variant="secondary", scale=1)
            answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)

            detect_btn.click(
                fn=run_detection,
                inputs=[image_input, conf_slider, iou_slider],
                outputs=[annotated_output, detection_table, counts_md, explanation_box, json_state],
            )
            ask_btn.click(fn=ask_llm, inputs=[question_input, json_state], outputs=[answer_box])
            download_json_btn.click(fn=save_json, inputs=[json_state], outputs=[download_json_btn])

        # ── Tab 2: Agent Chat ─────────────────────────────────────────────────
        with gr.Tab("Agent Chat"):
            gr.Markdown(
                """
                ### Agentic Blood Cell Analysis
                The LLM **decides which tools to call** based on your question.
                Upload an image and ask anything — the agent will detect, count,
                compare to normal ranges, and explain automatically.

                **Try asking:**
                - *"Analyse this image and tell me what you find"*
                - *"Is the WBC count normal?"*
                - *"Compare all cell counts to normal clinical ranges"*
                - *"How confident is the model about the platelet detections?"*
                """
            )
            agent_image = gr.Image(
                type="filepath",
                label="Upload Image (agent will auto-detect when you ask)",
                height=220,
            )
            agent_chatbot = gr.Chatbot(label="Agent", height=420)
            with gr.Row():
                agent_input = gr.Textbox(
                    placeholder="Ask the agent anything about the blood smear...",
                    label="", scale=5, container=False,
                )
                agent_send = gr.Button("Send", variant="primary", scale=1)
            agent_clear = gr.Button("Clear Chat", variant="secondary", size="sm")

            agent_send.click(
                fn=run_agent,
                inputs=[agent_image, agent_input, agent_history],
                outputs=[agent_chatbot, agent_input],
            ).then(lambda h: h, inputs=[agent_chatbot], outputs=[agent_history])

            agent_input.submit(
                fn=run_agent,
                inputs=[agent_image, agent_input, agent_history],
                outputs=[agent_chatbot, agent_input],
            ).then(lambda h: h, inputs=[agent_chatbot], outputs=[agent_history])

            agent_clear.click(lambda: ([], []), outputs=[agent_chatbot, agent_history])

        # ── Tab 3: Examples ───────────────────────────────────────────────────
        example_img = Path("datasets/raw/TXL-PBC_Dataset/example.png")
        if example_img.exists():
            with gr.Tab("Examples"):
                gr.Examples(
                    examples=[[str(example_img), 0.25, 0.45]],
                    inputs=[image_input, conf_slider, iou_slider],
                    label="Sample Blood Smear",
                )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, theme=THEME)

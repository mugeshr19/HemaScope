# HemaScope — Multi-Agent AI Healthcare System

Blood smear analysis pipeline split into focused agents.

```
HemaScope/
├── Agent1/   Blood Cell Detection  — YOLOv11 detector (runs first on raw image)
│             Detects & localises RBC / WBC / Platelet, crops each cell.
│             Every downstream agent consumes its crops + detection JSON.
│
└── Agent2/   LLM Reasoning         — consumes Agent 1 output (in development)
              Natural language explanation, Q&A, clinical range comparison.
```

See each agent's `README.md` for setup and usage instructions.

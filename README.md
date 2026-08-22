# Sensor Context Encoder Challenge

Evaluates the feasibility of converting windowed inertial time-series signals ($128 \times 9$) into continuous context embeddings consumed directly by a frozen language model (`HuggingFaceTB/SmolLM2-360M-Instruct`) without converting sensor readings into text.

## Setup Instructions

1. **Clone & Environment Setup:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/sensor-context-encoder.git
   cd sensor-context-encoder
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

# Sensor Context Encoder

Evaluates the feasibility of projecting windowed inertial time-series telemetry ($128 \times 9$) directly into the token embedding space of a frozen language model (`HuggingFaceTB/SmolLM2-360M-Instruct`) without intermediate string serialization.

## Setup & Reproduction

1. **Environment Setup:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/inertial-sensor-context-encoder.git
   cd inertial-sensor-context-encoder
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
# Sensor Context Encoder

Evaluates the feasibility of projecting windowed inertial time-series telemetry (128×9) directly into the token embedding space of a frozen language model (`HuggingFaceTB/SmolLM2-360M-Instruct`) without intermediate string serialization.

## Setup & Reproduction

### 1. Environment Setup
```bash
git clone [https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder.git](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder.git)
cd inertial-sensor-context-encoder

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
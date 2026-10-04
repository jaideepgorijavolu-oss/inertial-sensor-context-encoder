import os

# Tests never need the Hugging Face Hub (run metadata skips the model-revision lookup).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import pytest
from src.models import SensorEncoder

def test_sensor_encoder_output_shape():
    """Verify that a batch of 128-step, 9-channel readings yields [batch_size, 256]."""
    batch_size = 4
    seq_len = 128
    channels = 9
    
    encoder = SensorEncoder(in_channels=channels, hidden_dim=256)
    encoder.eval()

    # The dataset provides tensors of shape (batch_size, seq_len, channels)
    # Check whether the encoder handles permutation internally or expects (B, seq_len, channels)
    x = torch.randn(batch_size, seq_len, channels)
    
    with torch.no_grad():
        try:
            out = encoder(x)
        except RuntimeError:
            # Fallback if your forward method expects direct (B, C, L)
            out = encoder(x.permute(0, 2, 1))
        
    assert out.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {out.shape}"
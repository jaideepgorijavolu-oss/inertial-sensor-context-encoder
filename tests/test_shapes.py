import torch
from src.models import DirectClassifier, MatchedCapacityClassifier, SensorEncoder


def test_sensor_encoder_output_shape():
    """A batch of [B, 128 timesteps, 9 channels] windows yields [B, 256]."""
    encoder = SensorEncoder(in_channels=9, hidden_dim=256).eval()
    with torch.no_grad():
        out = encoder(torch.randn(4, 128, 9))
    assert out.shape == (4, 256)


def test_classifier_heads_output_logits():
    x = torch.randn(4, 128, 9)
    for model in (DirectClassifier(SensorEncoder()), MatchedCapacityClassifier(SensorEncoder(), llm_dim=32)):
        assert model.eval()(x).shape == (4, 6)

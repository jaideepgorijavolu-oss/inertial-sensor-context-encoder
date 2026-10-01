"""
ContextEmbeddingModel tests on a tiny randomly initialised Llama (same architecture
family as SmolLM2), so they run in seconds on CPU with no model download.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from src.models import ContextEmbeddingModel, SensorEncoder, trainable_state_dict


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return ContextEmbeddingModel(
        SensorEncoder(hidden_dim=16), LlamaModel(cfg),
        prefix_ids=torch.tensor([1, 5, 6, 7]), suffix_ids=torch.tensor([8, 9]),
    )


def test_output_shape(model):
    assert model.eval()(torch.randn(3, 128, 9)).shape == (3, 6)


def test_gradients_reach_encoder_and_projector(model):
    """Regression test: a no_grad() around the LLM previously left these untrained."""
    model.train()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 128, 9)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    for name in ("encoder", "projector", "classification_head"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None for g in grads), f"{name} received no gradient"
        assert any(g.abs().sum() > 0 for g in grads), f"{name} gradient is all zeros"


def test_llm_is_frozen_and_stays_in_eval_mode(model):
    model.train()
    assert not any(p.requires_grad for p in model.llm.parameters())
    assert not model.llm.training
    assert model.encoder.training


def test_prediction_depends_on_sensor_input(model):
    model.eval()
    with torch.no_grad():
        a = model(torch.randn(2, 128, 9))
        b = model(torch.randn(2, 128, 9))
    assert not torch.allclose(a, b)


def test_shuffle_permutes_sensor_tokens_across_batch(model):
    model.eval()
    x = torch.randn(8, 128, 9)
    with torch.no_grad():
        clean = model(x)
        torch.manual_seed(1)
        shuffled = model(x, shuffle_embeddings=True)
    # Same set of logit rows, different order (a permutation of the batch).
    assert not torch.allclose(clean, shuffled)
    assert torch.allclose(clean.sort(dim=0).values, shuffled.sort(dim=0).values, atol=1e-5)


def test_trainable_state_dict_is_a_snapshot_without_llm(model):
    """Regression test: tensor.cpu() on CPU aliases live weights, so 'best' == last epoch."""
    snap = trainable_state_dict(model)
    assert snap and not any(k.startswith("llm.") for k in snap)
    with torch.no_grad():
        model.classification_head.weight.add_(1.0)
    assert not torch.equal(snap["classification_head.weight"], model.classification_head.weight)

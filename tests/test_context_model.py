"""
ContextEmbeddingModel tests on a tiny randomly initialised Llama (same architecture
family as SmolLM2), so they run in seconds on CPU with no model download.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from src.models import ContextEmbeddingModel, SensorEncoder, add_lora, trainable_state_dict


def tiny_model(num_sensor_tokens: int = 1, lora_r: int = 0) -> ContextEmbeddingModel:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    llm = LlamaModel(cfg)
    if lora_r:
        llm = add_lora(llm, lora_r)
    return ContextEmbeddingModel(
        SensorEncoder(hidden_dim=16), llm,
        prefix_ids=torch.tensor([1, 5, 6, 7]), suffix_ids=torch.tensor([8, 9]),
        num_sensor_tokens=num_sensor_tokens,
    )


@pytest.fixture(params=[(1, 0), (4, 0), (4, 2)], ids=["1tok", "4tok", "4tok-lora"])
def model(request):
    return tiny_model(*request.param)


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


def test_llm_is_frozen_and_stays_in_eval_mode():
    model = tiny_model().train()
    assert not any(p.requires_grad for p in model.llm.parameters())
    assert not model.llm.training
    assert model.encoder.training


def test_lora_trains_only_adapters():
    model = tiny_model(num_sensor_tokens=4, lora_r=2).train()
    trainable = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    loss = model(torch.randn(4, 128, 9)).sum()
    loss.backward()
    lora_b = [p.grad for n, p in model.llm.named_parameters() if "lora_B" in n]
    assert lora_b and all(g is not None and g.abs().sum() > 0 for g in lora_b)
    # Adapters are part of the checkpoint; frozen base weights are not.
    snap = trainable_state_dict(model)
    assert any("lora_" in k for k in snap)
    assert not any(k.startswith("llm.") and "lora_" not in k for k in snap)


def test_multi_token_sequence_features():
    enc = SensorEncoder(hidden_dim=16).eval()
    with torch.no_grad():
        seq = enc.forward_sequence(torch.randn(3, 128, 9), num_tokens=8)
    assert seq.shape == (3, 8, 16)


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
    assert snap and not any(k.startswith("llm.") and "lora_" not in k for k in snap)
    with torch.no_grad():
        model.classification_head.weight.add_(1.0)
    assert not torch.equal(snap["classification_head.weight"], model.classification_head.weight)

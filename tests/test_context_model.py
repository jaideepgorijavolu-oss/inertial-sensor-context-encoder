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


def test_zero_sensor_control_ignores_the_input(model):
    model.eval()
    with torch.no_grad():
        a = model(torch.randn(3, 128, 9), zero_sensor=True)
        b = model(torch.randn(3, 128, 9), zero_sensor=True)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, a[:1].expand_as(a), atol=1e-6)  # identical prediction for every window


def test_global_controls_on_a_tiny_model():
    from torch.utils.data import DataLoader

    from src.dataset import HARDataset
    from src.train import control_metrics, derangement

    perm = derangement(50, torch.Generator().manual_seed(0))
    assert sorted(perm.tolist()) == list(range(50))
    assert not (perm == torch.arange(50)).any()  # no window is paired with itself
    assert torch.equal(perm, derangement(50, torch.Generator().manual_seed(0)))  # seeded

    loader = DataLoader(HARDataset(torch.randn(40, 128, 9).numpy(), (torch.arange(40) % 6).numpy()), batch_size=16)
    controls = control_metrics(tiny_model(), loader, torch.device("cpu"), seed=1, repeats=3)
    assert controls["shuffled_global"]["permutations"] == 3
    assert 0 <= controls["shuffled_global"]["macro_f1"] <= 1
    assert 0 <= controls["zero"]["macro_f1"] <= 1


def test_trainable_state_dict_is_a_snapshot_without_llm(model):
    """Regression test: tensor.cpu() on CPU aliases live weights, so 'best' == last epoch."""
    snap = trainable_state_dict(model)
    assert snap and not any(k.startswith("llm.") and "lora_" not in k for k in snap)
    with torch.no_grad():
        model.classification_head.weight.add_(1.0)
    assert not torch.equal(snap["classification_head.weight"], model.classification_head.weight)

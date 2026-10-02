import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LLM = "HuggingFaceTB/SmolLM2-360M-Instruct"

PROMPT_PREFIX = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\nSensor context: "
)
PROMPT_SUFFIX = "\n\nActivity:"


class SensorEncoder(nn.Module):
    """1D-CNN over a [B, T, C] inertial window -> [B, hidden_dim] feature vector."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T] for Conv1d
        x = x.transpose(1, 2)
        return self.feature_extractor(x).squeeze(-1)

    def forward_sequence(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """[B, T, C] -> [B, num_tokens, hidden_dim]: conv features pooled into temporal segments."""
        feats = self.feature_extractor[:-1](x.transpose(1, 2))  # everything but the global pool
        return F.adaptive_avg_pool1d(feats, num_tokens).transpose(1, 2)


class DirectClassifier(nn.Module):
    """Condition 1: encoder + linear head."""

    def __init__(self, encoder: SensorEncoder, num_classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def make_projector(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim)
    )


class MatchedCapacityClassifier(nn.Module):
    """
    Ablation: the exact trainable stack of the context model (encoder -> projector -> head)
    with the frozen LLM removed. Any gap between this and ContextEmbeddingModel is
    attributable to the LLM itself, not to the extra projector parameters.
    """

    def __init__(self, encoder: SensorEncoder, llm_dim: int = 960, num_classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.projector = make_projector(encoder.hidden_dim, llm_dim)
        self.classification_head = nn.Linear(llm_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.projector(self.encoder(x)))


class ContextEmbeddingModel(nn.Module):
    """
    The sensor window is encoded, projected into the LLM's token-embedding space, and
    spliced between prompt tokens as `num_sensor_tokens` soft tokens (1 = one pooled
    summary; >1 = consecutive temporal segments of the window). The LLM is frozen
    (requires_grad=False) except for any LoRA adapter weights, but gradients always flow
    *through* it into the projector and encoder; the classification head reads the last
    position's final hidden state.
    """

    def __init__(
        self,
        encoder: SensorEncoder,
        llm: nn.Module,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        num_classes: int = 6,
        num_sensor_tokens: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_sensor_tokens = num_sensor_tokens
        # Base transformer (no LM head): we only need hidden states, and the LM head
        # would compute vocab-sized logits for every position for nothing.
        self.llm = llm
        for name, param in self.llm.named_parameters():
            param.requires_grad = "lora_" in name

        llm_dim = self.llm.config.hidden_size
        self.projector = make_projector(encoder.hidden_dim, llm_dim)
        self.classification_head = nn.Linear(llm_dim, num_classes)

        # Buffers move with .to(device); non-persistent so checkpoints stay small.
        self.register_buffer("prefix_ids", prefix_ids.view(1, -1).long(), persistent=False)
        self.register_buffer("suffix_ids", suffix_ids.view(1, -1).long(), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        encoder: SensorEncoder,
        llm_model_name: str = DEFAULT_LLM,
        num_classes: int = 6,
        gradient_checkpointing: bool = False,
        num_sensor_tokens: int = 1,
        lora_r: int = 0,
    ) -> "ContextEmbeddingModel":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm = AutoModel.from_pretrained(llm_model_name, dtype=torch.float32)
        if lora_r:
            llm = add_lora(llm, lora_r)
        if gradient_checkpointing:
            llm.gradient_checkpointing_enable()
        prefix_ids = tokenizer.encode(PROMPT_PREFIX, return_tensors="pt", add_special_tokens=True)
        suffix_ids = tokenizer.encode(PROMPT_SUFFIX, return_tensors="pt", add_special_tokens=False)
        return cls(encoder, llm, prefix_ids, suffix_ids, num_classes, num_sensor_tokens)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode (no dropout) even while training."""
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, x: torch.Tensor, shuffle_embeddings: bool = False) -> torch.Tensor:
        B = x.size(0)
        if self.num_sensor_tokens == 1:
            sensor_feats = self.encoder(x).unsqueeze(1)  # [B, 1, H]
        else:
            sensor_feats = self.encoder.forward_sequence(x, self.num_sensor_tokens)  # [B, K, H]
        sensor_embed = self.projector(sensor_feats)  # [B, K, D]

        if shuffle_embeddings:
            # Negative control: pair every prompt with another sample's sensor token.
            sensor_embed = sensor_embed[torch.randperm(B, device=x.device)]

        embed_tokens = self.llm.get_input_embeddings()
        prefix_embeds = embed_tokens(self.prefix_ids.expand(B, -1))
        suffix_embeds = embed_tokens(self.suffix_ids.expand(B, -1))
        sensor_embed = sensor_embed.to(dtype=prefix_embeds.dtype)

        inputs_embeds = torch.cat([prefix_embeds, sensor_embed, suffix_embeds], dim=1)

        # No torch.no_grad() here: it would detach the output from the sensor embedding,
        # so the encoder/projector would never receive gradients. The frozen weights
        # already have requires_grad=False, so no gradients are stored for them.
        hidden = self.llm(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
        return self.classification_head(hidden[:, -1, :].float())


def add_lora(llm: nn.Module, r: int) -> nn.Module:
    """Inject rank-r LoRA adapters into the attention q/v projections (in place)."""
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
    return inject_adapter_in_model(config, llm)


def trainable_state_dict(model: nn.Module) -> dict:
    """
    Detached *copies* of everything except frozen LLM weights (LoRA adapters are kept).
    Copying matters: on CPU, tensor.cpu() returns the same storage, so a "best" snapshot
    taken that way silently tracks the live weights and ends up equal to the last epoch.
    """
    return {
        k: v.detach().to("cpu", copy=True)
        for k, v in model.state_dict().items()
        if not k.startswith("llm.") or "lora_" in k
    }

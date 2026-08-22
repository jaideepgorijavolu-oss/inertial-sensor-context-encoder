import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class SensorEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, hidden_dim: int = 256):
        super().__init__()
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
        x = x.transpose(1, 2)
        feat = self.feature_extractor(x).squeeze(-1)
        return feat

class DirectClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.classifier(feat)

class ContextEmbeddingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        num_classes: int = 6
    ):
        super().__init__()
        self.encoder = encoder

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)

        for param in self.llm.parameters():
            param.requires_grad = False

        llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(256, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

        self.classification_head = nn.Linear(llm_dim, num_classes)

        self.prefix_str = "Classify the activity as walking, walking upstairs, walking downstairs, sitting, standing, or laying.\n\nSensor context: "
        self.suffix_str = "\n\nActivity:"

        self.prefix_ids = self.tokenizer.encode(self.prefix_str, return_tensors="pt", add_special_tokens=True)
        self.suffix_ids = self.tokenizer.encode(self.suffix_str, return_tensors="pt", add_special_tokens=False)

    def forward(self, x: torch.Tensor, shuffle_embeddings: bool = False) -> torch.Tensor:
        B = x.size(0)
        device = x.device

        sensor_feats = self.encoder(x)
        sensor_embed = self.projector(sensor_feats).unsqueeze(1)

        if shuffle_embeddings:
            perm = torch.randperm(B, device=device)
            sensor_embed = sensor_embed[perm]

        embed_tokens = self.llm.get_input_embeddings()
        prefix_ids = self.prefix_ids.to(device).expand(B, -1)
        suffix_ids = self.suffix_ids.to(device).expand(B, -1)

        prefix_embeds = embed_tokens(prefix_ids)
        suffix_embeds = embed_tokens(suffix_ids)

        input_embeds = torch.cat([prefix_embeds, sensor_embed, suffix_embeds], dim=1)

        outputs = self.llm(inputs_embeds=input_embeds, output_hidden_states=True)

        final_hidden = outputs.hidden_states[-1][:, -1, :]

        return self.classification_head(final_hidden)

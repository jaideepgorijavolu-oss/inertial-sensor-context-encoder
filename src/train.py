import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from dataset import get_dataloaders
from models import SensorEncoder, DirectClassifier, ContextEmbeddingModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model: nn.Module, loader, device: torch.device, shuffle: bool = False) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if isinstance(model, ContextEmbeddingModel):
                logits = model(X, shuffle_embeddings=shuffle)
            else:
                logits = model(X)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return f1_score(all_labels, all_preds, average="macro")

def train_pipeline(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3
):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad or k in model.state_dict()}

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Macro-F1: {val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialized Experiment with Seed: {seed} | Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/UCI HAR Dataset",
        batch_size=64,
        val_subjects=(27, 28, 29, 30)
    )

    print("\n" + "="*50)
    print("Condition 1: Training Direct Sensor Classifier")
    print("="*50)
    encoder_direct = SensorEncoder()
    direct_model = DirectClassifier(encoder_direct).to(device)
    train_pipeline(direct_model, train_loader, val_loader, device, epochs=15, lr=1e-3)
    f1_direct = evaluate(direct_model, test_loader, device)
    print(f"--> Condition 1 (Direct Classifier) Test Macro-F1: {f1_direct:.4f}")

    print("\n" + "="*50)
    print("Condition 2: Training Context-Embedding Model (Frozen SmolLM2)")
    print("="*50)
    encoder_context = SensorEncoder()
    context_model = ContextEmbeddingModel(encoder_context).to(device)

    trainable_count = sum(p.numel() for p in context_model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in context_model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters : {trainable_count:,} (Constraint: < 10,000,000)")
    print(f"Frozen Parameters    : {frozen_count:,}")

    train_pipeline(context_model, train_loader, val_loader, device, epochs=15, lr=5e-4)
    f1_context = evaluate(context_model, test_loader, device, shuffle=False)
    print(f"--> Condition 2 (Context Model) Test Macro-F1: {f1_context:.4f}")

    print("\n" + "="*50)
    print("Condition 3: Sensor-Dependence Check (Shuffled Embeddings)")
    print("="*50)
    f1_shuffled = evaluate(context_model, test_loader, device, shuffle=True)
    print(f"--> Condition 3 (Shuffled Context) Test Macro-F1: {f1_shuffled:.4f}")

    print("\n" + "="*50)
    print("RESULTS SUMMARY TABLE")
    print("="*50)
    print(f"{'Condition':<45} | {'Macro-F1':<10}")
    print("-" * 58)
    print(f"{'Direct sensor classifier':<45} | {f1_direct:.4f}")
    print(f"{'Context-embedding model':<45} | {f1_context:.4f}")
    print(f"{'Context model with shuffled embeddings':<45} | {f1_shuffled:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

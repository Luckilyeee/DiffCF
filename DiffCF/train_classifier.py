import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import load_datasets
from src.models.classifier_fcn import FCNClassifier


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds, meta = load_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=False, num_workers=cfg["dataset"]["num_workers"])

    model = FCNClassifier(meta["channels"], meta["num_classes"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["classifier"]["lr"], weight_decay=cfg["classifier"]["weight_decay"])

    best_acc = 0.0
    run_dir = os.path.join(cfg["output"]["root"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "classifier.pt")

    for epoch in range(cfg["classifier"]["epochs"]):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
        print(f"[Classifier] epoch={epoch} loss={total_loss/len(train_ds):.4f} acc={acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)


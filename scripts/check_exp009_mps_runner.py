from __future__ import annotations

import platform

import torch


def main() -> None:
    print("===== EXP009 HOSTED MPS PREFLIGHT =====")
    print(f"machine={platform.machine()}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")

    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("hosted MPS preflight requires an ARM64 runner")
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError("GitHub-hosted runner does not expose an available MPS device")

    device = torch.device("mps")
    values = torch.arange(4096, dtype=torch.float32, device=device)
    result = torch.sqrt(torch.mean(values.square()) + 1.0)
    torch.mps.synchronize()

    if not bool(torch.isfinite(result).item()):
        raise RuntimeError("MPS smoke computation returned a non-finite result")

    print(f"mps_smoke_value={float(result.cpu()):.9g}")

    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(num_labels=77)
    model = DistilBertForSequenceClassification(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    torch.manual_seed(0)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (32, 128),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((32, 128), dtype=torch.long, device=device)
    labels = torch.randint(0, 77, (32,), dtype=torch.long, device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.mps.synchronize()

    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("synthetic DistilBERT training smoke returned non-finite loss")

    print(f"synthetic_train_loss={float(loss.detach().cpu()):.9g}")
    print("synthetic_train_batch_size=32")
    print("synthetic_train_sequence_length=128")
    print("synthetic_train_labels=77")
    print("banking77_loaded=NO")
    print("exp009_model_checkpoint_loaded=NO")
    print("result_bearing_training=NO")
    print("MPS_TRAINING_SHAPE_SMOKE=PASS")
    print("MPS_HOSTED_PREFLIGHT=PASS")


if __name__ == "__main__":
    main()

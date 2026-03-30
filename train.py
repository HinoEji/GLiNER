import argparse
import json
import torch
from pathlib import Path
from gliner import GLiNER
from gliner.utils import load_config_as_namespace, namespace_to_dict


def load_json_data(path: str):
    """Load JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_cfg: dict, train_cfg: dict):
    """Build or load GLiNER model."""
    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        return GLiNER.from_pretrained(prev_path)
    print("Initializing model from config...")
    return GLiNER.from_config(model_cfg)


def main(cfg_path: str):
    """Main training function."""
    # Load config
    cfg = load_config_as_namespace(cfg_path)

    # Convert to dicts for model building
    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)

    # Setup output directory
    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_json_data(cfg.data.train_data)
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if hasattr(cfg.data, "val_data_dir") and cfg.data.val_data_dir.lower() not in ("none", "null", ""):
        print(f"Loading validation data from: {cfg.data.val_data_dir}")
        eval_dataset = load_json_data(cfg.data.val_data_dir)
        print(f"Validation samples: {len(eval_dataset)}")

    # Build model
    model = build_model(model_cfg, train_cfg).to(dtype=torch.float32)
    print(f"Model type: {model.__class__.__name__}")
    
    # ======= PHẦN MỚI CHÈN VÀO ĐỂ LOAD CÁC LOẠI ENTITY TRƯỚC KHI TRAIN ========
    # Bạn có thể trỏ cố định đến file danh sách entity mong muốn của bạn!
    labels_path = "custom_train_data/v3.3.1/labels.json" # Ví dụ file labels.json
    try:
        model.eval_entity_types = load_json_data(labels_path)
        print(f"Loaded {len(model.eval_entity_types)} entity types for customized evaluation.")
    except Exception as e:
        print(f"No custom labels provided or unable to read {labels_path}. Fallback to automatic class extraction.")
    # =========================================================================
    
    # Get freeze components
    freeze_components = train_cfg.get("freeze_components", None)
    if freeze_components:
        print(f"Freezing components: {freeze_components}")

    # Train
    print("\nStarting training...")
    model.train_model(
        save_only_model=True,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="models",
        # Schedule
        # max_steps=-1,
        lr_scheduler_type=cfg.training.scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        # Batch & optimization
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.train_batch_size,
        learning_rate=float(cfg.training.lr_encoder),
        others_lr=float(cfg.training.lr_others),
        weight_decay=float(cfg.training.weight_decay_encoder),
        others_weight_decay=float(cfg.training.weight_decay_other),
        max_grad_norm=float(cfg.training.max_grad_norm),
        # Loss
        focal_loss_alpha=float(cfg.training.loss_alpha),
        focal_loss_gamma=float(cfg.training.loss_gamma),
        focal_loss_prob_margin=float(getattr(cfg.training, "loss_prob_margin", 0.0)),
        loss_reduction=cfg.training.loss_reduction,
        negatives=float(cfg.training.negatives),
        masking=cfg.training.masking,
        # Logging & saving
        save_strategy=getattr(cfg.training, "save_strategy", "epoch"),
        save_total_limit=getattr(cfg.training, "save_total_limit", 2),
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy=getattr(cfg.training, "eval_strategy", "epoch"),
        
        # Bật lại tính năng lưu model xịn nhất đồng bộ trực tiếp từ file YAML Config
        load_best_model_at_end=getattr(cfg.training, "load_best_model_at_end", True),
        metric_for_best_model=getattr(cfg.training, "metric_for_best_model", "eval_f1"),
        greater_is_better=getattr(cfg.training, "greater_is_better", True),

        num_train_epochs=cfg.training.num_train_epochs,
        report_to=cfg.training.report_to,
        # Freezing
        freeze_components=freeze_components,

        # Dtype
        bf16=True
    )

    print(f"\n✓ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER model")
    parser.add_argument("--config", type=str, default="/kaggle/working/GLiNER/cafebert_config.yaml", help="Path to config file (YAML or JSON)")
    args = parser.parse_args()
    main(args.config)

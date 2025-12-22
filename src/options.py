# TODO: Handle logger
# sd-v1.4
# Train: /data4/user/cheng535/sony_intern/CO-SPY/data/train
# Test:  /data4/user/cheng535/sony_intern/sy_custom_deepfake
# progan
# Train: /data4/user/cheng535/sony_intern/sony_intern_summer_2024/datasets
# Test:  /data4/user/cheng535/sony_intern/sony_intern_summer_2024/AIGCDetect_testset/test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Deep Fake Detection")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--mode", type=str, default="branch", choices=["branch", "fusion"], help="Training mode: branch (artifact/semantic) or fusion")
    parser.add_argument("--detector", type=str, default="artifact", choices=["artifact", "semantic"], help="Detector type (for branch mode)")
    parser.add_argument("--semantic_weights_path", type=str, default="ckpt/semantic/best_model.pth", help="Semantic weights path (for fusion mode)")
    parser.add_argument("--artifact_weights_path", type=str, default="ckpt/artifact/best_model.pth", help="Artifact weights path (for fusion mode)")
    parser.add_argument("--trainset_dirpath", type=str, default="data/train", help="Training directory (for branch mode)")
    parser.add_argument("--calibration_dirpath", type=str, default="data/train", help="Calibration directory (for fusion mode)")
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")

    args = parser.parse_args()

    seed_torch(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        mode=args.mode,
        device=device,
        detector=args.detector,
        semantic_weights_path=args.semantic_weights_path,
        artifact_weights_path=args.artifact_weights_path,
        trainset_dirpath=args.trainset_dirpath,
        calibration_dirpath=args.calibration_dirpath,
        ckpt=args.ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    trainer.train()

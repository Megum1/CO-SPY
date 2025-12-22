import torch
from train import Trainer
from evaluate import Detector
from utils import seed_torch


def main(args):
    #########################################
    # Phase 1: Training
    #########################################
    if args.phase == "train":
        # Initialize Trainer
        trainer = Trainer(
            mode=args.mode,
            device=args.device,
            branch=args.branch,
            train_dataset=args.train_dataset,
            label_smooth=args.label_smooth,
            ckpt=args.ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        # Start training
        trainer.train()
    #########################################
    # Phase 2: Evaluation
    #########################################
    elif args.phase == "eval":
        # Initialize Detector
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size
        )
        # Start evaluation
        detector.evaluate_benchmark()
    ##########################################
    # Phase 3: Test on a single image
    ##########################################
    elif args.phase == "test":
        # Initialize Detector
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size
        )
        # Test on a single image
        score = detector.test_image(args.image_path)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU id to use")
    parser.add_argument("--phase",
                        type=str,
                        default="test",
                        choices=["train", "eval", "test"],
                        help="Select the phase to run Co-Spy: train / eval / test")
    parser.add_argument("--mode",
                        type=str,
                        default="fusion",
                        choices=["branch", "fusion", "end2end"],
                        help="Select the mode of Co-Spy training")
    parser.add_argument("--train_dataset",
                        type=str,
                        default="sd-v1_4",
                        help="Training dataset")
    parser.add_argument("--branch",
                        type=str,
                        default="artifact",
                        choices=["artifact", "semantic"],
                        help="Branch detector (for branch mode)")
    parser.add_argument("--label_smooth",
                        action="store_true",
                        help="Whether to use label smoothing during training")
    parser.add_argument("--pretrain",
                        action="store_true",
                        help="Whether to use pre-trained weights for evaluation")
    parser.add_argument("--ckpt",
                        type=str,
                        default="ckpt",
                        help="Checkpoint directory")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument("--seed",
                        type=int,
                        default=1024,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    seed_torch(args.seed)

    # Set GPU device
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Run the experiment
    main(args)

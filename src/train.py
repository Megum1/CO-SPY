import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import average_precision_score

from utils import seed_torch
from datasets import RealFakeDataset

import warnings
warnings.filterwarnings("ignore")


class Detector:
    def __init__(self, args):
        super(Detector, self).__init__()

        self.device = args.device
        self.mode = args.mode

        # Get the detector based on mode
        if self.mode == "branch":
            from .detectors import ArtifactDetector, SemanticDetector
            if args.detector == "artifact":
                self.model = ArtifactDetector()
            elif args.detector == "semantic":
                self.model = SemanticDetector()
            else:
                raise ValueError(f"Unknown detector: {args.detector}")
            self._lr = 1e-4
        elif self.mode == "fusion":
            from ProGANDetectors import CospyCalibrateDetectorProGAN as CospyCalibrateDetector
            self.model = CospyCalibrateDetector(
                semantic_weights_path=args.semantic_weights_path,
                artifact_weights_path=args.artifact_weights_path)
            self._lr = 1e-1
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.model.to(self.device)

        # Initialize the fc layer
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

        # Optimizer
        _beta1 = 0.9
        _weight_decay = 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(params)}")

        self.optimizer = torch.optim.AdamW(params, lr=self._lr, betas=(_beta1, 0.999), weight_decay=_weight_decay)

        # Loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Scheduler
        self.delr_freq = 10

    def train_step(self, batch_data):
        inputs, labels = batch_data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        self.optimizer.step()

        eval_loss = loss.item()
        y_pred = outputs.sigmoid().flatten().tolist()
        y_true = labels.tolist()
        return eval_loss, y_pred, y_true

    def scheduler(self, status_dict):
        epoch = status_dict["epoch"]
        if epoch % self.delr_freq == 0 and epoch != 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.9
            self._lr = param_group["lr"]
        return True

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        prediction = outputs.sigmoid().flatten().tolist()
        return prediction


def evaluate(y_pred, y_true):
    ap = average_precision_score(y_true, y_pred)
    accuracy = ((np.array(y_pred) > 0.5) == y_true).mean()
    return ap, accuracy


def train(args):
    detector = Detector(args)

    # Determine data split and transform based on mode
    if args.mode == "branch":
        train_split = "train"
        train_transform = detector.model.train_transform
        data_path = args.trainset_dirpath
        model_dir = os.path.join(args.ckpt, args.detector)
    else:  # fusion
        train_split = "val"
        train_transform = detector.model.test_transform
        data_path = args.calibration_dirpath
        model_dir = os.path.join(args.ckpt, "cospy_calibrate")

    # Load the training dataset
    train_dataset = RealFakeDataset(data_path=data_path,
                                    split=train_split,
                                    transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    # Load the validation dataset (only for branch mode)
    if args.mode == "branch":
        val_dataset = RealFakeDataset(data_path=data_path,
                                      split="val",
                                      transform=detector.model.test_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)
        logger.info(f"Train size {len(train_dataset)} | Val size {len(val_dataset)}")
    else:
        val_loader = train_loader  # fusion mode uses same data for eval
        logger.info(f"Train size {len(train_dataset)}")

    # Set the saving directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_path = f"{model_dir}/training.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    logger_id = logger.add(
        log_path,
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG",
    )

    # Train the detector
    best_acc = 0
    for epoch in range(args.epochs):
        detector.model.train()
        time_start = time.time()
        for step_id, batch_data in enumerate(train_loader):
            eval_loss, y_pred, y_true = detector.train_step(batch_data)
            ap, accuracy = evaluate(y_pred, y_true)

            if (step_id + 1) % 100 == 0:
                time_end = time.time()
                logger.info(f"Epoch {epoch} | Batch {step_id + 1}/{len(train_loader)} | Loss {eval_loss:.4f} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}% | Time {time_end-time_start:.2f}s")
                time_start = time.time()

        # Evaluate the model
        detector.model.eval()
        y_pred, y_true = [], []
        for (images, labels) in val_loader:
            y_pred.extend(detector.predict(images))
            y_true.extend(labels.tolist())

        ap, accuracy = evaluate(y_pred, y_true)
        eval_type = "Test" if args.mode == "branch" else "Total"
        logger.info(f"Epoch {epoch} | {eval_type} AP {ap*100:.2f}% | {eval_type} Accuracy {accuracy*100:.2f}%")

        # Schedule the training
        status_dict = {"epoch": epoch, "AP": ap, "Accuracy": accuracy}
        proceed = detector.scheduler(status_dict)
        if not proceed:
            logger.info("Early stopping")
            break

        # Save the model
        if accuracy >= best_acc:
            best_acc = accuracy
            detector.model.save_weights(f"{model_dir}/best_model.pth")
            logger.info(f"Best model saved with accuracy {best_acc*100:.2f}%")

        if epoch % 5 == 0:
            detector.model.save_weights(f"{model_dir}/epoch_{epoch}.pth")
            logger.info(f"Model saved at epoch {epoch}")

    # Save the final model
    detector.model.save_weights(f"{model_dir}/final_model.pth")
    logger.info("Final model saved")

    logger.remove(logger_id)


def test(args):
    from Datasets import TestDataset, EVAL_DATASET_LIST, EVAL_MODEL_LIST

    detector = Detector(args)

    # Load the model
    if args.mode == "branch":
        model_dir = args.detector
    else:
        model_dir = "cospy_calibrate"

    weights_path = os.path.join(args.ckpt, model_dir, "best_model.pth")
    detector.model.load_weights(weights_path)
    detector.model.to(args.device)
    detector.model.eval()

    test_transform = detector.model.test_transform

    # Set the saving directory
    save_dir = os.path.join(args.ckpt, model_dir)
    save_result_path = os.path.join(save_dir, "result.json")
    save_output_path = os.path.join(save_dir, "output.json")

    # Begin the evaluation
    result_all = {}
    output_all = {}
    for dataset_name in EVAL_DATASET_LIST:
        result_all[dataset_name] = {}
        output_all[dataset_name] = {}
        for model_name in EVAL_MODEL_LIST:
            test_dataset = TestDataset(dataset=dataset_name, model=model_name, root_path=args.testset_dirpath, transform=test_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)

            y_pred, y_true = [], []
            for (images, labels) in tqdm(test_loader, desc=f"Evaluating {dataset_name} {model_name}"):
                y_pred.extend(detector.predict(images))
                y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            print(f"Evaluate on {dataset_name} {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

            result_all[dataset_name][model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
            output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}

    with open(save_result_path, "w") as f:
        json.dump(result_all, f, indent=4)

    with open(save_output_path, "w") as f:
        json.dump(output_all, f, indent=4)


def test_AIGC(args):
    detector = Detector(args)

    # Load the model
    if args.mode == "branch":
        model_dir = args.detector
    else:
        model_dir = "cospy_calibrate"

    weights_path = os.path.join(args.ckpt, model_dir, "best_model.pth")
    detector.model.load_weights(weights_path)
    detector.model.to(args.device)
    detector.model.eval()

    test_transform = detector.model.test_transform

    # Set the saving directory
    save_dir = os.path.join(args.ckpt, model_dir)
    save_result_path = os.path.join(save_dir, "result.json")
    save_output_path = os.path.join(save_dir, "output.json")

    # Begin the evaluation
    test_root = args.aigc_testset_dirpath
    aigc_model_list = os.listdir(test_root)

    result_all = {}
    output_all = {}

    for model_name in aigc_model_list:
        test_dataset = RealFakeDataset(data_path=os.path.join(test_root, model_name), split="test", transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

        y_pred, y_true = [], []
        for (images, labels) in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            y_pred.extend(detector.predict(images))
            y_true.extend(labels.tolist())

        ap, accuracy = evaluate(y_pred, y_true)
        print(f"Evaluate on {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

        result_all[model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
        output_all[model_name] = {"y_pred": y_pred, "y_true": y_true}

        with open(save_result_path, "w") as f:
            json.dump(result_all, f, indent=4)

        with open(save_output_path, "w") as f:
            json.dump(output_all, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Deep Fake Detection")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--mode", type=str, default="branch", choices=["branch", "fusion"], help="Training mode: branch (artifact/semantic) or fusion")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test", "test_aigc"], help="Phase of the experiment")
    parser.add_argument("--detector", type=str, default="artifact", choices=["artifact", "semantic"], help="Detector type (for branch mode)")
    parser.add_argument("--semantic_weights_path", type=str, default="ckpt/semantic/best_model.pth", help="Semantic weights path (for fusion mode)")
    parser.add_argument("--artifact_weights_path", type=str, default="ckpt/artifact/best_model.pth", help="Artifact weights path (for fusion mode)")
    parser.add_argument("--trainset_dirpath", type=str, default="data/train", help="Training directory (for branch mode)")
    parser.add_argument("--calibration_dirpath", type=str, default="data/train", help="Calibration directory (for fusion mode)")
    parser.add_argument("--testset_dirpath", type=str, default="data/test", help="Testset directory")
    parser.add_argument("--aigc_testset_dirpath", type=str, default="data/aigc_test", help="AIGC testset directory")
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")

    args = parser.parse_args()

    seed_torch(args.seed)

    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.phase == "train":
        train(args)
    elif args.phase == "test":
        test(args)
    elif args.phase == "test_aigc":
        test_AIGC(args)
    else:
        raise ValueError("Unknown phase")

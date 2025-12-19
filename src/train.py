import os
import time
import torch
import importlib
from loguru import logger

from utils import seed_torch, evaluate
from datasets import RealFakeDataset

import warnings
warnings.filterwarnings("ignore")


# TODO: Handle logger

class Trainer:
    def __init__(self,
                 mode: str,
                 device: str,
                 detector: str,
                 train_dataset: str,
                 semantic_weights_path: str = None,
                 artifact_weights_path: str = None,
                 label_smooth: bool = False,
                 ckpt: str = "ckpt",
                 epochs: int = 20,
                 batch_size: int = 32):
        super(Trainer, self).__init__()

        self.device = device
        self.mode = mode
        self.detector = detector
        self.train_dataset = train_dataset
        self.semantic_weights_path = semantic_weights_path
        self.artifact_weights_path = artifact_weights_path
        self.label_smooth = label_smooth
        self.ckpt = ckpt
        self.epochs = epochs
        self.batch_size = batch_size

        # Dynamically import the detector module from either "progan" or "sd-v1.4"
        detector_module = importlib.import_module(f"detectors.{train_dataset}")

        # Get the detector based on mode
        if self.mode == "branch":
            ArtifactDetector = getattr(detector_module, "ArtifactDetector")
            SemanticDetector = getattr(detector_module, "SemanticDetector")
            if self.detector == "artifact":
                self.model = ArtifactDetector()
            elif self.detector == "semantic":
                self.model = SemanticDetector()
            else:
                raise ValueError(f"Unknown detector: {self.detector}")
        elif self.mode == "fusion":
            CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
            self.model = CoSpyFusionDetector(
                semantic_weights_path=self.semantic_weights_path,
                artifact_weights_path=self.artifact_weights_path)
        elif self.mode == "end2end":
            End2EndDetector = getattr(detector_module, "End2EndDetector")
            self.model = End2EndDetector()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.model.to(self.device)

        # Initialize the fc layer
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)
        if self.mode == "end2end":
            torch.nn.init.normal_(self.model.sem.fc.weight.data, 0.0, 0.02)
            torch.nn.init.normal_(self.model.art.fc.weight.data, 0.0, 0.02)

        # Optimizer
        _beta1 = 0.9
        _weight_decay = 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(params)}")

        self._lr = 1e-4 if self.mode != "fusion" else 1e-1
        self.optimizer = torch.optim.AdamW(params, lr=self._lr, betas=(_beta1, 0.999), weight_decay=_weight_decay)

        # Loss function
        if self.label_smooth:
            self.criterion = LabelSmoothingBCEWithLogits(smoothing=0.1)
        else:
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

    def train(self):
        # Determine data split and transform based on mode
        if self.mode == "fusion":
            train_split = "train"
            train_transform = self.model.train_transform
            data_path = self.trainset_dirpath
            model_dir = os.path.join(self.ckpt, self.detector)
        else:  # fusion
            train_split = "val"
            train_transform = self.model.test_transform
            data_path = self.calibration_dirpath
            model_dir = os.path.join(self.ckpt, "cospy_calibrate")

        # Load the training dataset
        train_dataset = RealFakeDataset(data_path=data_path,
                                        split=train_split,
                                        transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)

        # Load the validation dataset (only for branch mode)
        if self.mode == "branch":
            val_dataset = RealFakeDataset(data_path=data_path,
                                          split="val",
                                          transform=self.model.test_transform)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=self.batch_size,
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
        for epoch in range(self.epochs):
            self.model.train()
            time_start = time.time()
            for step_id, batch_data in enumerate(train_loader):
                eval_loss, y_pred, y_true = self.train_step(batch_data)
                ap, accuracy = evaluate(y_pred, y_true)

                if (step_id + 1) % 100 == 0:
                    time_end = time.time()
                    logger.info(f"Epoch {epoch} | Batch {step_id + 1}/{len(train_loader)} | Loss {eval_loss:.4f} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}% | Time {time_end-time_start:.2f}s")
                    time_start = time.time()

            # Evaluate the model
            self.model.eval()
            y_pred, y_true = [], []
            for (images, labels) in val_loader:
                y_pred.extend(self.model.predict(images))
                y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            eval_type = "Test" if self.mode == "branch" else "Total"
            logger.info(f"Epoch {epoch} | {eval_type} AP {ap*100:.2f}% | {eval_type} Accuracy {accuracy*100:.2f}%")

            # Schedule the training
            status_dict = {"epoch": epoch, "AP": ap, "Accuracy": accuracy}
            proceed = self.scheduler(status_dict)
            if not proceed:
                logger.info("Early stopping")
                break

            # Save the model
            if accuracy >= best_acc:
                best_acc = accuracy
                self.model.save_weights(f"{model_dir}/best_model.pth")
                logger.info(f"Best model saved with accuracy {best_acc*100:.2f}%")

            if epoch % 5 == 0:
                self.model.save_weights(f"{model_dir}/epoch_{epoch}.pth")
                logger.info(f"Model saved at epoch {epoch}")

        # Save the final model
        self.model.save_weights(f"{model_dir}/final_model.pth")
        logger.info("Final model saved")

        logger.remove(logger_id)


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

import os
import time
import torch
import importlib
from loguru import logger

from utils import seed_torch, evaluate
from dataSets import TrainDataset

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self,
                 mode: str,
                 device: str,
                 branch: str,
                 train_dataset: str,
                 label_smooth: bool = False,
                 ckpt: str = "ckpt",
                 epochs: int = 20,
                 batch_size: int = 32):

        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.label_smooth = label_smooth
        self.ckpt = ckpt
        self.epochs = epochs
        self.batch_size = batch_size

        # Dynamically import the detector module from either "progan" or "sd-v1_4"
        detector_module = importlib.import_module(f"detectors.{train_dataset}")

        # Get the detector based on mode
        if self.mode == "branch":
            ArtifactDetector = getattr(detector_module, "ArtifactDetector")
            SemanticDetector = getattr(detector_module, "SemanticDetector")
            if self.branch == "artifact":
                self.model = ArtifactDetector()
            elif self.branch == "semantic":
                self.model = SemanticDetector()
            else:
                raise ValueError(f"Unknown detector: {self.branch}")
        elif self.mode == "fusion":
            semantic_weights_path = os.path.join(self.ckpt, self.train_dataset, "semantic", "best_model.pth")
            artifact_weights_path = os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth")
            if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path):
                raise ValueError("Semantic or Artifact weights path does not exist for fusion mode")
            CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
            self.model = CoSpyFusionDetector(
                semantic_weights_path=semantic_weights_path,
                artifact_weights_path=artifact_weights_path)
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

    def train(self):
        # Determine data split and transform based on mode
        if self.mode == "fusion":
            train_split, val_split = "val", "val"
            train_transform = self.model.test_transform
            test_transform = self.model.test_transform
        else:  # branch or end2end mode
            train_split, val_split = "train", "val"
            train_transform = self.model.train_transform
            test_transform = self.model.test_transform
        
        # Determine save directory
        if self.mode == "branch":
            subdir = self.branch
        else:
            subdir = self.mode
        # Set the saving directory
        model_dir = os.path.join(self.ckpt, self.train_dataset, subdir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Setup logger
        log_path = f"{model_dir}/training.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        
        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        # Add JPEG compression for sd-v1_4 dataset
        self.add_jpeg = True if self.train_dataset == "sd-v1_4" else False

        # Load the training and validation dataset
        train_dataset = TrainDataset(train_dataset=self.train_dataset,
                                     split=train_split,
                                     add_jpeg=self.add_jpeg,
                                     transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)
        val_dataset = TrainDataset(train_dataset=self.train_dataset,
                                   split=val_split,
                                   add_jpeg=self.add_jpeg,
                                   transform=test_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

        logger.info(f"Train size {len(train_dataset)} | Val size {len(val_dataset)}")

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
            self.scheduler(status_dict)

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

        # Remove the logger
        logger.remove(logger_id)

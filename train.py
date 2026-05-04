import os
import time
import torch
import importlib
from PIL import Image
from loguru import logger

from utils import evaluate
from dataSets import TrainDataset, FusionTrainDataset

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self,
                 mode: str,
                 device: str,
                 branch: str,
                 train_dataset: str,
                 ckpt: str = "ckpt",
                 epochs: int = 10,
                 batch_size: int = 32,
                 feat_interp: bool = False,
                 feat_interp_alpha: float = 0.2,
                 feat_interp_ratio: float = 0.5):

        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.ckpt = ckpt
        self.epochs = epochs
        self.batch_size = batch_size
        self.feat_interp = feat_interp
        self.feat_interp_alpha = feat_interp_alpha
        self.feat_interp_ratio = feat_interp_ratio

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
            self.weight_init = 0.5 if self.train_dataset == "sd-v1_4" else 0.0
            self.bias_floor = 2.0 if self.train_dataset == "sd-v1_4" else 0.0
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

        if self.mode == "fusion":
            with torch.no_grad():
                self.model.fc.weight.data += self.weight_init
                self.model.fc.bias.data.fill_(self.bias_floor)

        # Optimizer
        _beta1 = 0.9
        _weight_decay = 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(params)}")

        self._lr = 1e-4 if self.mode != "fusion" else 1e-1
        self._lr_min = self._lr / self.epochs
        self.optimizer = torch.optim.AdamW(params, lr=self._lr, betas=(_beta1, 0.999), weight_decay=_weight_decay)

        # Loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.lr_scheduler = None

        # Set lr_step_per_batch based on the training dataset
        if self.train_dataset == "sd-v1_4":
            self.lr_step_per_batch = True
        elif self.train_dataset == "progan":
            self.lr_step_per_batch = False
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

    def _mix_features(self, feats, labels):
        """Feature-space Interpolation on a fraction of the batch.

        For self.feat_interp_ratio of the batch, interpolate:
            feat_mixed  = delta * feat_a + (1 - delta) * feat_b
            label_mixed = delta * label_a + (1 - delta) * label_b
        with delta ~ Beta(alpha, alpha) sampled independently per mixed sample.
        """
        B = feats.size(0)
        device = feats.device
        labels_f = labels.float().clone()
        mix_mask = torch.rand(B, device=device) < self.feat_interp_ratio
        n_mix = int(mix_mask.sum().item())
        if n_mix == 0:
            return feats, labels_f
        mix_idx = torch.where(mix_mask)[0]
        partner = torch.randint(0, B, (n_mix,), device=device)
        beta = torch.distributions.Beta(self.feat_interp_alpha, self.feat_interp_alpha)
        delta = beta.sample((n_mix,)).to(device)
        feats_new = feats.clone()
        d2 = delta.unsqueeze(1)
        feats_new[mix_idx] = d2 * feats[mix_idx] + (1.0 - d2) * feats[partner]
        labels_new = labels_f.clone()
        labels_new[mix_idx] = delta * labels_f[mix_idx] + (1.0 - delta) * labels_f[partner]
        return feats_new, labels_new

    def train_step(self, batch_data):
        self.optimizer.zero_grad()

        if self.mode == "fusion":
            sem_t, art_t, labels = batch_data
            sem_t = sem_t.to(self.device, non_blocking=True)
            art_t = art_t.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            pred_sem = self.model.sem(sem_t)
            pred_art = self.model.art(art_t)
            feat = torch.cat([pred_sem, pred_art], dim=1)
            outputs = self.model.fc(feat)
            loss = self.criterion(outputs, labels.unsqueeze(1).float())
            b = self.model.fc.bias[0]
            loss = loss + torch.nn.functional.relu(self.bias_floor - b)
        else:
            inputs, labels = batch_data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Feature-space interpolation
            use_feat_interp = (
                self.feat_interp
                and self.mode == "branch"
                and self.branch == "semantic"
            )
            if use_feat_interp:
                with torch.no_grad():
                    # Fallback: sd-v1_4 open_clip uses .encode_image, progan HF CLIPModel uses .get_image_features
                    encode_fn = getattr(self.model.clip, "encode_image", None) or self.model.clip.get_image_features
                    feats = encode_fn(inputs)
                feats_mixed, labels_soft = self._mix_features(feats, labels)
                outputs = self.model.fc(feats_mixed)
                loss = self.criterion(outputs, labels_soft.unsqueeze(1))
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None and self.lr_step_per_batch:
            self.lr_scheduler.step()

        eval_loss = loss.item()
        y_pred = outputs.sigmoid().flatten().tolist()
        y_true = labels.tolist()
        return eval_loss, y_pred, y_true

    def _build_scheduler(self, steps_per_epoch: int):
        """Construct the LR scheduler appropriate for this training dataset.

        - sd-v1_4: per-step CosineAnnealingLR from self._lr to self._lr_min over (epochs * steps_per_epoch) optimizer steps.

        - progan: _step_lr_per_epoch() adjusts lr *= 0.9 when epoch % 10 == 0 and != 0, applied after each epoch's validation.
        """
        if self.lr_step_per_batch:
            total_steps = self.epochs * steps_per_epoch
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self._lr_min,
            )
            logger.info(
                f"LR cosine {self._lr:.1e} -> {self._lr_min:.1e} over {total_steps} steps"
            )
        else:
            logger.info(
                f"LR step-decay start {self._lr:.1e}, x0.9 every 10 epochs"
            )

    def _step_lr_per_epoch(self, epoch: int):
        """Original Co-Spy rule: lr *= 0.9 when epoch is a non-zero multiple
        of 10. `epoch` is the index just finished (0-based)."""
        if epoch % 10 == 0 and epoch != 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= 0.9

    def train(self):
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

        # Load the training and validation dataset
        if self.mode == "fusion":
            # Match train_fusion_v4: each branch applies its own transform to
            # the raw PIL image (so sem / art see their native aug + resize + normalize).
            base_train = TrainDataset(train_dataset=self.train_dataset, split="val", transform=None)
            base_val = TrainDataset(train_dataset=self.train_dataset, split="val", transform=None)
            train_dataset = FusionTrainDataset(base_train, self.model.sem.train_transform, self.model.art.train_transform)
            val_dataset = FusionTrainDataset(base_val, self.model.sem.val_transform, self.model.art.val_transform)
        else:
            train_dataset = TrainDataset(train_dataset=self.train_dataset,
                                         split="train",
                                         transform=self.model.train_transform)
            val_dataset = TrainDataset(train_dataset=self.train_dataset,
                                       split="val",
                                       transform=self.model.val_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

        logger.info(f"Train size {len(train_dataset)} | Val size {len(val_dataset)}")
        if self.feat_interp and self.mode == "branch" and self.branch == "semantic":
            logger.info(
                f"feat_interp ON | alpha={self.feat_interp_alpha} "
                f"ratio={self.feat_interp_ratio}"
            )

        # Build LR scheduler now that we know len(train_loader).
        self._build_scheduler(steps_per_epoch=len(train_loader))

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
            with torch.no_grad():
                for batch in val_loader:
                    if self.mode == "fusion":
                        sem_t, art_t, labels = batch
                        sem_t = sem_t.to(self.device, non_blocking=True)
                        art_t = art_t.to(self.device, non_blocking=True)
                        feat = torch.cat([self.model.sem(sem_t), self.model.art(art_t)], dim=1)
                        logits = self.model.fc(feat)
                        y_pred.extend(logits.sigmoid().flatten().cpu().tolist())
                        y_true.extend(labels.tolist())
                    else:
                        images, labels = batch
                        y_pred.extend(self.model.predict(images))
                        y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            eval_type = "Test" if self.mode == "branch" else "Total"
            cur_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} | {eval_type} AP {ap*100:.2f}% | {eval_type} Accuracy {accuracy*100:.2f}% | LR {cur_lr:.2e}")

            if not self.lr_step_per_batch:
                self._step_lr_per_epoch(epoch)

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

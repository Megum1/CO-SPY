import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import average_precision_score

from utils import seed_torch
from Detectors import CospyDetector, LabelSmoothingBCEWithLogits
from Datasets import TrainDataset, TestDataset, EVAL_DATASET_LIST, EVAL_MODEL_LIST

import warnings
warnings.filterwarnings("ignore")


class Detector():
    def __init__(self, args):
        super(Detector, self).__init__()

        # Device
        self.device = args.device

        # Get the detector
        self.model = CospyDetector()

        # Put the model on the device
        self.model.to(self.device)

        # Initialize the fc layer
        torch.nn.init.normal_(self.model.sem.fc.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(self.model.art.fc.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

        # Optimizer
        _lr = 1e-4
        _beta1 = 0.9
        _weight_decay = 0.0
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
        print(f"Trainable parameters: {len(params)}")

        self.optimizer = torch.optim.AdamW(params, lr=_lr, betas=(_beta1, 0.999), weight_decay=_weight_decay)

        # Loss function
        if args.no_label_smooth:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = LabelSmoothingBCEWithLogits(smoothing=0.1)

        # Scheduler
        self.delr_freq = 10

    # Training function for the detector
    def train_step(self, batch_data):
        # Decompose the batch data
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

    # Schedule the training
    # Early stopping / learning rate adjustment
    def scheduler(self, status_dict):
        epoch = status_dict["epoch"]
        if epoch % self.delr_freq == 0 and epoch != 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.9
            self.lr = param_group["lr"]
        return True
    
    # Prediction function
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
    # Get the detector
    detector = Detector(args)

    # Load the dataset
    train_dataset = TrainDataset(data_path=args.trainset_dirpath,
                                 split="train",
                                 transform=detector.model.train_transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True)

    test_dataset  = TrainDataset(data_path=args.trainset_dirpath,
                                 split="val",
                                 transform=detector.model.test_transform)
    test_loader   = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True)

    logger.info(f"Train size {len(train_dataset)} | Test size {len(test_dataset)}")

    # Set the saving directory
    model_dir = os.path.join(args.ckpt, "cospy")
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
        # Set the model to training mode
        detector.model.train()
        time_start = time.time()
        for step_id, batch_data in enumerate(train_loader):
            eval_loss, y_pred, y_true = detector.train_step(batch_data)
            ap, accuracy = evaluate(y_pred, y_true)

            # Log the training information
            if (step_id + 1) % 100 == 0:
                time_end = time.time()
                logger.info(f"Epoch {epoch} | Batch {step_id + 1}/{len(train_loader)} | Loss {eval_loss:.4f} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}% | Time {time_end-time_start:.2f}s")
                time_start = time.time()
        
        # Evaluate the model
        detector.model.eval()
        y_pred, y_true = [], []
        for (images, labels) in test_loader:
            y_pred.extend(detector.predict(images))
            y_true.extend(labels.tolist())

        ap, accuracy = evaluate(y_pred, y_true)
        logger.info(f"Epoch {epoch} | Test AP {ap*100:.2f}% | Test Accuracy {accuracy*100:.2f}%")

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
            logger.info(f"Best model saved with accuracy {best_acc.mean()*100:.2f}%")

        if epoch % 5 == 0:
            detector.model.save_weights(f"{model_dir}/epoch_{epoch}.pth")
            logger.info(f"Model saved at epoch {epoch}")

    # Save the final model
    detector.model.save_weights(f"{model_dir}/final_model.pth")
    logger.info("Final model saved")

    # Remove the logger
    logger.remove(logger_id)


def test(args):
    # Initialize the detector
    detector = Detector(args)

    # Load the [best/final] model
    weights_path = os.path.join(args.ckpt, "cospy", "best_model.pth")

    detector.model.load_weights(weights_path)
    detector.model.to(args.device)
    detector.model.eval()

    # Set the pre-processing function
    test_transform = detector.model.test_transform

    # Set the saving directory
    save_dir = os.path.join(args.ckpt, "cospy")
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

            # Evaluate the model
            y_pred, y_true = [], []
            for (images, labels) in tqdm(test_loader, desc=f"Evaluating {dataset_name} {model_name}"):
                y_pred.extend(detector.predict(images))
                y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            print(f"Evaluate on {dataset_name} {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

            result_all[dataset_name][model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
            output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}

    # Save the results
    with open(save_result_path, "w") as f:
        json.dump(result_all, f, indent=4)

    with open(save_output_path, "w") as f:
        json.dump(output_all, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Deep Fake Detection")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--phase", type=str, default="test", choices=["train", "test"], help="Phase of the experiment")
    parser.add_argument("--no_label_smooth", action="store_true", help="Whether to use label smoothing")
    parser.add_argument("--trainset_dirpath", type=str, default="data/train", help="Trainset directory")
    parser.add_argument("--testset_dirpath", type=str, default="data/test", help="Testset directory")
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")

    args = parser.parse_args()

    # Set the random seed
    seed_torch(args.seed)

    # Set the GPU ID
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Begin the experiment
    if args.phase == "train":
        train(args)
    elif args.phase == "test":
        test(args)
    else:
        raise ValueError("Unknown phase")

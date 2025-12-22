import os
import json
import torch
import importlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from loguru import logger

from dataSets import *
from utils import seed_torch, evaluate

import warnings
warnings.filterwarnings("ignore")


# Load pre-trained models for evaluation
class Detector:
    def __init__(self,
                 device: str,
                 mode: str = "fusion",
                 train_dataset: str = "sd-v1_4",
                 pretrain: bool = False,
                 ckpt: str = "ckpt",
                 batch_size: int = 32):

        # Device
        self.device = device
        self.mode = mode
        self.train_dataset = train_dataset
        self.pretrain = pretrain
        self.ckpt = ckpt
        self.batch_size = batch_size

        # Dynamically import the detector module from either "progan" or "sd-v1_4"
        detector_module = importlib.import_module(f"detectors.{train_dataset}")

        # Get the detector and load weights based on mode
        if pretrain:
            # Only provide pre-trained weights for fusion mode
            # Hardcode to fusion mode
            self.mode = "fusion"
            # Load the fusion detector with pre-trained weights
            semantic_weights_path = f"pretrained/{self.train_dataset}/semantic_weights.pth"
            artifact_weights_path = f"pretrained/{self.train_dataset}/artifact_weights.pth"
            fusion_weights_path = f"pretrained/{self.train_dataset}/fusion_weights.pth"
            if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path) or not os.path.exists(fusion_weights_path):
                raise ValueError("The pre-trained weights are not complete for evaluation")
            CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
            self.model = CoSpyFusionDetector(
                semantic_weights_path=semantic_weights_path,
                artifact_weights_path=artifact_weights_path)
            self.model.load_weights(fusion_weights_path)
        else:
            if self.mode == "fusion":
                semantic_weights_path = os.path.join(self.ckpt, self.train_dataset, "semantic", "best_model.pth")
                artifact_weights_path = os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth")
                fusion_weights_path = os.path.join(self.ckpt, self.train_dataset, "fusion", "best_model.pth")
                if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path) or not os.path.exists(fusion_weights_path):
                    raise ValueError("Semantic, Artifact or Fusion weights path does not exist for fusion mode")
                CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
                self.model = CoSpyFusionDetector(
                    semantic_weights_path=semantic_weights_path,
                    artifact_weights_path=artifact_weights_path)
                self.model.load_weights(fusion_weights_path)
            elif self.mode == "end2end":
                End2EndDetector = getattr(detector_module, "End2EndDetector")
                self.model = End2EndDetector()
                end2end_weights_path = os.path.join(self.ckpt, self.train_dataset, "end2end", "best_model.pth")
                if not os.path.exists(end2end_weights_path):
                    raise ValueError("End2End weights path does not exist for end2end mode")
                self.model.load_weights(end2end_weights_path)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        
        # Put the model on the device and set to eval
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_benchmark(self):
        # Select the appropriate test dataset and evaluation lists
        if self.train_dataset == "progan":
            benchmark_name = "AIGCDetectionBenchMark"
            TestDataset = AIGCDetectTestDataset
            eval_dataset_list = AIGCDetectionBenchMark_DATASET_LIST
            eval_model_list = AIGCDetectionBenchMark_MODEL_LIST
        elif self.train_dataset == "sd-v1_4":
            benchmark_name = "Co-Spy-Bench"
            TestDataset = CoSpyBenchTestDataset
            eval_dataset_list = CoSpyBench_DATASET_LIST
            eval_model_list = CoSpyBench_MODEL_LIST
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

        # Set the saving directory
        save_dir = os.path.join(self.ckpt, self.train_dataset, self.mode, f"eval_{benchmark_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Setup logger
        log_path = f"{save_dir}/evaluation.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        
        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        # Save raw model prediction
        save_output_path = os.path.join(save_dir, "output.json")
        # Save summarized evaluation result
        save_result_path = os.path.join(save_dir, "result.json")

        # Begin the evaluation
        result_all = {}
        output_all = {}
        for dataset_name in eval_dataset_list:
            result_all[dataset_name] = {}
            output_all[dataset_name] = {}
            for model_name in eval_model_list:
                test_dataset = TestDataset(dataset=dataset_name, model=model_name, transform=self.model.test_transform)
                test_loader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=4,
                                                          pin_memory=True)

                # Evaluate the model
                y_pred, y_true = [], []
                for (images, labels) in tqdm(test_loader, desc=f"Evaluating {benchmark_name} - {dataset_name} - {model_name}"):
                    y_pred.extend(self.model.predict(images))
                    y_true.extend(labels.tolist())

                ap, accuracy = evaluate(y_pred, y_true)
                logger.info(f"Evaluate on {benchmark_name} - {dataset_name} - {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

                result_all[dataset_name][model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
                output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}

        # Save the results
        with open(save_result_path, "w") as f:
            json.dump(result_all, f, indent=4)

        with open(save_output_path, "w") as f:
            json.dump(output_all, f, indent=4)

    def scan(self):
        # Load the image
        image_filepath = input("Please enter the image filepath for scanning: ")
        if not os.path.exists(image_filepath):
            print(f"Image file not found: {image_filepath}")
            image_filepath = input("Please enter the image filepath for scanning: ")

        image = Image.open(image_filepath).convert("RGB")
        image = self.model.test_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        # Make the prediction
        prediction = self.predict(image)[0]

        if prediction > 0.5:
            print(f"Co-Spy Prediction: {prediction:.3f} - AI-Generated")
        else:
            print(f"Co-Spy Prediction: {prediction:.3f} - Real")

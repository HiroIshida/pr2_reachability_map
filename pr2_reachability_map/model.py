from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


@dataclass
class ReachabilityClassifier:
    fcn: FCN
    threshold: float = 0.1

    def predict(self, x: np.ndarray) -> bool:
        x = torch.from_numpy(x).float().unsqueeze(0)
        return self.fcn(x).item() > self.threshold


def get_model_path(arm: Literal["rarm", "larm"]):
    pretrained_path = (Path(__file__).parent / "pretrained").expanduser()
    if arm == "rarm":
        model_path = pretrained_path / "best_model_rarm.pth"
    elif arm == "larm":
        model_path = pretrained_path / "best_model_larm.pth"
    else:
        raise ValueError(f"Invalid arm {arm}")
    return model_path


def load_model(arm: Literal["rarm", "larm"]) -> ReachabilityClassifier:
    model = FCN(7, 1)
    model_path = get_model_path(arm)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    dummy_input = torch.randn(1, 7)
    traced = torch.jit.trace(model, (dummy_input,))
    optimized = torch.jit.optimize_for_inference(traced)
    return ReachabilityClassifier(optimized)


if __name__ == "__main__":
    model = load_model("rarm")
    print(model)
    print("Model loaded successfully")

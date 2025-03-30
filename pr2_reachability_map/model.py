from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw


@dataclass
class Domain:
    lb: np.ndarray = np.array([-0.5, -1.5, 0.0, -np.pi, -np.pi, -np.pi])
    ub: np.ndarray = np.array([1.5, 1.5, 2.0, np.pi, np.pi, np.pi])

    def sample_point(self) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub)


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
    domain: Domain = Domain()

    def predict(self, x: np.ndarray) -> bool:
        pos = x[:3]
        if np.any(pos < self.domain.lb[:3]) or np.any(pos > self.domain.ub[:3]):
            return False
        if len(x) == 6:
            rpy = x[3:]
            quat = wxyz2xyzw(rpy2quaternion(rpy[::-1]))
            x = np.hstack([pos, quat])
        x = torch.from_numpy(x).float().unsqueeze(0)
        ret = self.fcn(x).item()
        return ret < self.threshold


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
    domain = Domain()
    p = np.array([0.6, -0.3, 0.8, 0.0, 0.0, 0.0])
    print(model.predict(p))

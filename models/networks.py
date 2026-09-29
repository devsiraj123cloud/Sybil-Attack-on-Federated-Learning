import torch
import torch.nn as nn
import torch.nn.functional as F

class FastMLP(nn.Module):
    """
    Lightweight Multi-Layer Perceptron for fast FL experimentation.
    Input: 28x28 = 784 features (MNIST/Fashion-MNIST)
    Output: 10 classes
    """
    def __init__(self, input_dim=784, hidden_dim1=128, hidden_dim2=64, num_classes=10):
        super(FastMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FederatedCNN(nn.Module):
    """
    Standard Convolutional Neural Network for Image Classification in FL.
    Input: (B, 1, 28, 28)
    Output: (B, 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(FederatedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(model_name="cnn", device="cpu"):
    """Factory function to instantiate models."""
    if model_name.lower() == "mlp":
        model = FastMLP()
    elif model_name.lower() == "cnn":
        model = FederatedCNN()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return model.to(device)

def parameters_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)

def vector_to_parameters(vec: torch.Tensor, model: nn.Module) -> None:
    """Load a 1D tensor back into model parameters."""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data.copy_(vec[pointer : pointer + num_param].view_as(param))
        pointer += num_param

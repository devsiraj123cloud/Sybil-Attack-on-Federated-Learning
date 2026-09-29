import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.networks import parameters_to_vector, vector_to_parameters

class FederatedClient:
    """
    Simulates a single client in the Federated Learning network.
    Can be configured as an Honest Client or Sybil Client.
    """
    def __init__(
        self,
        client_id: int,
        dataset,
        model_template: nn.Module,
        device="cpu",
        batch_size=32,
        local_epochs=2,
        learning_rate=0.01,
        is_sybil=False,
        label_flip_target=None
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model_template).to(device)
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.is_sybil = is_sybil
        self.label_flip_target = label_flip_target # tuple (source_class, target_class)

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def local_train(self, global_params: torch.Tensor) -> tuple:
        """
        Executes local SGD training on local client data.
        Returns:
            update: 1D torch.Tensor = w_local - w_global
            num_samples: int
            metrics: dict (avg_loss, accuracy)
        """
        vector_to_parameters(global_params, self.model)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # If this Sybil performs label flipping attack
                if self.is_sybil and self.label_flip_target is not None:
                    src, tgt = self.label_flip_target
                    mask = (batch_y == src)
                    batch_y = batch_y.clone()
                    batch_y[mask] = tgt

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total_samples += batch_x.size(0)

        local_params = parameters_to_vector(self.model)
        update = local_params - global_params
        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)

        return update, total_samples, {"loss": avg_loss, "accuracy": acc}

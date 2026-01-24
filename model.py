import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
import copy

# ============================================================================
# ARCHITECTURES NEURONALES
# ============================================================================


class Linear_QNet(nn.Module):
    """Réseau de neurones dense simple (MLP)."""

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_activations(self, x):
        """Retourne les activations pour la visualisation."""
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)

        activations = [x]
        x1 = F.relu(self.linear1(x))
        activations.append(x1)
        x2 = F.relu(self.linear2(x1))
        activations.append(x2)
        x3 = self.linear3(x2)
        activations.append(x3)
        return activations

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        if not file_name.endswith(".pth"):
            file_name += ".pth"
        torch.save(self.state_dict(), file_name)


class ConvNet_QNet(nn.Module):
    """
    CNN amélioré pour le Snake :
    - 3 couches convolutives avec BatchNorm
    - Dropout pour régularisation
    - Architecture conçue pour limiter le sous-apprentissage
    """

    def __init__(self, output_size=3, input_channels=4):
        super().__init__()
        # Input: (4, 24, 32) (C, H, W) - 4 canaux: Corps, Tête, Nourriture, Murs

        # Conv 1: 4 -> 32 canaux. Résolution conservée.
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv 2: 32 -> 64 canaux. Downsample (24x32 -> 12x16).
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv 3: 64 -> 128 canaux. Downsample (12x16 -> 6x8).
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected (Flatten: 128 * 6 * 8 = 6144)
        self.fc1 = nn.Linear(6144, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activations(self, x):
        """Pour visualisation uniquement."""
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        return [x]

    def save(
        self,
        file_name="model.pth",
        n_games=0,
        total_time=0,
        optimizer_state=None,
        epsilon=None,
        record=0,
    ):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        if not file_name.endswith(".pth"):
            file_name += ".pth"

        data = {
            "model_state": self.state_dict(),
            "n_games": n_games,
            "total_time": total_time,
            "optimizer_state": optimizer_state,
            "epsilon": epsilon,
            "record": record,
        }
        try:
            torch.save(data, file_name)
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")

    def load(self, file_name="model.pth", device="cpu"):
        model_folder_path = "./model"
        file_path = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_path):
            try:
                checkpoint = torch.load(file_path, map_location=device)

                # Format dictionnaire complet
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    self.load_state_dict(checkpoint["model_state"])
                    return (
                        checkpoint.get("n_games", 0),
                        checkpoint.get("total_time", 0),
                        checkpoint.get("optimizer_state", None),
                        checkpoint.get("epsilon", None),
                        checkpoint.get("record", 0),
                    )
                else:
                    # Format legacy
                    self.load_state_dict(checkpoint)
                    return 0, 0, None, None, 0
            except Exception as e:
                print(f"Erreur chargement modèle: {e}")
                return None
        return None


# ============================================================================
# ENTRAÎNEUR (Q-LEARNING)
# ============================================================================


class QTrainer:
    """
    Entraîneur DQN avec :
    - Double DQN (Target Network)
    - Huber Loss (Robustesse)
    - Planning de taux d'apprentissage (Scheduler)
    - Gradient Clipping
    """

    def __init__(self, model, lr, gamma, device="cpu", tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.tau = tau  # Facteur Soft Update
        self.device = device

        # Target Network (Évaluation)
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Huber Loss (SmoothL1Loss)
        self.criterion = nn.SmoothL1Loss()

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=100,
            min_lr=1e-6,
            verbose=True,
        )

    def soft_update(self):
        """Mise à jour progressive du Target Network."""
        for target_param, local_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def train_step(self, state, action, reward, next_state, done, weights=None):
        # Conversion Vectorisée
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(
            self.device
        )
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        # Ajout dimension Batch si nécessaire
        if len(state.shape) == 1 or len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
            if weights is not None:
                weights = torch.unsqueeze(weights, 0)

        # 1. Prédiction Q(s, a)
        pred = self.model(state)

        # 2. Target Q(s', a') via Target Network
        with torch.no_grad():
            next_pred = self.target_model(next_state)

        target = pred.clone()
        td_errors = []

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_pred[idx])

            current_val = pred[idx][torch.argmax(action[idx]).item()]

            # TD Error
            td_error = abs(Q_new - current_val).item()
            td_errors.append(td_error)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()

        # Calcul de la perte
        if weights is not None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
            loss_elementwise = loss_fn(target, pred)
            loss = (loss_elementwise.mean(dim=1) * weights).mean()
        else:
            loss = self.criterion(target, pred)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.soft_update()

        return np.array(td_errors)

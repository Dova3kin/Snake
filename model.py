import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
import copy

# ============================================================================
# CERVEAU DU SERPENT (RÉSEAUX DE NEURONES)
# ============================================================================


class ReseauNeurones(nn.Module):
    """
    MLP compact pour Snake.

    Entrée: 9 features (sans pixels)
    Sortie: 3 actions (tout droit, droite, gauche)

    Architecture: 9→128→64→3
    Pas de dropout : inutile avec si peu de paramètres.
    """

    def __init__(self, input_size=9, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 3:
            x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sauvegarder(
        self,
        nom_fichier="modele.pth",
        nb_parties=0,
        temps_total=0,
        etat_optimiseur=None,
        epsilon=None,
        record=0,
    ):
        dossier = "./model"
        if not os.path.exists(dossier):
            os.makedirs(dossier)

        chemin = os.path.join(dossier, nom_fichier)
        if not chemin.endswith(".pth"):
            chemin += ".pth"

        donnees = {
            "version": 2,  # Versioning pour compatibilité future
            "etat_modele": self.state_dict(),
            "nb_parties": nb_parties,
            "temps_total": temps_total,
            "etat_optimiseur": etat_optimiseur,
            "epsilon": epsilon,
            "record": record,
        }
        try:
            torch.save(donnees, chemin)
        except Exception as e:
            print(f"Erreur de sauvegarde : {e}")

    def charger(self, nom_fichier="modele.pth", device="cpu"):
        dossier = "./model"
        chemin = os.path.join(dossier, nom_fichier)

        if os.path.exists(chemin):
            try:
                checkpoint = torch.load(chemin, map_location=device)

                # Format v2 (nouveau)
                if isinstance(checkpoint, dict) and "etat_modele" in checkpoint:
                    self.load_state_dict(checkpoint["etat_modele"])
                    return (
                        checkpoint.get("nb_parties", 0),
                        checkpoint.get("temps_total", 0),
                        checkpoint.get("etat_optimiseur", None),
                        checkpoint.get("epsilon", None),
                        checkpoint.get("record", 0),
                    )
                # Format legacy (anglais)
                elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    self.load_state_dict(checkpoint["model_state"])
                    return (
                        checkpoint.get("n_games", 0),
                        checkpoint.get("total_time", 0),
                        checkpoint.get("optimizer_state", None),
                        checkpoint.get("epsilon", None),
                        checkpoint.get("record", 0),
                    )
                else:
                    # Très vieux format
                    self.load_state_dict(checkpoint)
                    return 0, 0, None, None, 0
            except Exception as e:
                print(f"Erreur chargement : {e}")
                return None
        return None


# ============================================================================
# ENTRAINEUR (COACH)
# ============================================================================


class Entraineur:
    """
    Entraîne le modèle avec DQN.

    Hyperparamètres justifiés:
    - gamma=0.97 : Horizon adapté à Snake (pas trop long)
    - tau=0.005 : Soft update lente pour stabilité
    - SmoothL1Loss : Robuste aux Q-values extrêmes
    """

    def __init__(self, modele, lr, gamma, device="cpu", tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.modele = modele
        self.tau = tau
        self.device = device

        # Target network pour stabiliser l'apprentissage
        self.target_model = copy.deepcopy(modele).to(device)
        self.target_model.eval()

        self.optimiseur = optim.Adam(modele.parameters(), lr=self.lr)
        self.critere = nn.SmoothL1Loss()  # Huber loss

        self.scheduler = ReduceLROnPlateau(
            self.optimiseur,
            mode="max",
            factor=0.5,
            patience=100,
            min_lr=1e-6,
            verbose=True,
        )

    def mise_a_jour_douce(self):
        """Soft update du target network."""
        for target_param, local_param in zip(
            self.target_model.parameters(), self.modele.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def etape_d_apprentissage(self, etat, action, recompense, etat_suiv, finis):
        """
        Une étape d'apprentissage DQN.

        Args:
            etat: états actuels (batch)
            action: actions prises (entiers 0, 1, 2)
            recompense: récompenses reçues
            etat_suiv: états suivants
            finis: booléens indiquant fin de partie
        """
        # Conversion en tenseurs
        etat = torch.tensor(np.array(etat), dtype=torch.float).to(self.device)
        etat_suiv = torch.tensor(np.array(etat_suiv), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        recompense = torch.tensor(np.array(recompense), dtype=torch.float).to(
            self.device
        )
        finis = torch.tensor(np.array(finis), dtype=torch.bool).to(self.device)

        # Aplatir si nécessaire
        if len(etat.shape) == 4:
            etat = etat.view(etat.size(0), -1)
            etat_suiv = etat_suiv.view(etat_suiv.size(0), -1)

        # S'assurer d'avoir dimension batch
        if len(etat.shape) == 1:
            etat = etat.unsqueeze(0)
            etat_suiv = etat_suiv.unsqueeze(0)
            action = action.unsqueeze(0)
            recompense = recompense.unsqueeze(0)
            finis = finis.unsqueeze(0)

        # 1. Q-values actuelles
        pred = self.modele(etat)

        # 2. Q-values cibles (avec target network)
        with torch.no_grad():
            next_pred = self.target_model(etat_suiv)
            max_next_q = next_pred.max(dim=1)[0]

        # 3. Calcul du target Q — vectorisé (pas de boucle Python)
        # Q_bellman = r + gamma * max_Q_next  (si not done)
        # Q_bellman = r                       (si done)
        Q_bellman = recompense + self.gamma * max_next_q * (~finis).float()

        # Mettre à jour uniquement la Q-value de l'action prise
        target = pred.clone()
        target.scatter_(1, action.unsqueeze(1), Q_bellman.unsqueeze(1))

        # 4. Backprop
        self.optimiseur.zero_grad()
        loss = self.critere(pred, target)
        loss.backward()

        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.modele.parameters(), max_norm=1.0)

        self.optimiseur.step()
        self.mise_a_jour_douce()

        return loss.item()

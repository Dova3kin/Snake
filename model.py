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
    Le cerveau du serpent
    """

    def __init__(self, output_size=3, input_channels=4):
        super().__init__()
        # On a 4 "images" en entrée : Corps, Tête, Pomme, Murs
        # Taille 24x32

        # Couche 1: On cherche des formes simples
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Couche 2: On combine les formes (et on réduit la taille de l'image)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Couche 3: Des concepts plus abstraits
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # On met tout à plat pour la décision finale
        # Calcul taille: 128 filtres * 6 hauteur * 8 largeur = 6144
        self.fc1 = nn.Linear(6144, 512)
        self.dropout = nn.Dropout(0.2)  # Pour éviter d'apprendre par cœur
        self.fc2 = nn.Linear(512, 128)
        self.decision = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Aplatissement

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.decision(x)
        return x

    def recuperer_activations(self, x):
        """affichage graphique."""
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        return [x]

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
            print(f"erreur de sauvegarde : {e}")

    def charger(self, nom_fichier="modele.pth", device="cpu"):
        dossier = "./model"
        chemin = os.path.join(dossier, nom_fichier)

        if os.path.exists(chemin):
            try:
                checkpoint = torch.load(chemin, map_location=device)

                # Si c'est notre nouveau format
                if isinstance(checkpoint, dict) and "etat_modele" in checkpoint:
                    self.load_state_dict(checkpoint["etat_modele"])
                    return (
                        checkpoint.get("nb_parties", 0),
                        checkpoint.get("temps_total", 0),
                        checkpoint.get("etat_optimiseur", None),
                        checkpoint.get("epsilon", None),
                        checkpoint.get("record", 0),
                    )
                # Compatibilité avec les anciens modèles (anglais/legacy)
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
                    # Vieux format brut
                    self.load_state_dict(checkpoint)
                    return 0, 0, None, None, 0
            except Exception as e:
                print(f"Erreur chargement : {e}")
                return None
        return None


# ============================================================================
# COACH (ENTRAÎNEMENT)
# ============================================================================


class Entraineur:
    """
    C'est lui qui apprend au modèle.
    """

    def __init__(self, modele, lr, gamma, device="cpu", tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.modele = modele
        self.tau = tau  # Vitesse de mise à jour du modèle cible
        self.device = device

        # On crée une copie du modèle pour stabiliser l'apprentissage
        self.target_model = copy.deepcopy(modele).to(device)
        self.target_model.eval()

        self.optimiseur = optim.Adam(modele.parameters(), lr=self.lr)

        # On utilise Huber Loss parce que c'est moins sensible aux gros bugs de valeurs
        self.critere = nn.SmoothL1Loss()

        self.scheduler = ReduceLROnPlateau(
            self.optimiseur,
            mode="max",
            factor=0.5,
            patience=100,
            min_lr=1e-6,
            verbose=True,
        )

    def mise_a_jour_douce(self):
        """Mise à jour progressive du Target Network."""
        for target_param, local_param in zip(
            self.target_model.parameters(), self.modele.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def etape_d_apprentissage(
        self, etat, action, recompense, etat_suiv, finis, weights=None
    ):
        # On transforme tout en tenseurs PyTorch
        etat = torch.tensor(np.array(etat), dtype=torch.float).to(self.device)
        etat_suiv = torch.tensor(np.array(etat_suiv), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        recompense = torch.tensor(np.array(recompense), dtype=torch.float).to(
            self.device
        )

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        # On s'assure qu'on a bien une dimension "batch"
        if len(etat.shape) == 1 or len(etat.shape) == 3:
            etat = torch.unsqueeze(etat, 0)
            etat_suiv = torch.unsqueeze(etat_suiv, 0)
            action = torch.unsqueeze(action, 0)
            recompense = torch.unsqueeze(recompense, 0)
            finis = (finis,)
            if weights is not None:
                weights = torch.unsqueeze(weights, 0)

        # 1. Qu'est-ce que le modèle pense ?
        pred = self.modele(etat)

        # 2. Qu'est-ce qu'il devrait penser ? (Target)
        with torch.no_grad():
            next_pred = self.target_model(etat_suiv)

        target = pred.clone()
        erreurs_td = []

        for idx in range(len(finis)):
            Q_nouveau = recompense[idx]
            if not finis[idx]:
                Q_nouveau = recompense[idx] + self.gamma * torch.max(next_pred[idx])

            valeur_actuelle = pred[idx][torch.argmax(action[idx]).item()]

            # Calcul de l'erreur (surprise)
            erreur = abs(Q_nouveau - valeur_actuelle).item()
            erreurs_td.append(erreur)

            target[idx][torch.argmax(action[idx]).item()] = Q_nouveau

        self.optimiseur.zero_grad()

        # Calcul de la perte (Loss)
        if weights is not None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
            loss_element = loss_fn(target, pred)
            loss = (loss_element.mean(dim=1) * weights).mean()
        else:
            loss = self.critere(target, pred)

        loss.backward()

        # On empêche les gradients d'exploser
        torch.nn.utils.clip_grad_norm_(self.modele.parameters(), max_norm=1.0)

        self.optimiseur.step()
        self.mise_a_jour_douce()

        return np.array(erreurs_td)

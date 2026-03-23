"""
Configuration centralisée du projet Snake IA.

Toutes les constantes et hyperparamètres sont regroupés ici
pour faciliter la configuration et les expérimentations.
"""
from dataclasses import dataclass, field


@dataclass
class ConfigEntrainement:
    """Hyperparamètres d'entraînement DQN."""

    # === ENVIRONNEMENT ===
    nb_environnements: int = 1000       # Jeux parallèles
    largeur: int = 640                  # Largeur de la grille (pixels)
    hauteur: int = 480                  # Hauteur de la grille (pixels)
    taille_bloc: int = 20               # Taille d'une case (pixels)

    # === MÉMOIRE ===
    memoire_max: int = 200_000          # Taille du replay buffer
    taille_batch: int = 256             # Taille du mini-batch

    # === RÉSEAU ===
    input_size: int = 9                 # 7 features existantes + 2 position tête normalisée
                                        # Suppression des 3072 pixels MLP-inutilisables
    output_size: int = 3               # 3 actions: tout droit, droite, gauche
    taille_couche_1: int = 128          # Réduit (était 256, inutile avec 9 inputs)
    taille_couche_2: int = 64           # Réduit (était 128)

    # === OPTIMISEUR ===
    taux_apprentissage: float = 0.0003  # Adam LR
    gamma: float = 0.97                 # Facteur de discount
    tau: float = 0.005                  # Soft update target network

    # === EXPLORATION ===
    epsilon_depart: float = 1.0
    epsilon_fin: float = 0.05
    epsilon_frames: int = 1_000_000     # Steps de simulation pour atteindre epsilon_fin
                                    # Indépendant du nombre d'envs parallèles
    taux_exploration_aleatoire: float = 0.02  # Exploration pure

    # === ENTRAÎNEMENT ===
    freq_entrainement: int = 8          # Entraîner toutes les N frames
    eval_intervalle: int = 5_000        # Éval toutes les N parties
    famine_base: int = 100              # Base du timeout de famine
    famine_par_case: int = 3            # Steps supplémentaires par case de longueur
                                    # timeout = famine_base + longueur * famine_par_case
                                    # ex: longueur 3 -> 109, longueur 30 -> 190

    # === SCHEDULER LR ===
    lr_scheduler_patience: int = 100
    lr_scheduler_factor: float = 0.5
    lr_min: float = 1e-6


@dataclass
class ConfigAffichage:
    """Paramètres de l'interface graphique."""

    largeur_fenetre: int = 1280
    hauteur_fenetre: int = 720
    largeur_menu: int = 250
    hauteur_barre_bas: int = 40
    dossier_screenshots: str = "screenshots"
    intervalle_screenshot_defaut: int = 60  # secondes


# Instance globale (import facile)
CONFIG = ConfigEntrainement()
CONFIG_UI = ConfigAffichage()

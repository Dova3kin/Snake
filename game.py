"""
Jeu Snake pour le projet d'IA
"""

import pygame
import numpy as np
from enum import Enum
from collections import namedtuple

# On initialise pygame
pygame.init()

# Configuration
TAILLE_BLOC = 20
FPS = 144
VITESSE = 20

# Les couleurs
BLANC = (255, 255, 255)
NOIR = (20, 20, 30)
ROUGE = (220, 50, 50)
VERT = (50, 200, 50)
BLEU = (50, 100, 200)
GRIS = (40, 40, 50)

# Polices
FONT = pygame.font.SysFont("arial", 25)
FONT_GRAND = pygame.font.SysFont("arial", 50)


class Direction(Enum):
    DROITE = 1
    GAUCHE = 2
    HAUT = 3
    BAS = 4


Point = namedtuple("Point", "x, y")


class JeuVectorise:
    def __init__(self, n_envs=256, largeur=640, hauteur=480, taille_bloc=TAILLE_BLOC):
        self.n_envs = n_envs
        self.taille_bloc = taille_bloc
        self.l = largeur
        self.h = hauteur
        self.grille_l = largeur // taille_bloc
        self.grille_h = hauteur // taille_bloc
        self.max_len = self.grille_l * self.grille_h // 2

        # Vectors de mouvement: 0=Droite, 1=Bas, 2=Gauche, 3=Haut
        self.vec_mouvements = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32
        )

        self.tetes = np.zeros((n_envs, 2), dtype=np.int32)
        self.pommes = np.zeros((n_envs, 2), dtype=np.int32)
        self.directions = np.zeros(n_envs, dtype=np.int32)
        self.scores = np.zeros(n_envs, dtype=np.int32)
        self.finis = np.zeros(n_envs, dtype=bool)
        self.etapes_depuis_pomme = np.zeros(n_envs, dtype=np.int32)

        # Corps des serpents
        self.corps = np.full((n_envs, self.max_len, 2), -1, dtype=np.int32)
        self.longueurs = np.full(n_envs, 3, dtype=np.int32)

        self.buffer_etat = np.zeros(
            (n_envs, 4, self.grille_h, self.grille_l), dtype=np.float32
        )

        # Canal des murs (pré-calculé car ça bouge pas)
        self.canal_murs = np.zeros((self.grille_h, self.grille_l), dtype=np.float32)
        self.canal_murs[0, :] = 1.0  # Haut
        self.canal_murs[-1, :] = 1.0  # Bas
        self.canal_murs[:, 0] = 1.0  # Gauche
        self.canal_murs[:, -1] = 1.0  # Droite

        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = np.arange(self.n_envs)

        n = len(indices)
        if n == 0:
            return

        # On remet au centre
        cx, cy = self.grille_l // 2, self.grille_h // 2
        self.tetes[indices] = [cx, cy]
        self.directions[indices] = 0  # Droite

        # On remet le corps
        self.corps[indices, :, :] = -1
        self.corps[indices, 0] = [cx, cy]
        self.corps[indices, 1] = [cx - 1, cy]
        self.corps[indices, 2] = [cx - 2, cy]
        self.longueurs[indices] = 3

        self.scores[indices] = 0
        self.finis[indices] = False
        self.etapes_depuis_pomme[indices] = 0

        self._spawn_pommes(indices)

    def _spawn_pommes(self, indices):
        n = len(indices)
        xs = np.random.randint(0, self.grille_l - 1, size=n)
        ys = np.random.randint(0, self.grille_h - 1, size=n)
        self.pommes[indices] = np.stack([xs, ys], axis=1)

    def step(self, actions):
        """Calculer la prochaine étape pour tous les environnements."""
        # Calcul distance AVANT (pour le reward shaping)
        dist_avant = np.abs(self.tetes[:, 0] - self.pommes[:, 0]) + np.abs(
            self.tetes[:, 1] - self.pommes[:, 1]
        )

        # Changement de direction
        # actions: 0=tout droit, 1=droite, 2=gauche
        shifts = np.array([0, 1, -1])[actions]
        self.directions = (self.directions + shifts) % 4

        # Bouger la tête
        mouvements = self.vec_mouvements[self.directions]
        nouvelles_tetes = self.tetes + mouvements

        # Collision Mur ?
        mur_touche = (
            (nouvelles_tetes[:, 0] < 0)
            | (nouvelles_tetes[:, 0] >= self.grille_l)
            | (nouvelles_tetes[:, 1] < 0)
            | (nouvelles_tetes[:, 1] >= self.grille_h)
        )

        # Collision Corps ?
        corps_touche = np.zeros(self.n_envs, dtype=bool)
        for i in range(self.n_envs):
            tete = nouvelles_tetes[i]
            longueur = self.longueurs[i]
            # On vérifie si la tête touche une partie du corps (sauf la fin)
            partie_corps = self.corps[i, : longueur - 1]
            if np.any(np.all(tete == partie_corps, axis=1)):
                corps_touche[i] = True

        # Miam ?
        pomme_mangee = np.all(nouvelles_tetes == self.pommes, axis=1)

        # Trop long sans manger ?
        self.etapes_depuis_pomme += 1
        self.etapes_depuis_pomme[pomme_mangee] = 0
        famine = self.etapes_depuis_pomme > 150

        # Calcul Reward
        nouvelles_tetes_safe = np.clip(
            nouvelles_tetes, [0, 0], [self.grille_l - 1, self.grille_h - 1]
        )
        dist_apres = np.abs(nouvelles_tetes_safe[:, 0] - self.pommes[:, 0]) + np.abs(
            nouvelles_tetes_safe[:, 1] - self.pommes[:, 1]
        )

        # On encourage si on se rapproche (shaping)
        reward_distance = 0.3 * (dist_avant - dist_apres).astype(np.float32)

        recompenses = np.zeros(self.n_envs, dtype=np.float32)
        recompenses[pomme_mangee] = 10.0
        recompenses += reward_distance
        recompenses += -0.01  # Petite pénalité de temps

        self.finis = mur_touche | corps_touche | famine
        recompenses[self.finis] = -20.0

        # Mise à jour physiques
        self.corps[:, 1:] = self.corps[:, :-1]
        self.corps[:, 0] = nouvelles_tetes

        self.longueurs[pomme_mangee] += 1
        self.scores[pomme_mangee] += 1
        self._spawn_pommes(np.where(pomme_mangee)[0])
        self.tetes = nouvelles_tetes

        # Auto-Reset des morts
        scores_finaux = self.scores.copy()
        finis_finaux = self.finis.copy()

        if np.any(self.finis):
            self.reset(np.where(self.finis)[0])

        return self.recuperer_etats(), recompenses, finis_finaux, scores_finaux

    def recuperer_etats(self):
        """Fabrique l'image (tenseur) pour l'IA."""
        self.buffer_etat.fill(0)
        ids = np.arange(self.n_envs)

        # 3. Murs
        self.buffer_etat[:, 3, :, :] = self.canal_murs

        # 2. Pommes
        px, py = self.pommes[:, 0], self.pommes[:, 1]
        self.buffer_etat[ids, 2, py, px] = 1.0

        # 1. Têtes + Direction
        hx, hy = self.tetes[:, 0], self.tetes[:, 1]
        hx = np.clip(hx, 0, self.grille_l - 1)
        hy = np.clip(hy, 0, self.grille_h - 1)
        # On encode la direction (1 à 4 divisé par 5 pour normaliser)
        val_dir = (self.directions + 1) * 0.2
        self.buffer_etat[ids, 1, hy, hx] = val_dir

        # 0. Corps (avec dégradé pour donner info de l'ordre)
        for i in range(self.n_envs):
            longueur = self.longueurs[i]
            c = self.corps[i, :longueur]

            indices = np.arange(longueur, dtype=np.float32)
            valeurs = 1.0 - (indices / longueur)

            cx = np.clip(c[:, 0], 0, self.grille_l - 1)
            cy = np.clip(c[:, 1], 0, self.grille_h - 1)

            self.buffer_etat[i, 0, cy, cx] = valeurs

        return self.buffer_etat.copy()

    def actions_gloutonnes(self):
        """
        Une petite IA heuristique (pas de réseau de neurones) qui essaie juste de pas mourir
        et d'aller vers la pomme. Sert pour guider l'IA au début.
        """
        masque_sur = np.zeros((self.n_envs, 3), dtype=bool)
        distances = np.full((self.n_envs, 3), np.inf)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs_possibles = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs_possibles]
            prochaines_tetes = self.tetes + vecs

            # Murs
            mur = (
                (prochaines_tetes[:, 0] < 0)
                | (prochaines_tetes[:, 0] >= self.grille_l)
                | (prochaines_tetes[:, 1] < 0)
                | (prochaines_tetes[:, 1] >= self.grille_h)
            )

            # Corps
            corps_hit = np.zeros(self.n_envs, dtype=bool)
            for i in range(self.n_envs):
                tete = prochaines_tetes[i]
                long = self.longueurs[i]
                partie = self.corps[i, : long - 1]
                if np.any(np.all(tete == partie, axis=1)):
                    corps_hit[i] = True

            est_sur = ~mur & ~corps_hit
            masque_sur[:, action] = est_sur

            # Distance
            dists = np.abs(prochaines_tetes[:, 0] - self.pommes[:, 0]) + np.abs(
                prochaines_tetes[:, 1] - self.pommes[:, 1]
            )
            distances[np.where(est_sur), action] = dists[np.where(est_sur)]

        meilleures_actions = np.argmin(distances, axis=1)

        # Si l'action choisie nous tue, on en prend une autre au hasard qui tue pas
        choix_ok = masque_sur[np.arange(self.n_envs), meilleures_actions]
        indices_dangereux = np.where(~choix_ok)[0]

        if len(indices_dangereux) > 0:
            for i in indices_dangereux:
                actions_sures = np.where(masque_sur[i])[0]
                if len(actions_sures) > 0:
                    meilleures_actions[i] = np.random.choice(actions_sures)
                else:
                    meilleures_actions[i] = np.random.randint(0, 3)

        return meilleures_actions

    def actions_aleatoires_sures(self):
        """Retourne des actions au hasard MAIS qui ne tuent pas (si possible)"""
        masque_sur = np.zeros((self.n_envs, 3), dtype=bool)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]
            tetes = self.tetes + vecs

            mur = (
                (tetes[:, 0] < 0)
                | (tetes[:, 0] >= self.grille_l)
                | (tetes[:, 1] < 0)
                | (tetes[:, 1] >= self.grille_h)
            )

            corps_hit = np.zeros(self.n_envs, dtype=bool)
            for i in range(self.n_envs):
                t = tetes[i]
                long = self.longueurs[i]
                p = self.corps[i, : long - 1]
                if np.any(np.all(t == p, axis=1)):
                    corps_hit[i] = True

            masque_sur[:, action] = ~mur & ~corps_hit

        actions_rand = np.zeros(self.n_envs, dtype=np.int32)
        for i in range(self.n_envs):
            ok = np.where(masque_sur[i])[0]
            if len(ok) > 0:
                actions_rand[i] = np.random.choice(ok)
            else:
                actions_rand[i] = np.random.randint(0, 3)

        return actions_rand

"""
Jeu Snake vectorisé pour entraînement IA.

Performances:
- Toutes les opérations de collision utilisent NumPy vectorisé
  (broadcast 3D : n_envs × max_len × 2) → pas de boucles Python
- Gain estimé: 5-20x sur le throughput vs boucles naïves
"""

import pygame
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()

# Configuration
TAILLE_BLOC = 20

# Couleurs
BLANC = (255, 255, 255)
NOIR = (20, 20, 30)
ROUGE = (220, 50, 50)
VERT = (50, 200, 50)
BLEU = (50, 100, 200)
GRIS = (40, 40, 50)


class Direction(Enum):
    DROITE = 1
    GAUCHE = 2
    HAUT = 3
    BAS = 4


Point = namedtuple("Point", "x, y")


class JeuVectorise:
    """
    Environnement Snake vectorisé (N jeux en parallèle).

    État retourné: tenseur (n_envs, features) où features inclut:
    - Pixels aplatis (4 canaux × 24 × 32 = 3072)
    - Features calculées (7 valeurs):
        * Distance pomme normalisée
        * Direction pomme X (-1, 0, 1)
        * Direction pomme Y (-1, 0, 1)
        * Danger avant (0 ou 1)
        * Danger gauche (0 ou 1)
        * Danger droite (0 ou 1)
        * Faim normalisée (0 à 1)
    """

    def __init__(self, n_envs=256, largeur=640, hauteur=480, taille_bloc=TAILLE_BLOC):
        self.n_envs = n_envs
        self.taille_bloc = taille_bloc
        self.l = largeur
        self.h = hauteur
        self.grille_l = largeur // taille_bloc
        self.grille_h = hauteur // taille_bloc
        self.max_len = self.grille_l * self.grille_h // 2

        # Vecteurs de mouvement: 0=Droite, 1=Bas, 2=Gauche, 3=Haut
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

        # Buffer pour les états (pixels)
        self.buffer_etat = np.zeros(
            (n_envs, 4, self.grille_h, self.grille_l), dtype=np.float32
        )

        # Canal des murs (pré-calculé)
        self.canal_murs = np.zeros((self.grille_h, self.grille_l), dtype=np.float32)
        self.canal_murs[0, :] = 1.0
        self.canal_murs[-1, :] = 1.0
        self.canal_murs[:, 0] = 1.0
        self.canal_murs[:, -1] = 1.0

        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = np.arange(self.n_envs)

        n = len(indices)
        if n == 0:
            return

        cx, cy = self.grille_l // 2, self.grille_h // 2
        self.tetes[indices] = [cx, cy]
        self.directions[indices] = 0  # Droite

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

    def _collision_corps_vectorisee(self, tetes_candidates: np.ndarray) -> np.ndarray:
        """
        Détecte les collisions corps pour toutes les envs en une seule opération.

        Args:
            tetes_candidates: (n_envs, 2) positions à tester

        Returns:
            (n_envs,) booléen True si collision corps

        Méthode:
            On étend tetes_candidates → (n_envs, 1, 2)
            On étend self.corps → (n_envs, max_len, 2)
            Le broadcast compare toutes les positions simultanément.
            On masque les positions invalides (-1) pour éviter les faux positifs.
        """
        # (n_envs, 1, 2) vs (n_envs, max_len, 2)
        tetes_exp = tetes_candidates[:, np.newaxis, :]  # (n_envs, 1, 2)
        match = np.all(tetes_exp == self.corps, axis=2)  # (n_envs, max_len)

        # Masquer les cases vides (valeur -1 dans le corps)
        valide = self.corps[:, :, 0] >= 0  # (n_envs, max_len)

        # Créer un masque longueur: ne tester que les n-1 premiers segments
        # (la queue va disparaître au prochain move) - Totalement vectorisé
        longueur_masque = np.arange(self.max_len)[np.newaxis, :] < (self.longueurs[:, np.newaxis] - 1)

        return np.any(match & valide & longueur_masque, axis=1)

    def _calculer_dangers(self) -> np.ndarray:
        """Calcule si chaque direction (avant, gauche, droite) est dangereuse."""
        dangers = np.zeros((self.n_envs, 3), dtype=np.float32)

        for action in [0, 1, 2]:  # tout droit, droite, gauche
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]
            tetes = self.tetes + vecs

            # Murs (vectorisé)
            mur = (
                (tetes[:, 0] < 0)
                | (tetes[:, 0] >= self.grille_l)
                | (tetes[:, 1] < 0)
                | (tetes[:, 1] >= self.grille_h)
            )

            # Corps (vectorisé via broadcast 3D)
            corps_hit = self._collision_corps_vectorisee(tetes)

            dangers[:, action] = (mur | corps_hit).astype(np.float32)

        return dangers

    def step(self, actions: np.ndarray):
        """Une étape de simulation pour tous les environnements."""
        # Distance AVANT
        dist_avant = np.abs(self.tetes[:, 0] - self.pommes[:, 0]) + np.abs(
            self.tetes[:, 1] - self.pommes[:, 1]
        )

        # Changement de direction
        shifts = np.array([0, 1, -1])[actions]
        self.directions = (self.directions + shifts) % 4

        # Bouger la tête
        mouvements = self.vec_mouvements[self.directions]
        nouvelles_tetes = self.tetes + mouvements

        # Collision Mur (vectorisé)
        mur_touche = (
            (nouvelles_tetes[:, 0] < 0)
            | (nouvelles_tetes[:, 0] >= self.grille_l)
            | (nouvelles_tetes[:, 1] < 0)
            | (nouvelles_tetes[:, 1] >= self.grille_h)
        )

        # Collision Corps — vectorisée via broadcast 3D
        corps_touche = self._collision_corps_vectorisee(nouvelles_tetes)

        # Pomme mangée
        pomme_mangee = np.all(nouvelles_tetes == self.pommes, axis=1)

        # Famine
        self.etapes_depuis_pomme += 1
        self.etapes_depuis_pomme[pomme_mangee] = 0
        # Famine adaptative : timeout croît avec la longueur du serpent
        # Un long serpent a besoin de plus de temps pour contourner son propre corps
        famine_timeout = 100 + self.longueurs * 3
        famine = self.etapes_depuis_pomme > famine_timeout

        # Distance APRÈS
        nouvelles_tetes_safe = np.clip(
            nouvelles_tetes, [0, 0], [self.grille_l - 1, self.grille_h - 1]
        )
        dist_apres = np.abs(nouvelles_tetes_safe[:, 0] - self.pommes[:, 0]) + np.abs(
            nouvelles_tetes_safe[:, 1] - self.pommes[:, 1]
        )

        # === REWARDS NORMALISÉS ===
        # Tous du même ordre de grandeur pour apprentissage stable
        recompenses = np.full(self.n_envs, -0.001, dtype=np.float32)  # Pénalité temps

        # Bonus approche (0.01 par case rapprochée)
        recompenses += 0.01 * (dist_avant - dist_apres).astype(np.float32)

        # Pomme mangée = +1.0
        recompenses[pomme_mangee] = 1.0

        # Mort = -1.0
        self.finis = mur_touche | corps_touche | famine
        recompenses[self.finis] = -1.0

        # Mise à jour physique
        self.corps[:, 1:] = self.corps[:, :-1]
        self.corps[:, 0] = nouvelles_tetes

        self.longueurs[pomme_mangee] += 1
        self.scores[pomme_mangee] += 1
        self._spawn_pommes(np.where(pomme_mangee)[0])
        self.tetes = nouvelles_tetes

        # Auto-Reset
        scores_finaux = self.scores.copy()
        finis_finaux = self.finis.copy()

        if np.any(self.finis):
            self.reset(np.where(self.finis)[0])

        return self.recuperer_etats(), recompenses, finis_finaux, scores_finaux

    def recuperer_etats(self):
        """
        Retourne l'état augmenté: pixels aplatis + features calculées.
        Shape: (n_envs, 3079) = 3072 pixels + 7 features
        """
        # 1. Construire les canaux de pixels
        self.buffer_etat.fill(0)
        ids = np.arange(self.n_envs)

        # Murs
        self.buffer_etat[:, 3, :, :] = self.canal_murs

        # Pommes
        px, py = self.pommes[:, 0], self.pommes[:, 1]
        self.buffer_etat[ids, 2, py, px] = 1.0

        # Têtes + Direction
        hx, hy = self.tetes[:, 0], self.tetes[:, 1]
        hx = np.clip(hx, 0, self.grille_l - 1)
        hy = np.clip(hy, 0, self.grille_h - 1)
        val_dir = (self.directions + 1) * 0.2
        self.buffer_etat[ids, 1, hy, hx] = val_dir

        # Corps (dégradé) — vectorisé avec indexation avancée
        max_len_utile = int(self.longueurs.max())
        indices_globaux = np.arange(self.n_envs)
        for seg in range(max_len_utile):
            # Masque: seulement les envs dont le corps est assez long
            masque_actif = seg < self.longueurs
            if not np.any(masque_actif):
                break
            envs_actifs = indices_globaux[masque_actif]
            # Positions du segment 'seg' pour ces envs
            corps_seg = self.corps[envs_actifs, seg]  # (k, 2)
            cx = np.clip(corps_seg[:, 0], 0, self.grille_l - 1)
            cy = np.clip(corps_seg[:, 1], 0, self.grille_h - 1)
            valeurs = np.asarray(
                1.0 - (seg / self.longueurs[envs_actifs]), dtype=np.float32
            )
            self.buffer_etat[envs_actifs, 0, cy, cx] = valeurs

        # 2. Aplatir les pixels
        pixels_flat = self.buffer_etat.reshape(self.n_envs, -1)

        # 3. Calculer les features augmentées
        # Distance pomme normalisée
        dist_pomme = (
            np.abs(self.tetes[:, 0] - self.pommes[:, 0])
            + np.abs(self.tetes[:, 1] - self.pommes[:, 1])
        ) / (self.grille_l + self.grille_h)

        # Direction pomme (-1, 0, 1)
        dir_pomme_x = np.sign(self.pommes[:, 0] - self.tetes[:, 0])
        dir_pomme_y = np.sign(self.pommes[:, 1] - self.tetes[:, 1])

        # Dangers (avant, droite, gauche)
        dangers = self._calculer_dangers()

        # Faim normalisée
        faim = self.etapes_depuis_pomme / 150.0

        # 4. Combiner tout
        features = np.stack(
            [
                dist_pomme,
                dir_pomme_x,
                dir_pomme_y,
                dangers[:, 0],  # danger avant
                dangers[:, 1],  # danger droite
                dangers[:, 2],  # danger gauche
                faim,
            ],
            axis=1,
        )

        return np.concatenate([pixels_flat, features], axis=1)

    def actions_gloutonnes(self) -> np.ndarray:
        """
        Heuristique: aller vers la pomme en évitant les obstacles.
        Sert de "professeur" pour l'apprentissage par imitation.
        Vectorisé: pas de boucles Python sur n_envs.
        """
        masque_sur = np.zeros((self.n_envs, 3), dtype=bool)
        distances = np.full((self.n_envs, 3), np.inf)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs_possibles = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs_possibles]
            prochaines_tetes = self.tetes + vecs

            # Murs (vectorisé)
            mur = (
                (prochaines_tetes[:, 0] < 0)
                | (prochaines_tetes[:, 0] >= self.grille_l)
                | (prochaines_tetes[:, 1] < 0)
                | (prochaines_tetes[:, 1] >= self.grille_h)
            )

            # Corps (vectorisé via broadcast 3D)
            corps_hit = self._collision_corps_vectorisee(prochaines_tetes)

            est_sur = ~mur & ~corps_hit
            masque_sur[:, action] = est_sur

            dists = np.abs(prochaines_tetes[:, 0] - self.pommes[:, 0]) + np.abs(
                prochaines_tetes[:, 1] - self.pommes[:, 1]
            )
            distances[np.where(est_sur), action] = dists[np.where(est_sur)]

        meilleures_actions = np.argmin(distances, axis=1)

        # Fallback totalement vectorisé: corriger les choix dangereux
        choix_ok = masque_sur[np.arange(self.n_envs), meilleures_actions]
        indices_dangereux = np.where(~choix_ok)[0]

        if len(indices_dangereux) > 0:
            m_dangereux = masque_sur[indices_dangereux]  # (K, 3) boolean
            r = np.random.rand(*m_dangereux.shape)
            r[~m_dangereux] = -1.0  # Ignorer les directions non sûres
            
            # Si aucune direction n'est sûre, prendre une action au hasard (r contiendra que des -1)
            actions_remplacement = np.argmax(r, axis=1)
            aucune_sure = np.all(~m_dangereux, axis=1)
            if np.any(aucune_sure):
                actions_remplacement[aucune_sure] = np.random.randint(0, 3, size=np.sum(aucune_sure))
                
            meilleures_actions[indices_dangereux] = actions_remplacement

        return meilleures_actions

    def actions_aleatoires_sures(self) -> np.ndarray:
        """Actions aléatoires mais qui ne tuent pas. Vectorisé."""
        masque_sur = np.zeros((self.n_envs, 3), dtype=bool)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]
            tetes = self.tetes + vecs

            # Murs (vectorisé)
            mur = (
                (tetes[:, 0] < 0)
                | (tetes[:, 0] >= self.grille_l)
                | (tetes[:, 1] < 0)
                | (tetes[:, 1] >= self.grille_h)
            )

            # Corps (vectorisé via broadcast 3D)
            corps_hit = self._collision_corps_vectorisee(tetes)
            masque_sur[:, action] = ~mur & ~corps_hit

        # Sélection aléatoire parmi les actions sûres - Totalement vectorisé
        r = np.random.rand(self.n_envs, 3)
        r[~masque_sur] = -1.0
        actions_rand = np.argmax(r, axis=1)
        
        aucune_sure = np.all(~masque_sur, axis=1)
        if np.any(aucune_sure):
            actions_rand[aucune_sure] = np.random.randint(0, 3, size=np.sum(aucune_sure))

        return actions_rand

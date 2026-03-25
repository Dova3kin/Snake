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

    def _calculer_flood_fill(self) -> np.ndarray:
        """
        Pour chaque direction (avant/droite/gauche), compte les cases
        accessibles via BFS complet (sans limite arbitraire de profondeur).
        Normalisé par grille_l * grille_h pour une valeur proportionnelle réelle.

        Corrections vs version naïve :
        - Pas de max_depth : le BFS explore tout l'espace accessible
        - La queue (corps[i, longueur-1]) est exclue : elle va disparaître
        - Normalisation par taille de grille, pas par une constante arbitraire
        """
        from collections import deque as dq

        total_cases = self.grille_l * self.grille_h
        resultats = np.zeros((self.n_envs, 3), dtype=np.float32)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]
            starts = self.tetes + vecs

            for i in range(self.n_envs):
                sx, sy = starts[i]
                if sx < 0 or sx >= self.grille_l or sy < 0 or sy >= self.grille_h:
                    continue

                # Grille des obstacles : tous les segments SAUF la queue
                # La queue disparaît au prochain move → ne pas la compter comme mur
                longueur = self.longueurs[i]
                grille = np.zeros((self.grille_l, self.grille_h), dtype=bool)
                segments = self.corps[i, :longueur - 1]   # longueur-1 exclut la queue
                masque_valide = segments[:, 0] >= 0
                segments_valides = segments[masque_valide]
                if len(segments_valides) > 0:
                    grille[segments_valides[:, 0], segments_valides[:, 1]] = True

                # BFS complet — exploration totale de l'espace accessible
                visited = np.zeros((self.grille_l, self.grille_h), dtype=bool)
                visited[sx, sy] = True
                queue = dq([(sx, sy)])
                count = 0

                while queue:
                    x, y = queue.popleft()
                    if grille[x, y]:
                        continue
                    count += 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grille_l and 0 <= ny < self.grille_h
                                and not visited[nx, ny] and not grille[nx, ny]):
                            visited[nx, ny] = True
                            queue.append((nx, ny))

                resultats[i, action] = count / total_cases

        return resultats

    def _direction_one_hot(self) -> np.ndarray:
        """One-hot de la direction actuelle. Shape: (n_envs, 4)."""
        one_hot = np.zeros((self.n_envs, 4), dtype=np.float32)
        one_hot[np.arange(self.n_envs), self.directions] = 1.0
        return one_hot

    def _calculer_distances_murs(self) -> np.ndarray:
        """Distance au mur dans chaque direction (avant/droite/gauche), normalisée."""
        distances = np.zeros((self.n_envs, 3), dtype=np.float32)
        max_dist = max(self.grille_l, self.grille_h)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]

            dx, dy = vecs[:, 0], vecs[:, 1]
            dist_x = np.where(dx > 0, self.grille_l - 1 - self.tetes[:, 0],
                     np.where(dx < 0, self.tetes[:, 0],
                     max_dist)).astype(np.float32)
            dist_y = np.where(dy > 0, self.grille_h - 1 - self.tetes[:, 1],
                     np.where(dy < 0, self.tetes[:, 1],
                     max_dist)).astype(np.float32)
            dist = np.minimum(dist_x, dist_y)
            distances[:, action] = dist / max_dist

        return distances

    def _direction_queue(self) -> np.ndarray:
        """Direction relative vers le bout de la queue. Shape: (n_envs, 2)."""
        idx_queue = self.longueurs - 1  # (n_envs,)
        queues = self.corps[np.arange(self.n_envs), idx_queue]  # (n_envs, 2)

        dir_x = np.sign(queues[:, 0] - self.tetes[:, 0]).astype(np.float32)
        dir_y = np.sign(queues[:, 1] - self.tetes[:, 1]).astype(np.float32)
        return np.stack([dir_x, dir_y], axis=1)

    def _calculer_dangers_profonds(self) -> np.ndarray:
        """Danger à 2 cases dans chaque direction (avant/droite/gauche)."""
        dangers = np.zeros((self.n_envs, 3), dtype=np.float32)

        for action in [0, 1, 2]:
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            dirs = (self.directions + shift) % 4
            vecs = self.vec_mouvements[dirs]
            tetes_2 = self.tetes + vecs * 2

            mur = (
                (tetes_2[:, 0] < 0) | (tetes_2[:, 0] >= self.grille_l)
                | (tetes_2[:, 1] < 0) | (tetes_2[:, 1] >= self.grille_h)
            )
            corps_hit = self._collision_corps_vectorisee(tetes_2)
            dangers[:, action] = (mur | corps_hit).astype(np.float32)

        return dangers

    def _calculer_densite_corps(self, rayon=3) -> np.ndarray:
        """Ratio de cases occupées par le corps dans un carré autour de la tête."""
        cote = 2 * rayon + 1
        total_cases = cote * cote

        # Masque des segments valides par env : (n_envs, max_len)
        indices = np.arange(self.max_len)[np.newaxis, :]  # (1, max_len)
        masque_valide = indices < self.longueurs[:, np.newaxis]  # (n_envs, max_len)

        # Distance de chaque segment à la tête : (n_envs, max_len, 2)
        tetes_exp = self.tetes[:, np.newaxis, :]  # (n_envs, 1, 2)
        diff = np.abs(self.corps - tetes_exp)  # (n_envs, max_len, 2)

        # Segment dans le carré de rayon
        dans_rayon = (diff[:, :, 0] <= rayon) & (diff[:, :, 1] <= rayon)  # (n_envs, max_len)

        # Compter les segments valides ET dans le rayon
        count = np.sum(masque_valide & dans_rayon, axis=1).astype(np.float32)

        return count / total_cases

    def recuperer_etats(self):
        """
        Retourne 26 features par environnement.
        Shape: (n_envs, 26)
        """
        # === Existantes (9) ===
        dist_pomme = (
            np.abs(self.tetes[:, 0] - self.pommes[:, 0])
            + np.abs(self.tetes[:, 1] - self.pommes[:, 1])
        ) / (self.grille_l + self.grille_h)
        dir_pomme_x = np.sign(self.pommes[:, 0] - self.tetes[:, 0]).astype(np.float32)
        dir_pomme_y = np.sign(self.pommes[:, 1] - self.tetes[:, 1]).astype(np.float32)
        dangers = self._calculer_dangers()
        famine_timeout = 100 + self.longueurs * 3
        faim = (self.etapes_depuis_pomme / famine_timeout.astype(np.float32)).astype(np.float32)
        pos_x = self.tetes[:, 0] / self.grille_l
        pos_y = self.tetes[:, 1] / self.grille_h

        # === Tier S (7) ===
        floods = self._calculer_flood_fill()
        dir_one_hot = self._direction_one_hot()

        # === Tier A (6) ===
        longueur_norm = (self.longueurs / self.max_len).astype(np.float32)
        dist_murs = self._calculer_distances_murs()
        dir_queue = self._direction_queue()

        # === Tier B (4) ===
        dangers_profonds = self._calculer_dangers_profonds()
        densite = self._calculer_densite_corps()

        return np.stack([
            dist_pomme, dir_pomme_x, dir_pomme_y,
            dangers[:, 0], dangers[:, 1], dangers[:, 2],
            faim, pos_x, pos_y,
            # Tier S
            floods[:, 0], floods[:, 1], floods[:, 2],
            dir_one_hot[:, 0], dir_one_hot[:, 1], dir_one_hot[:, 2], dir_one_hot[:, 3],
            # Tier A
            longueur_norm,
            dist_murs[:, 0], dist_murs[:, 1], dist_murs[:, 2],
            dir_queue[:, 0], dir_queue[:, 1],
            # Tier B
            dangers_profonds[:, 0], dangers_profonds[:, 1], dangers_profonds[:, 2],
            densite,
        ], axis=1).astype(np.float32)


    def step(self, actions: np.ndarray):
        """Une étape de simulation pour tous les environnements."""
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

        # === REWARDS ===
        recompenses = np.full(self.n_envs, -0.001, dtype=np.float32)  # Pénalité temps

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

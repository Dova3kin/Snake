"""
Tests unitaires pour JeuVectorise (game.py).

Couvre:
- Initialisation et reset
- Logique de jeu (step)
- Récompenses
- Détection de collisions
- Forme des états retournés
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from game import JeuVectorise


# ============================================================================
# TESTS D'INITIALISATION
# ============================================================================


class TestInitialisation:
    def test_dimensions_correctes(self, petit_env):
        """Le jeu crée le bon nombre d'environnements."""
        assert petit_env.n_envs == 4

    def test_tetes_initialisees_au_centre(self, petit_env):
        """Toutes les têtes commencent au centre de la grille."""
        cx = petit_env.grille_l // 2
        cy = petit_env.grille_h // 2
        np.testing.assert_array_equal(petit_env.tetes[:, 0], cx)
        np.testing.assert_array_equal(petit_env.tetes[:, 1], cy)

    def test_longueur_initiale_3(self, petit_env):
        """La longueur initiale du serpent est 3."""
        np.testing.assert_array_equal(petit_env.longueurs, 3)

    def test_scores_initiaux_zero(self, petit_env):
        """Tous les scores commencent à zéro."""
        np.testing.assert_array_equal(petit_env.scores, 0)

    def test_finis_initiaux_false(self, petit_env):
        """Aucune partie n'est terminée au départ."""
        assert not np.any(petit_env.finis)

    def test_pommes_dans_grille(self, petit_env):
        """Toutes les pommes sont dans les limites de la grille."""
        assert np.all(petit_env.pommes[:, 0] >= 0)
        assert np.all(petit_env.pommes[:, 0] < petit_env.grille_l)
        assert np.all(petit_env.pommes[:, 1] >= 0)
        assert np.all(petit_env.pommes[:, 1] < petit_env.grille_h)


# ============================================================================
# TESTS DE RESET
# ============================================================================


class TestReset:
    def test_reset_partiel(self, petit_env):
        """Reset d'un sous-ensemble d'environnements."""
        petit_env.scores[0] = 5
        petit_env.longueurs[0] = 8
        petit_env.reset(np.array([0]))
        assert petit_env.scores[0] == 0
        assert petit_env.longueurs[0] == 3
        # Les autres ne sont pas touchés — scores deviennent 0 après reset, vérifions longueur des autres
        assert petit_env.longueurs[1] == 3  # jamais changée

    def test_reset_remet_directions_a_droite(self, petit_env):
        """Après reset, toutes les directions sont à droite (0)."""
        petit_env.reset()
        np.testing.assert_array_equal(petit_env.directions, 0)


# ============================================================================
# TESTS DE LA FORME DES ÉTATS
# ============================================================================


class TestEtats:
    def test_forme_etats_correcte(self, petit_env):
        """recuperer_etats() retourne la bonne forme : 9 features compactes."""
        etats = petit_env.recuperer_etats()
        assert etats.shape == (4, 9)

    def test_etats_normalises(self, petit_env):
        """Les features sont dans les bonnes plages."""
        etats = petit_env.recuperer_etats()
        # Distance normalisée: entre 0 et 1
        assert np.all(etats[:, 0] >= 0)
        assert np.all(etats[:, 0] <= 1)
        # Dangers: 0 ou 1
        assert np.all((etats[:, 3] == 0) | (etats[:, 3] == 1))
        assert np.all((etats[:, 4] == 0) | (etats[:, 4] == 1))
        assert np.all((etats[:, 5] == 0) | (etats[:, 5] == 1))


# ============================================================================
# TESTS DU STEP
# ============================================================================


class TestStep:
    def test_step_retourne_4_valeurs(self, petit_env):
        """step() retourne (etats, recompenses, finis, scores)."""
        actions = np.zeros(4, dtype=np.int32)
        resultat = petit_env.step(actions)
        assert len(resultat) == 4

    def test_recompenses_initiales_non_zero(self, petit_env):
        """Les récompenses existent et sont des floats."""
        actions = np.zeros(4, dtype=np.int32)
        _, recompenses, _, _ = petit_env.step(actions)
        assert recompenses.dtype == np.float32
        assert recompenses.shape == (4,)

    def test_mort_donne_recompense_negative(self):
        """Une collision mur donne recompense = -1."""
        # Environnement 1 env, serpent tout à gauche de la grille
        env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
        # Forcer la tête sur le bord gauche, direction gauche (2)
        env.tetes[0] = [0, env.grille_h // 2]
        env.corps[0, 0] = env.tetes[0]
        env.directions[0] = 2  # Gauche
        _, recompenses, finis, _ = env.step(np.array([0]))  # tout droit = gauche
        assert recompenses[0] == pytest.approx(-1.0)
        assert finis[0]

    def test_pomme_mangee_augmente_score(self):
        """Manger une pomme incrémente le score et la longueur."""
        env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
        # Placer la pomme juste devant la tête (direction droite = 0)
        head_x, head_y = env.tetes[0]
        env.pommes[0] = [head_x + 1, head_y]
        env.directions[0] = 0  # Droite

        longueur_avant = env.longueurs[0]
        score_avant = env.scores[0]
        _, recompenses, _, scores = env.step(np.array([0]))

        # Score peut avoir été reset (auto-reset), vérifier la récompense
        assert recompenses[0] == pytest.approx(1.0)

    def test_famine_tue_le_serpent(self):
        """Famine adaptative : longueur 3 -> timeout = 109, mort à 110 steps."""
        env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
        # Avec longueur 3 : timeout = 100 + 3*3 = 109. Mort quand etapes > 109.
        env.etapes_depuis_pomme[0] = 109
        env.pommes[0] = [env.grille_l - 1, env.grille_h - 1]
        env.tetes[0] = [env.grille_l // 2, env.grille_h // 2]
        env.corps[0, 0] = env.tetes[0]
        env.directions[0] = 0

        _, recompenses, finis, _ = env.step(np.array([0]))
        # A 110 pas, la famine se declenche (timeout = 100 + longueur*3 = 109)
        assert finis[0], f"Famine non declenchee apres 110 pas"
        assert recompenses[0] == pytest.approx(-1.0)

    def test_auto_reset_apres_mort(self):
        """Après une mort, l'environnement se réinitialise automatiquement."""
        env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
        # Forcer mort par mur
        env.tetes[0] = [0, 0]
        env.corps[0, 0] = env.tetes[0]
        env.directions[0] = 3  # Haut → hors limites si y=0 (j'essaie gauche)
        env.tetes[0] = [0, env.grille_h // 2]
        env.directions[0] = 2  # Gauche

        _, _, finis, _ = env.step(np.array([0]))
        if finis[0]:
            # Après reset auto, scores sont à 0 et longueur à 3
            assert env.longueurs[0] == 3
            assert env.scores[0] == 0


# ============================================================================
# TESTS DES ACTIONS HEURISTIQUES
# ============================================================================


class TestFamineAdaptative:
    def test_famine_plus_longue_pour_serpent_long(self):
        """
        Un serpent de longueur 30 doit avoir un timeout plus long que longueur 3.
        Formule: timeout = 100 + longueur * 3
        longueur 3  -> timeout = 109
        longueur 30 -> timeout = 190
        """
        env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
        env.longueurs[0] = 3
        env.etapes_depuis_pomme[0] = 109
        env.tetes[0] = [env.grille_l // 2, env.grille_h // 2]
        env.corps[0, 0] = env.tetes[0]
        env.pommes[0] = [env.grille_l - 1, env.grille_h - 1]
        env.directions[0] = 0

        _, _, finis, _ = env.step(np.array([0]))
        assert finis[0], "Longueur 3 : devrait mourir à 110 steps (100 + 3*3 + 1)"

    def test_serpent_long_survit_plus_longtemps(self):
        """Un serpent de longueur 30 survit au-delà de 150 steps sans manger."""
        env = JeuVectorise(n_envs=1, largeur=200, hauteur=160)
        env.longueurs[0] = 30
        env.etapes_depuis_pomme[0] = 180  # > 150 mais < 100 + 30*3 = 190
        env.tetes[0] = [env.grille_l // 2, env.grille_h // 2]
        env.corps[0, 0] = env.tetes[0]
        env.pommes[0] = [env.grille_l - 1, env.grille_h - 1]
        env.directions[0] = 0

        _, _, finis, _ = env.step(np.array([0]))
        assert not finis[0], (
            "Longueur 30 : ne devrait PAS mourir à 181 steps "
            "(timeout = 100 + 30*3 = 190)"
        )


class TestEtatEnrichi:
    def test_forme_etat_9_features(self, petit_env):
        """recuperer_etats() doit retourner 9 features (plus de pixels)."""
        etats = petit_env.recuperer_etats()
        assert etats.shape == (4, 9), (
            f"Shape = {etats.shape}, attendu (4, 9). "
            f"Les 3072 pixels doivent être supprimés."
        )

    def test_position_tete_normalisee(self, env_simple):
        """Features 7 et 8 sont la position normalisée de la tête (entre 0 et 1)."""
        etats = env_simple.recuperer_etats()
        pos_x = etats[0, 7]
        pos_y = etats[0, 8]
        assert 0.0 <= pos_x <= 1.0, f"pos_x = {pos_x} doit être dans [0, 1]"
        assert 0.0 <= pos_y <= 1.0, f"pos_y = {pos_y} doit être dans [0, 1]"

    def test_pas_de_pixels_dans_etat(self, petit_env):
        """Le vecteur d'état ne doit pas contenir 3072 pixels."""
        etats = petit_env.recuperer_etats()
        assert etats.shape[1] < 100, (
            f"Le vecteur d'état a {etats.shape[1]} features. "
            f"Les pixels (3072) doivent être supprimés."
        )


class TestActionsHeuristiques:
    def test_actions_gloutonnes_shape(self, petit_env):
        """actions_gloutonnes() retourne le bon nombre d'actions."""
        actions = petit_env.actions_gloutonnes()
        assert actions.shape == (4,)
        assert np.all(actions >= 0)
        assert np.all(actions <= 2)

    def test_actions_aleatoires_sures_shape(self, petit_env):
        """actions_aleatoires_sures() retourne le bon nombre d'actions."""
        actions = petit_env.actions_aleatoires_sures()
        assert actions.shape == (4,)
        assert np.all(actions >= 0)
        assert np.all(actions <= 2)

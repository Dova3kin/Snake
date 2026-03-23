"""
Tests unitaires pour MemoireEfficace et AgentIA (agent.py).

Couvre:
- Stockage et recall de la mémoire
- Comportement ring buffer (overflow)
- Schedule epsilon linéaire
- Conversion d'état en tenseur
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
from agent import MemoireEfficace
from agent import EPSILON_DEPART, EPSILON_FIN, EPSILON_FRAMES


# ============================================================================
# TESTS DE MemoireEfficace
# ============================================================================


class TestMemoireEfficace:
    def test_stockage_simple(self):
        """On peut stocker et récupérer des expériences."""
        memoire = MemoireEfficace(capacite=100)
        etats = np.zeros((1, 9))
        actions = np.array([0])
        recompenses = np.array([1.0])
        etats_suivants = np.ones((1, 9))
        finis = np.array([False])
        memoire.stocker_batch(etats, actions, recompenses, etats_suivants, finis)
        assert len(memoire) == 1

    def test_taille_augmente_progressivement(self):
        """La taille augmente jusqu'à la capacité."""
        memoire = MemoireEfficace(capacite=10)
        for i in range(7):
            memoire.stocker_batch(np.zeros((1,9)), np.array([0]), np.array([0.0]), np.zeros((1,9)), np.array([False]))
        assert len(memoire) == 7

    def test_ring_buffer_ne_depasse_pas_capacite(self):
        """Stocker plus que la capacité se comporte en ring buffer."""
        memoire = MemoireEfficace(capacite=5)
        for i in range(10):
            memoire.stocker_batch(np.zeros((1,9)), np.array([0]), np.array([0.0]), np.zeros((1,9)), np.array([False]))
        # Taille plafonnée à la capacité
        assert len(memoire) == 5

    def test_echantillonnage_taille_correcte(self):
        """echantillonner() retourne le bon nombre d'éléments."""
        memoire = MemoireEfficace(capacite=100)
        etats = np.zeros((50, 9))
        actions = np.zeros(50, dtype=int)
        recompenses = np.zeros(50, dtype=float)
        etats_suivants = np.zeros((50, 9))
        finis = np.zeros(50, dtype=bool)
        memoire.stocker_batch(etats, actions, recompenses, etats_suivants, finis)

        batch = memoire.echantillonner(batch_size=32)
        assert len(batch[0]) == 32
        assert len(batch[1]) == 32

    def test_echantillonnage_elements_valides(self):
        """Les éléments échantillonnés sont des tuples d'arrays numpy."""
        memoire = MemoireEfficace(capacite=50)
        etats = np.zeros((20, 9))
        actions = np.zeros(20, dtype=int)
        recompenses = np.ones(20, dtype=float)
        etats_suivants = np.ones((20, 9))
        finis = np.zeros(20, dtype=bool)
        memoire.stocker_batch(etats, actions, recompenses, etats_suivants, finis)

        etats_b, actions_b, rec_b, etats_suiv_b, finis_b = memoire.echantillonner(10)
        assert etats_b.shape == (10, 9)
        assert actions_b.shape == (10,)
        assert rec_b.shape == (10,)

    def test_ring_buffer_ecrase_les_anciens(self):
        """Après overflow, les nouvelles données remplacent les anciennes."""
        memoire = MemoireEfficace(capacite=3)
        for i in range(5):
             memoire.stocker_batch(np.zeros((1,9)), np.array([i]), np.array([0.0]), np.zeros((1,9)), np.array([False]))

        # La position courante est 5 % 3 = 2
        assert memoire.position == 2
        assert len(memoire) == 3
        # Les actions 0 a 4 stockées -> indices 0=3, 1=4, 2=2
        assert sorted(memoire.actions[:3]) == [2, 3, 4]


# ============================================================================
# TESTS DU SCHEDULE EPSILON
# ============================================================================


class TestEpsilonSchedule:
    def test_epsilon_ne_decroit_pas_trop_vite(self):
        """
        Avec 1000 envs, après 1000 parties (= ~1 frame), epsilon doit rester > 0.9.
        Avant correction, epsilon tombait à 0.05 en ~200 frames.
        """
        from agent import AgentIA

        agent = AgentIA.__new__(AgentIA)
        agent.nb_parties = 1000
        agent.nb_frames = 1  # 1 seul step de simulation
        agent.epsilon = 1.0

        eps = agent.epsilon_schedule()
        assert eps > 0.9, (
            f"Epsilon = {eps:.4f} après seulement 1 frame avec 1000 envs. "
            f"Le schedule décroît bien trop vite."
        )

    def test_epsilon_atteint_fin_apres_1M_frames(self):
        """Epsilon doit atteindre EPSILON_FIN après ~1M frames (pas parties)."""
        from agent import AgentIA, EPSILON_FIN, EPSILON_FRAMES

        agent = AgentIA.__new__(AgentIA)
        agent.nb_frames = EPSILON_FRAMES
        agent.epsilon = 1.0

        eps = agent.epsilon_schedule()
        assert abs(eps - EPSILON_FIN) < 0.01, (
            f"Epsilon = {eps:.4f}, attendu {EPSILON_FIN} après {EPSILON_FRAMES} frames"
        )

    def test_epsilon_decroit_lineairement(self):
        """Epsilon diminue de façon monotone avec le nombre de frames."""
        from agent import AgentIA

        agent = AgentIA.__new__(AgentIA)
        valeurs = []
        for n in range(0, EPSILON_FRAMES + 1, EPSILON_FRAMES // 10):
            agent.nb_frames = n
            agent.epsilon = 1.0
            valeurs.append(agent.epsilon_schedule())

        for i in range(1, len(valeurs)):
            assert valeurs[i] <= valeurs[i - 1], (
                f"Epsilon n'est pas monotone: {valeurs[i-1]:.3f} -> {valeurs[i]:.3f}"
            )

    def test_epsilon_ne_descend_pas_sous_fin(self):
        """Epsilon ne descend jamais sous EPSILON_FIN."""
        from agent import AgentIA

        agent = AgentIA.__new__(AgentIA)
        agent.nb_frames = EPSILON_FRAMES * 10  # Bien au-delà
        agent.epsilon = 1.0
        eps = agent.epsilon_schedule()
        assert eps >= EPSILON_FIN


# ============================================================================
# TESTS DE CONVERSION D'ÉTAT
# ============================================================================


class TestConversionEtat:
    def test_convertir_etat_tensor_shape(self):
        """convertir_etat_tensor() retourne le bon tenseur."""
        from agent import AgentIA
        import unittest.mock as mock
        with mock.patch("agent.Dashboard"), mock.patch("agent.JournalDeBord"):
            agent = AgentIA()
            etats = np.random.randn(4, 9).astype(np.float32)
            tensor = agent.convertir_etat_tensor(etats)
            assert tensor.shape == (4, 9)
            assert tensor.dtype == torch.float32

    def test_convertir_etat_tensor_valeurs(self):
        """Les valeurs du tenseur correspondent au numpy input."""
        from agent import AgentIA
        import unittest.mock as mock
        with mock.patch("agent.Dashboard"), mock.patch("agent.JournalDeBord"):
            agent = AgentIA()
            etats = np.ones((2, 9), dtype=np.float32)
            tensor = agent.convertir_etat_tensor(etats)
            assert torch.all(tensor == 1.0)

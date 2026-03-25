"""
Fixtures partagées pour tous les tests.
"""
import pytest
import numpy as np
import torch


# Seed fixe pour reproductibilité
SEED = 42


@pytest.fixture(autouse=True)
def set_seeds():
    """Fixe les seeds pour des tests reproductibles."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture
def petit_env():
    """Environnement Snake minimal (4 envs) pour tests rapides."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game import JeuVectorise
    env = JeuVectorise(n_envs=4, largeur=100, hauteur=80)
    return env


@pytest.fixture
def env_simple():
    """Environnement Snake à 1 env pour tests précis."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game import JeuVectorise
    env = JeuVectorise(n_envs=1, largeur=100, hauteur=80)
    return env


@pytest.fixture
def modele():
    """Réseau de neurones minimal pour tests."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import ReseauNeurones
    return ReseauNeurones(input_size=26, output_size=3)


@pytest.fixture
def entraineur(modele):
    """Entraineur avec un petit réseau pour tests."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import Entraineur
    return Entraineur(modele, lr=0.001, gamma=0.97, device="cpu")

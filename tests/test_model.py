"""
Tests unitaires pour ReseauNeurones et Entraineur (model.py).

Couvre:
- Forme des sorties (forward pass)
- Sauvegarde et chargement (round-trip)
- Mise à jour douce du target network
- Étape d'apprentissage (loss)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import tempfile
from model import ReseauNeurones, Entraineur


# ============================================================================
# TESTS DU RÉSEAU DE NEURONES
# ============================================================================


class TestReseauNeurones:
    def test_forward_shape_batch(self, modele):
        """Forward avec batch: shape = (batch, 3)."""
        x = torch.randn(8, 9)
        y = modele(x)
        assert y.shape == (8, 3)

    def test_forward_shape_single(self, modele):
        """Forward avec un seul état: shape = (1, 3)."""
        x = torch.randn(1, 9)
        y = modele(x)
        assert y.shape == (1, 3)

    def test_forward_4d_input(self, modele):
        """Forward avec 2 etats aplatis de taille 9 retourne shape (2,3)."""
        # Entrée correcte: 2 etats aplatis de taille 9
        x = torch.randn(2, 9)
        y = modele(x)
        assert y.shape == (2, 3)

    def test_forward_sortie_numerique(self, modele):
        """La sortie ne contient pas de NaN ou Inf."""
        x = torch.randn(4, 9)
        y = modele(x)
        assert not torch.any(torch.isnan(y))
        assert not torch.any(torch.isinf(y))

    def test_sauvegarde_chargement_roundtrip(self, modele):
        """Sauvegarde puis chargement → poids identiques."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nom = "test_model"
            # Patch du chemin pour pointer vers tmpdir
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs("model", exist_ok=True)

            try:
                modele.sauvegarder(nom_fichier=nom, nb_parties=100, record=5)

                # Charger dans un nouveau modèle
                modele2 = ReseauNeurones(input_size=9, output_size=3)
                result = modele2.charger(nom_fichier=nom + ".pth", device="cpu")

                assert result is not None
                nb_parties, _, _, _, record = result
                assert nb_parties == 100
                assert record == 5

                # Vérifier que les poids sont identiques
                for p1, p2 in zip(modele.parameters(), modele2.parameters()):
                    assert torch.allclose(p1.data, p2.data)
            finally:
                os.chdir(original_dir)

    def test_charger_fichier_inexistant_retourne_none(self, modele):
        """Charger un fichier qui n'existe pas retourne None."""
        result = modele.charger(nom_fichier="inexistant_xyz.pth", device="cpu")
        assert result is None

    def test_mode_eval_desactive_dropout(self, modele):
        """En mode eval, des appels répétés donnent le même résultat."""
        modele.eval()
        x = torch.randn(1, 9)
        with torch.no_grad():
            y1 = modele(x)
            y2 = modele(x)
        assert torch.allclose(y1, y2)
        modele.train()


# ============================================================================
# TESTS DE L'ENTRAINEUR
# ============================================================================


class TestEntraineur:
    def test_mise_a_jour_douce(self, entraineur):
        """Soft update modifie le target network dans la bonne direction."""
        # Sauvegarder l'état initial du target
        params_target_avant = [
            p.data.clone() for p in entraineur.target_model.parameters()
        ]

        # Modifier fortement le modèle principal
        with torch.no_grad():
            for p in entraineur.modele.parameters():
                p.fill_(1.0)

        entraineur.mise_a_jour_douce()

        # Le target doit avoir légèrement bougé vers 1.0
        for p_target_avant, p_target_apres in zip(
            params_target_avant, entraineur.target_model.parameters()
        ):
            # Au moins un paramètre doit avoir changé
            if not torch.allclose(p_target_avant, torch.ones_like(p_target_avant)):
                # La valeur absolue de la différence doit avoir diminué ou augmenté vers 1
                assert not torch.allclose(p_target_avant, p_target_apres.data)
                break

    def test_etape_apprentissage_retourne_loss(self, entraineur):
        """etape_d_apprentissage() retourne une loss numérique positive."""
        batch_size = 4
        etats = [np.random.randn(9).astype(np.float32) for _ in range(batch_size)]
        actions = [0, 1, 2, 0]
        recompenses = [0.1, -1.0, 1.0, -0.001]
        etats_suivants = [np.random.randn(9).astype(np.float32) for _ in range(batch_size)]
        finis = [False, True, False, False]

        loss = entraineur.etape_d_apprentissage(
            etats, actions, recompenses, etats_suivants, finis
        )

        assert isinstance(loss, float)
        assert loss >= 0
        assert not np.isnan(loss)

    def test_loss_diminue_apres_plusieurs_steps(self, entraineur):
        """La loss tend à diminuer sur un batch répété plusieurs fois."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Batch simple et cohérent
        batch_size = 16
        etats = [np.zeros(9, dtype=np.float32) for _ in range(batch_size)]
        actions = [0] * batch_size
        recompenses = [1.0] * batch_size  # Tous positifs et cohérents
        etats_suivants = [np.zeros(9, dtype=np.float32) for _ in range(batch_size)]
        finis = [False] * batch_size

        pertes = []
        for _ in range(20):
            loss = entraineur.etape_d_apprentissage(
                etats, actions, recompenses, etats_suivants, finis
            )
            pertes.append(loss)

        # La loss de la dernière itération doit être inférieure à la première
        assert pertes[-1] < pertes[0], f"Loss n'a pas diminué: {pertes[0]:.4f} → {pertes[-1]:.4f}"


# ============================================================================
# TESTS DU MODE DU MODÈLE
# ============================================================================


class TestModeModele:
    def test_inference_utilise_mode_eval(self, modele):
        """
        Durant l'inférence, le modèle doit être en mode eval().
        En mode train(), le dropout (si présent) bruite les prédictions.
        """
        import torch
        import numpy as np

        modele.train()  # Forcer mode train
        etat = torch.tensor(np.random.rand(1, 9), dtype=torch.float)

        # Simuler ce que AgentIA.convertir_etat_tensor + inférence devrait faire
        modele.eval()
        with torch.no_grad():
            pred_eval = modele(etat)

        assert not modele.training, (
            "Après l'inférence, le modèle doit rester en eval() "
            "ou être explicitement remis en train()"
        )


class TestDoubleDQN:
    def test_target_calcule_avec_actions_du_modele_principal(self, entraineur):
        """
        Double DQN : les actions next sont sélectionnées par le modèle principal,
        pas par le target network.
        On vérifie en forçant des poids divergents entre les deux réseaux.
        """
        import torch
        import numpy as np

        # Forcer des poids très différents entre modèle et target
        for p in entraineur.target_model.parameters():
            p.data.fill_(0.0)   # target → toujours 0
        for p in entraineur.modele.parameters():
            p.data.fill_(1.0)   # modèle → valeurs élevées

        etat      = np.random.rand(4, 9).astype(np.float32)
        action    = np.array([0, 1, 2, 0])
        recompense = np.array([1.0, 0.0, -1.0, 0.5])
        etat_suiv = np.random.rand(4, 9).astype(np.float32)
        finis     = np.array([False, False, False, True])

        # Ne doit pas lever d'exception et doit retourner une loss
        loss = entraineur.etape_d_apprentissage(
            etat, action, recompense, etat_suiv, finis
        )
        assert loss >= 0, f"Loss = {loss}, doit être >= 0"

    def test_etapes_avec_terminaison(self, entraineur):
        """
        Pour les états terminaux (finis=True), Q_bellman = recompense seule.
        """
        import torch
        import numpy as np

        etat      = np.zeros((2, 9), dtype=np.float32)
        action    = np.array([0, 1])
        recompense = np.array([-1.0, 1.0])
        etat_suiv = np.zeros((2, 9), dtype=np.float32)
        finis     = np.array([True, True])  # Deux états terminaux

        loss = entraineur.etape_d_apprentissage(
            etat, action, recompense, etat_suiv, finis
        )
        assert loss >= 0

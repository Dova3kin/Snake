import torch
import numpy as np
import random
import time
import sys
import pygame
from collections import deque

from game import JeuVectorise, Point, TAILLE_BLOC
from model import ReseauNeurones, Entraineur
from dashboard import Dashboard
from logger import JournalDeBord

# ============================================================================
# SEEDS (reproductibilité)
# ============================================================================
GRAINE = 42
random.seed(GRAINE)
np.random.seed(GRAINE)
torch.manual_seed(GRAINE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GRAINE)


def journal(message):
    """Affiche un message horodaté."""
    heure = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{heure}] {message}")


# ============================================================================
# CONFIGURATION (hyperparamètres justifiés)
# ============================================================================
NB_ENVIRONNEMENTS = 1000  # Jeux en parallèle
TAILLE_BATCH = 256
MEMOIRE_MAX = 200_000

# Learning rate: 0.0001-0.0005 est optimal pour DQN
TAUX_APPRENTISSAGE = 0.0003

# Gamma: 0.97 = horizon adapté à Snake (pas trop long, pas trop court)
GAMMA = 0.97

# Fréquence d'entraînement: toutes les 8 frames
FREQ_ENTRAINEMENT = 8

# Epsilon: schedule linéaire (plus prévisible que decay exponentiel)
EPSILON_DEPART = 1.0
EPSILON_FIN = 0.05
EPSILON_FRAMES = 50_000  # Steps pour atteindre epsilon_fin (indépendant du nb d'envs)
                         # ~28 min à 30 frames/sec avec 1000 envs


# ============================================================================
# MÉMOIRE EFFICACE (Ring Buffer avec numpy)
# ============================================================================


class MemoireEfficace:
    """
    Ring buffer vectorisé pour Experience Replay.
    Utilise des arrays NumPy purs pour zéro surcharge Python.
    """

    def __init__(self, capacite, taille_etat=9):
        self.capacite = capacite
        self.etats = np.zeros((capacite, taille_etat), dtype=np.float32)
        self.actions = np.zeros(capacite, dtype=np.int32)
        self.recompenses = np.zeros(capacite, dtype=np.float32)
        self.etats_suivants = np.zeros((capacite, taille_etat), dtype=np.float32)
        self.finis = np.zeros(capacite, dtype=bool)
        
        self.position = 0
        self.taille = 0

    def stocker_batch(self, etats, actions, recompenses, etats_suivants, finis):
        n = len(etats)
        if self.position + n <= self.capacite:
            self.etats[self.position:self.position+n] = etats
            self.actions[self.position:self.position+n] = actions
            self.recompenses[self.position:self.position+n] = recompenses
            self.etats_suivants[self.position:self.position+n] = etats_suivants
            self.finis[self.position:self.position+n] = finis
        else:
            part1 = self.capacite - self.position
            part2 = n - part1
            
            self.etats[self.position:self.capacite] = etats[:part1]
            self.actions[self.position:self.capacite] = actions[:part1]
            self.recompenses[self.position:self.capacite] = recompenses[:part1]
            self.etats_suivants[self.position:self.capacite] = etats_suivants[:part1]
            self.finis[self.position:self.capacite] = finis[:part1]
            
            self.etats[0:part2] = etats[part1:]
            self.actions[0:part2] = actions[part1:]
            self.recompenses[0:part2] = recompenses[part1:]
            self.etats_suivants[0:part2] = etats_suivants[part1:]
            self.finis[0:part2] = finis[part1:]
            
        self.position = (self.position + n) % self.capacite
        self.taille = min(self.taille + n, self.capacite)

    def echantillonner(self, batch_size):
        indices = np.random.randint(0, self.taille, size=batch_size)
        return (
            self.etats[indices],
            self.actions[indices],
            self.recompenses[indices],
            self.etats_suivants[indices],
            self.finis[indices]
        )

    def __len__(self):
        return self.taille


# ============================================================================
# RENDU PYGAME
# ============================================================================


class RenduPygame:
    """Affiche le serpent n°0 à l'écran."""

    def __init__(self, env, index_env=0):
        self.env = env
        self.idx = index_env
        self.largeur = env.l
        self.hauteur = env.h
        self.surface = pygame.Surface((self.largeur, self.hauteur))

    def dessiner(self):
        self.surface.fill((0, 0, 0))

        # Serpent avec dégradé
        points_serpent = self.serpent
        nb_points = len(points_serpent)
        for i, pt in enumerate(points_serpent):
            ratio = 1 - (i / nb_points)
            luminosite = max(0.3, ratio)
            c = (int(50 * luminosite), int(200 * luminosite), int(50 * luminosite))

            pygame.draw.rect(self.surface, c, (pt.x, pt.y, TAILLE_BLOC, TAILLE_BLOC))
            pygame.draw.rect(
                self.surface, (0, 50, 0), (pt.x, pt.y, TAILLE_BLOC, TAILLE_BLOC), 1
            )

        # Pomme
        pomme = self.pomme
        pygame.draw.rect(
            self.surface, (255, 0, 0), (pomme.x, pomme.y, TAILLE_BLOC, TAILLE_BLOC)
        )

        return self.surface

    @property
    def serpent(self):
        longueur = self.env.longueurs[self.idx]
        corps = self.env.corps[self.idx, :longueur]
        return [Point(x * TAILLE_BLOC, y * TAILLE_BLOC) for x, y in corps]

    @property
    def tetes(self):
        hx, hy = self.env.tetes[self.idx]
        return Point(hx * TAILLE_BLOC, hy * TAILLE_BLOC)

    @property
    def pomme(self):
        fx, fy = self.env.pommes[self.idx]
        return Point(fx * TAILLE_BLOC, fy * TAILLE_BLOC)


# ============================================================================
# AGENT IA
# ============================================================================


class AgentIA:
    def __init__(self):
        self.nb_parties = 0
        self.nb_frames = 0   # Compteur de steps de simulation (base du schedule epsilon)

        # Epsilon: schedule linéaire
        self.epsilon = EPSILON_DEPART

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        journal(f"Device: {self.device}")

        # Réseau de neurones (MLP compact)
        # Input: 9 features (sans pixels)
        self.modele = ReseauNeurones(input_size=9, output_size=3).to(self.device)
        self.entraineur = Entraineur(
            self.modele, lr=TAUX_APPRENTISSAGE, gamma=GAMMA, device=self.device
        )

        # Mémoire efficace (ring buffer)
        self.memoire = MemoireEfficace(capacite=MEMOIRE_MAX, taille_etat=9)
        journal("Mémoire initialisée (ring buffer efficace)")

        self.logger = JournalDeBord()
        self.record = 0
        self.scores_historique = deque(maxlen=500)
        self.debut_entrainement = time.time()

        # === TEST SET INDÉPENDANT ===
        # 10 environnements dédiés au test (pas utilisés pour l'entraînement)
        self.scores_test = deque(maxlen=100)
        self.derniere_eval = 0
        self.eval_intervalle = 5000  # Éval toutes les 5000 parties

    def evaluer_test_set(self, env_test):
        """
        Évalue le modèle sur un test set indépendant.
        Exécute 100 parties en mode greedy (epsilon=0).
        """
        self.modele.eval()
        scores = []

        for _ in range(100):
            env_test.reset()
            done = False
            while not done:
                etat = env_test.recuperer_etats()
                etat_tensor = torch.tensor(etat, dtype=torch.float).to(self.device)
                with torch.no_grad():
                    action = torch.argmax(self.modele(etat_tensor)).item()
                _, _, done_arr, score_arr = env_test.step(np.array([action]))
                done = done_arr[0]
                if done:
                    scores.append(score_arr[0])

        self.modele.train()
        return np.mean(scores), np.std(scores)

    def epsilon_schedule(self):
        """
        Schedule linéaire basé sur les FRAMES (steps de simulation).
        Robuste au nombre d'envs parallèles : 1000 envs = 1000 parties/frame,
        mais toujours 1 frame/frame.
        """
        return max(
            EPSILON_FIN,
            EPSILON_DEPART
            - (self.nb_frames / EPSILON_FRAMES) * (EPSILON_DEPART - EPSILON_FIN),
        )

    def convertir_etat_tensor(self, etats_numpy):
        return torch.tensor(etats_numpy, dtype=torch.float).to(self.device)

    def memoriser_batch(self, etats, actions, recompenses, etats_suivants, finis):
        """Stocke les expériences en batch massif sans boucle Python."""
        self.memoire.stocker_batch(etats, actions, recompenses, etats_suivants, finis)

    def entrainer_memoire(self):
        """Apprentissage sur un mini-batch directement avec numpy."""
        if len(self.memoire) > TAILLE_BATCH:
            etats, actions, rewards, next_states, dones = self.memoire.echantillonner(TAILLE_BATCH)

            self.entraineur.etape_d_apprentissage(
                etats, actions, rewards, next_states, dones
            )

    def moyenne_mobile(self, n=100):
        """Moyenne sur les N dernières parties."""
        if len(self.scores_historique) == 0:
            return 0.0
        recent = list(self.scores_historique)[-n:]
        return sum(recent) / len(recent)


# ============================================================================
# BOUCLE D'ENTRAÎNEMENT
# ============================================================================


def lancer_entrainement():
    env = JeuVectorise(n_envs=NB_ENVIRONNEMENTS)
    env_test = JeuVectorise(n_envs=1)  # Test set indépendant
    agent = AgentIA()
    dashboard = Dashboard()
    visu = RenduPygame(env, index_env=0)

    t0 = time.time()
    frames = 0
    donnees_graphique = []
    moyennes_graphique = []
    score_cumule = 0
    dernier_update_graph = 0
    last_screen_time = time.time()

    # Métriques test set
    test_scores = []
    derniere_eval_test = 0

    etats = env.recuperer_etats()

    journal(f"Démarrage avec {NB_ENVIRONNEMENTS} environnements parallèles")
    journal(
        f"Epsilon schedule: {EPSILON_DEPART} → {EPSILON_FIN} sur {EPSILON_FRAMES} frames"
    )
    journal("Test set: évaluation toutes les 5000 parties")

    while True:
        # Gestion événements
        evenements = pygame.event.get()
        action_user = None
        for event in evenements:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            act = dashboard.handle_input(event)
            if act:
                action_user = act

        if dashboard.state != "RUNNING":
            dashboard.update()
            continue

        if action_user:
            if action_user == "QUIT":
                pygame.quit()
                sys.exit()
            elif action_user == "EXPORT":
                agent.logger.exporter_excel()
            elif isinstance(action_user, tuple):
                cmd, fichier = action_user
                if cmd == "SAVE":
                    temps_jeu = time.time() - agent.debut_entrainement
                    agent.modele.sauvegarder(
                        nom_fichier=fichier,
                        nb_parties=agent.nb_parties,
                        temps_total=temps_jeu,
                        etat_optimiseur=agent.entraineur.optimiseur.state_dict(),
                        epsilon=agent.epsilon,
                        record=agent.record,
                    )
                    journal(f"Sauvegardé: {fichier}")
                elif cmd == "LOAD":
                    res = agent.modele.charger(nom_fichier=fichier, device=agent.device)
                    if res is not None:
                        nb, t, opt, eps, rec = res
                        agent.nb_parties = nb
                        agent.debut_entrainement = time.time() - t
                        agent.record = rec
                        if eps is not None:
                            agent.epsilon = eps
                        if opt is not None:
                            try:
                                agent.entraineur.optimiseur.load_state_dict(opt)
                            except RuntimeError as e:
                                journal(f"Optimiseur non chargé: {e}")
                        agent.entraineur.target_model.load_state_dict(
                            agent.modele.state_dict()
                        )
                        journal(f"Chargé: {fichier}")

        # Mise à jour epsilon (schedule linéaire)
        agent.epsilon = agent.epsilon_schedule()

        # Inférence
        etat_tensor = agent.convertir_etat_tensor(etats)

        agent.modele.eval()
        with torch.no_grad():
            prediction = agent.modele(etat_tensor)
        agent.modele.train()

        # Epsilon-greedy strict : l'IA explore par elle-même et apprend de ses erreurs
        # plutôt que de simplement imiter un algorithme (professeur).
        masque_exploration = np.random.random(NB_ENVIRONNEMENTS) < agent.epsilon
        actions_modele = torch.argmax(prediction, dim=1).cpu().numpy()

        # Exploration : actions totalement aléatoires pour qu'il découvre les conséquences (murs, corps, pommes)
        actions_aleatoires = np.random.randint(0, 3, size=NB_ENVIRONNEMENTS)

        coups_finaux = np.where(masque_exploration, actions_aleatoires, actions_modele)

        # Step
        etats_suivants, recompenses, finis, scores = env.step(coups_finaux)

        # Mémoriser
        agent.memoriser_batch(etats, coups_finaux, recompenses, etats_suivants, finis)

        # Entraîner
        if agent.nb_parties > 100:
            if frames % FREQ_ENTRAINEMENT == 0:
                agent.entrainer_memoire()
        else:
            agent.entrainer_memoire()

        etats = etats_suivants
        agent.nb_frames += 1

        # Suivi scores
        nb_morts = np.sum(finis)
        if nb_morts > 0:
            agent.nb_parties += nb_morts
            scores_morts = scores[finis]
            for s in scores_morts:
                agent.scores_historique.append(s)

        max_actuel = np.max(scores)
        if max_actuel > agent.record:
            agent.record = max_actuel
            journal(f"🏆 Nouveau Record: {agent.record}")
            agent.modele.sauvegarder(
                nb_parties=agent.nb_parties,
                temps_total=time.time() - agent.debut_entrainement,
                etat_optimiseur=agent.entraineur.optimiseur.state_dict(),
                epsilon=agent.epsilon,
                record=agent.record,
            )

        frames += 1
        if time.time() - t0 > 1.0:
            tps = frames * NB_ENVIRONNEMENTS
            moyenne = agent.moyenne_mobile(100)
            journal(
                f"{tps} TPS | Parties: {agent.nb_parties} | Eps: {agent.epsilon:.3f} | "
                f"Moy100: {moyenne:.1f} | Record: {agent.record}"
            )

            agent.logger.noter_stats(
                agent.nb_parties, agent.epsilon, agent.record, moyenne, tps
            )

            # === ÉVALUATION TEST SET ===
            if agent.nb_parties - derniere_eval_test >= agent.eval_intervalle:
                derniere_eval_test = agent.nb_parties
                moy_test, std_test = agent.evaluer_test_set(env_test)
                test_scores.append(moy_test)
                journal(
                    f"📊 TEST SET: {moy_test:.1f} ± {std_test:.1f} (100 parties greedy)"
                )

            frames = 0
            t0 = time.time()

        # Screenshots auto
        if dashboard.auto_screen_active:
            if time.time() - last_screen_time >= dashboard.screen_interval:
                dashboard._take_screenshot()
                last_screen_time = time.time()

        # Rendu visuel
        if dashboard.state == "RUNNING":
            surface_jeu = visu.dessiner()
            dashboard.update_game(surface_jeu)

            dashboard.update_info(
                agent.nb_parties,
                time.time() - agent.debut_entrainement,
                agent.epsilon,
                agent.record,
            )

        # Graphiques
        if agent.nb_parties - dernier_update_graph > 100:
            dernier_update_graph = agent.nb_parties
            if len(agent.scores_historique) > 0:
                moy = agent.moyenne_mobile(100)

                # Ajustement LR
                agent.entraineur.scheduler.step(moy)

                donnees_graphique.append(moy)
                score_cumule += moy
                moy_globale = score_cumule / len(donnees_graphique)
                moyennes_graphique.append(moy_globale)

                dashboard.update_plots(
                    donnees_graphique, moyennes_graphique, agent.record
                )
                dashboard.update_global_plot(list(agent.scores_historique))
            dashboard.update()
        else:
            dashboard.update()


if __name__ == "__main__":
    lancer_entrainement()

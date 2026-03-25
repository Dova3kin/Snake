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
NB_ENVIRONNEMENTS = 1000
TAILLE_BATCH_PQN = 512      # Samples du frame courant par update
TAUX_APPRENTISSAGE = 0.0003
GAMMA = 0.97
BLEND_FRAMES = 15_000       # Transition heuristique → safe-random (~10 min)

# Epsilon schedule linéaire
EPSILON_DEPART = 1.0
EPSILON_FIN = 0.05
EPSILON_FRAMES = 50_000


# ============================================================================
# MÉMOIRE EFFICACE (Ring Buffer avec numpy)
# ============================================================================


class MemoireEfficace:
    """
    Ring buffer vectorisé pour Experience Replay.
    Utilise des arrays NumPy purs pour zéro surcharge Python.
    """

    def __init__(self, capacite, taille_etat=26):
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
        # Input: 26 features enrichies (flood fill, one-hot direction, etc.)
        self.modele = ReseauNeurones(input_size=26, output_size=3).to(self.device)
        self.entraineur = Entraineur(
            self.modele, lr=TAUX_APPRENTISSAGE, gamma=GAMMA, device=self.device
        )

        self.logger = JournalDeBord()
        self.record = 0
        self.scores_historique = deque(maxlen=500)
        self.scores_tous = []          # historique complet pour le graphique global
        self.debut_entrainement = time.time()

        # Buffer dédié aux expériences positives (pommes mangées)
        # Surreprésentées dans chaque batch : 50% pommes, 50% normal
        # Résout le problème de sparse reward (~1 pomme/256 samples → 128/256)
        self.memoire_pommes = MemoireEfficace(capacite=20_000, taille_etat=26)

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

    def entrainer_on_policy(self, etats, actions, recompenses, etats_suivants, finis):
        """
        PQN: entraînement on-policy sur le batch courant.
        Pas de replay buffer → pas de distribution mismatch.
        Oversampling des pommes via memoire_pommes.
        """
        n = len(etats)
        n_sample = min(n, TAILLE_BATCH_PQN)
        indices = np.random.choice(n, n_sample, replace=False)

        if len(self.memoire_pommes) >= 128:
            e2, a2, r2, n2, d2 = self.memoire_pommes.echantillonner(128)
            E = np.concatenate([etats[indices], e2])
            A = np.concatenate([actions[indices], a2])
            R = np.concatenate([recompenses[indices], r2])
            N = np.concatenate([etats_suivants[indices], n2])
            D = np.concatenate([finis[indices], d2])
        else:
            E, A, R, N, D = etats[indices], actions[indices], recompenses[indices], etats_suivants[indices], finis[indices]

        self.entraineur.etape_d_apprentissage(E, A, R, N, D)

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
    donnees_graphique = deque(maxlen=2000)
    moyennes_graphique = deque(maxlen=2000)
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

        masque_exploration = np.random.random(NB_ENVIRONNEMENTS) < agent.epsilon
        actions_modele = torch.argmax(prediction, dim=1).cpu().numpy()

        # Transition progressive heuristique → safe-random sur BLEND_FRAMES
        # p=1.0 au début (100% heuristique), p=0.0 à la fin (100% safe-random)
        # Évite l'effondrement brutal et laisse le temps au réseau d'apprendre
        p_heuristic = max(0.0, 1.0 - agent.nb_frames / BLEND_FRAMES)
        if p_heuristic > 0:
            masque_h = np.random.random(NB_ENVIRONNEMENTS) < p_heuristic
            actions_h = env.actions_gloutonnes()
            actions_s = env.actions_aleatoires_sures()
            actions_aleatoires = np.where(masque_h, actions_h, actions_s)
        else:
            actions_aleatoires = env.actions_aleatoires_sures()

        coups_finaux = np.where(masque_exploration, actions_aleatoires, actions_modele)

        # Step
        etats_suivants, recompenses, finis, scores = env.step(coups_finaux)

        # Stocker les pommes dans le buffer dédié (oversampling food)
        masque_positif = recompenses > 0.5
        if np.any(masque_positif):
            agent.memoire_pommes.stocker_batch(
                etats[masque_positif],
                coups_finaux[masque_positif],
                recompenses[masque_positif],
                etats_suivants[masque_positif],
                finis[masque_positif],
            )

        # PQN: entraîner directement sur le frame courant (on-policy)
        agent.entrainer_on_policy(etats, coups_finaux, recompenses, etats_suivants, finis)

        etats = etats_suivants
        agent.nb_frames += 1

        # Suivi scores
        nb_morts = np.sum(finis)
        if nb_morts > 0:
            agent.nb_parties += nb_morts
            scores_morts = scores[finis]
            for s in scores_morts:
                agent.scores_historique.append(s)
                agent.scores_tous.append(int(s))

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
            score_agent0 = int(env.scores[0])
            dashboard.update_game(surface_jeu, score_actuel=score_agent0)
            dashboard.update_nn(etats[0])

            dashboard.update_info(
                agent.nb_parties,
                time.time() - agent.debut_entrainement,
                agent.epsilon,
                agent.record,
                pommes_total=len(agent.scores_tous),
            )

        # Graphiques
        if agent.nb_parties - dernier_update_graph > 100:
            dernier_update_graph = agent.nb_parties
            if len(agent.scores_historique) > 0:
                moy = agent.moyenne_mobile(100)

                if agent.epsilon < 0.2:
                    agent.entraineur.scheduler.step(moy)

                donnees_graphique.append(moy)
                moy_globale = sum(donnees_graphique) / len(donnees_graphique)
                moyennes_graphique.append(moy_globale)

                dashboard.update_plots(
                    list(donnees_graphique), list(moyennes_graphique), agent.record
                )
                dashboard.update_global_plot(agent.scores_tous)
            dashboard.update()
        else:
            dashboard.update()


if __name__ == "__main__":
    lancer_entrainement()

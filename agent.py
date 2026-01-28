import torch
import numpy as np
import random
import time
import sys
import pygame
from collections import deque

# Imports de nos propres fichiers (avec les nouveaux noms français)
from game import JeuVectorise, Point, TAILLE_BLOC
from model import ReseauNeurones, Entraineur
from dashboard import Dashboard
from logger import JournalDeBord

# ============================================================================
# RÉGLAGES POUR AVOIR TOUJOURS LE MÊME RÉSULTAT (SEEDS)
# ============================================================================
GRAINE = 42
random.seed(GRAINE)
np.random.seed(GRAINE)
torch.manual_seed(GRAINE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GRAINE)


def journal(message):
    """Petite fonction pour afficher l'heure dans la console."""
    heure = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{heure}] {message}")


# ============================================================================
# CONFIGURATION DE L'IA
# ============================================================================
NB_ENVIRONNEMENTS = 1000  # Nombre de parties en parallèle
TAILLE_BATCH = 256  # Nombre d'exemples pour apprendre à chaque fois
MEMOIRE_MAX = 200_000  # Taille de la mémoire courte
TAUX_APPRENTISSAGE = 0.0001
GAMMA = 0.99  # Importance du futur (0.99 = très important)
FREQ_ENTRAINEMENT = 8  # On entraîne le modèle toutes les 8 frames

# Paramètres pour le "Prioritized Experience Replay" (PER)
# C'est une technique pour apprendre plus des erreurs importantes
PER_ALPHA = 0.6
PER_BETA_DEBUT = 0.4
PER_BETA_DUREE = 100_000


# ============================================================================
# GESTION DE LA MÉMOIRE (PER)
# ============================================================================


class ArbreSomme:
    """
    Structure de données un peu complexe pour retrouver vite fait
    les priorités des souvenirs. C'est un arbre binaire.
    """

    def __init__(self, capacite):
        self.capacite = capacite
        self.arbre = np.zeros(2 * capacite - 1, dtype=np.float64)
        self.donnees = np.zeros(capacite, dtype=object)
        self.curseur = 0
        self.nb_entrees = 0

    def _propager(self, idx, chgmt):
        parent = (idx - 1) // 2
        self.arbre[parent] += chgmt
        if parent != 0:
            self._propager(parent, chgmt)

    def _retrouver(self, idx, s):
        gauche = 2 * idx + 1
        droite = gauche + 1

        if gauche >= len(self.arbre):
            return idx

        if s <= self.arbre[gauche]:
            return self._retrouver(gauche, s)
        else:
            return self._retrouver(droite, s - self.arbre[gauche])

    def total(self):
        return self.arbre[0]

    def ajouter(self, priorite, data):
        idx = self.curseur + self.capacite - 1

        self.donnees[self.curseur] = data
        self.maj(idx, priorite)

        self.curseur = (self.curseur + 1) % self.capacite
        self.nb_entrees = min(self.nb_entrees + 1, self.capacite)

    def maj(self, idx, priorite):
        chgmt = priorite - self.arbre[idx]
        self.arbre[idx] = priorite
        self._propager(idx, chgmt)

    def recuperer(self, s):
        idx = self._retrouver(0, s)
        data_idx = idx - self.capacite + 1
        return idx, self.arbre[idx], self.donnees[data_idx]


class MemoirePrioritaire:
    """
    Mémoire intelligente qui retient les moments importants.
    """

    def __init__(self, capacite, alpha=0.6, beta_debut=0.4, beta_frames=100_000):
        self.arbre = ArbreSomme(capacite)
        self.capacite = capacite
        self.alpha = alpha
        self.beta_debut = beta_debut
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priorite = 1.0
        self.min_priorite = 1e-5

    def _calculer_beta(self):
        # Beta augmente petit à petit jusqu'à 1
        return min(
            1.0,
            self.beta_debut + self.frame * (1.0 - self.beta_debut) / self.beta_frames,
        )

    def stocker(self, experience):
        # On donne la priorité max par défaut pour être sûr que ce soit revu au moins une fois
        priorite = self.max_priorite**self.alpha
        self.arbre.ajouter(priorite, experience)

    def echantillonner(self, batch_size):
        batch = []
        indices = []
        priorites = []
        segment = self.arbre.total() / batch_size
        beta = self._calculer_beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priorite, data = self.arbre.recuperer(s)

            # Sécurité si données invalides
            if data is None or (isinstance(data, int) and data == 0):
                s = np.random.uniform(0, self.arbre.total())
                idx, priorite, data = self.arbre.recuperer(s)

            batch.append(data)
            indices.append(idx)
            priorites.append(priorite)

        # Calcul des poids
        probabilites = np.array(priorites) / self.arbre.total()
        probabilites = np.clip(probabilites, 1e-8, 1.0)
        poids = (self.arbre.nb_entrees * probabilites) ** (-beta)
        poids = poids / poids.max()

        self.frame += 1
        return batch, indices, poids

    def maj_priorites(self, indices, erreurs_td):
        for idx, erreur in zip(indices, erreurs_td):
            prio = (abs(erreur) + self.min_priorite) ** self.alpha
            self.max_priorite = max(self.max_priorite, prio)
            self.arbre.maj(idx, prio)

    def __len__(self):
        return self.arbre.nb_entrees


class RenduPygame:
    """
    Fait le lien entre le jeu vectorisé et Pygame
    pour dessiner le serpent n°0 à l'écran
    """

    def __init__(self, env, index_env=0):
        self.env = env
        self.idx = index_env
        self.largeur = env.l
        self.hauteur = env.h
        self.surface = pygame.Surface((self.largeur, self.hauteur))

    def dessiner(self):
        self.surface.fill((0, 0, 0))

        # On dessine le serpent
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

        # La pomme
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


class AgentIA:
    def __init__(self):
        self.nb_parties = 0
        self.epsilon = 1.0  # Au début, l'IA fait n'importe quoi (exploration max)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # Diminue très doucement

        # Système "coup de pied" si ça stagne
        self.compteur_stagnation = 0
        self.dernier_score_moyen = 0.0
        self.seuil_stagnation = 500

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        journal(f"Cerveau initialisé sur : {self.device}")

        # On crée le réseau de neurones
        self.modele = ReseauNeurones(output_size=3).to(self.device)
        self.entraineur = Entraineur(
            self.modele, lr=TAUX_APPRENTISSAGE, gamma=GAMMA, device=self.device
        )

        # La mémoire (Experience Replay)
        self.memoire = MemoirePrioritaire(
            capacite=MEMOIRE_MAX,
            alpha=PER_ALPHA,
            beta_debut=PER_BETA_DEBUT,
            beta_frames=PER_BETA_DUREE,
        )
        journal("Mémoire activée")

        self.logger = JournalDeBord()
        self.record = 0
        self.scores_historique = deque(maxlen=2000)
        self.debut_entrainement = time.time()

    def convertir_etat_tensor(self, etats_numpy):
        return torch.tensor(etats_numpy, dtype=torch.float).to(self.device)

    def memoriser_batch(self, etats, actions, recompenses, etats_suivants, finis):
        """Stocke tout ce qui vient de se passer dans la mémoire."""
        action_one_hots = np.zeros((NB_ENVIRONNEMENTS, 3), dtype=int)
        action_one_hots[np.arange(NB_ENVIRONNEMENTS), actions] = 1

        for i in range(NB_ENVIRONNEMENTS):
            exp = (
                etats[i],
                action_one_hots[i],
                recompenses[i],
                etats_suivants[i],
                finis[i],
            )
            self.memoire.stocker(exp)

    def entrainer_memoire(self):
        """C'est là que l'IA apprend en revoyant ses souvenirs."""
        if len(self.memoire) > TAILLE_BATCH:
            mini_batch, indices, poids = self.memoire.echantillonner(TAILLE_BATCH)

            # On filtre au cas où y'a des trucs bizarres
            batch_valide = [
                e for e in mini_batch if e is not None and not isinstance(e, int)
            ]
            if len(batch_valide) < TAILLE_BATCH // 2:
                return

            etats, actions, rewards, next_states, dones = zip(*batch_valide)

            td_errors = self.entraineur.etape_d_apprentissage(
                etats,
                actions,
                rewards,
                next_states,
                dones,
                weights=poids[: len(batch_valide)],
            )

            if td_errors is not None and len(td_errors) > 0:
                self.memoire.maj_priorites(indices[: len(td_errors)], td_errors)

    def maj_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def verifier_stagnation(self, score_moyen):
        """Si le score n'augmente plus, on ré-augmente epsilon pour explorer."""
        if score_moyen <= self.dernier_score_moyen:
            self.compteur_stagnation += 1
        else:
            self.compteur_stagnation = 0
            self.dernier_score_moyen = score_moyen

        if self.compteur_stagnation >= self.seuil_stagnation and self.epsilon < 0.3:
            vieux_eps = self.epsilon
            self.epsilon = min(0.4, self.epsilon + 0.1)
            self.compteur_stagnation = 0
            journal(
                f"COUP DE POUCE : Epsilon augmente de {vieux_eps:.3f} à {self.epsilon:.3f}"
            )
            return True
        return False


def lancer_entrainement():
    env = JeuVectorise(n_envs=NB_ENVIRONNEMENTS)
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

    etats = env.recuperer_etats()

    journal(f"C'est parti ! {NB_ENVIRONNEMENTS} serpents s'entraînent en même temps.")

    while True:
        # --- Gestion Clavier/Souris ---
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
                # Sauvegarder ou Charger
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
                    journal(f"Sauvegardé sous : {fichier}")
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
                            except Exception as e:
                                journal(
                                    f"Pas pu charger l'optimiseur ({e}), on repart à zéro pour lui."
                                )
                        agent.entraineur.target_model.load_state_dict(
                            agent.modele.state_dict()
                        )
                        journal(f"Chargé : {fichier}")

        # --- L'IA réfléchit ---
        etat_tensor = agent.convertir_etat_tensor(etats)

        with torch.no_grad():
            prediction = agent.modele(etat_tensor)

        # Stratégie Epsilon-Greedy : Exploration vs Imitation
        masque_imitation = np.random.random(NB_ENVIRONNEMENTS) < agent.epsilon
        actions_modele = torch.argmax(prediction, dim=1).cpu().numpy()

        # Le "professeur" (algo glouton) donne la bonne réponse
        actions_prof = env.actions_gloutonnes()

        coups_finaux = np.where(masque_imitation, actions_prof, actions_modele)

        # Un peu de hasard pur pour débloquer les situations coincées
        masque_random = np.random.random(NB_ENVIRONNEMENTS) < 0.05
        if np.any(masque_random):
            actions_sur = env.actions_aleatoires_sures()
            coups_finaux = np.where(masque_random, actions_sur, coups_finaux)

        # --- On joue ---
        etats_suivants, recompenses, finis, scores = env.step(coups_finaux)

        # --- On mémorise ---
        agent.memoriser_batch(etats, coups_finaux, recompenses, etats_suivants, finis)

        if agent.nb_parties > 100:
            if frames % FREQ_ENTRAINEMENT == 0:
                agent.entrainer_memoire()
        else:
            agent.entrainer_memoire()

        etats = etats_suivants

        # --- Suivi des scores ---
        nb_morts = np.sum(finis)
        if nb_morts > 0:
            agent.nb_parties += nb_morts
            agent.maj_epsilon()
            scores_morts = scores[finis]
            for s in scores_morts:
                agent.scores_historique.append(s)

        max_actuel = np.max(scores)
        if max_actuel > agent.record:
            agent.record = max_actuel
            journal(f"Nouveau Record : {agent.record}")
            # Auto-save record
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
            journal(
                f"{tps} TPS | Parties: {agent.nb_parties} | Eps: {agent.epsilon:.3f} | Record: {agent.record}"
            )

            moyenne = 0
            if agent.scores_historique:
                moyenne = sum(agent.scores_historique) / len(agent.scores_historique)

            agent.logger.noter_stats(
                agent.nb_parties, agent.epsilon, agent.record, moyenne, tps
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
            activations = agent.modele.recuperer_activations(
                etat_tensor[0].unsqueeze(0)
            )
            surface_jeu = visu.dessiner()
            dashboard.update_game(surface_jeu)
            dashboard.update_nn(agent.modele, activations)
            dashboard.update_info(
                agent.nb_parties,
                time.time() - agent.debut_entrainement,
                agent.epsilon,
                agent.record,
            )

        # Graphiques (pas tout le temps pour pas ramer)
        if agent.nb_parties - dernier_update_graph > 100:
            dernier_update_graph = agent.nb_parties
            if len(agent.scores_historique) > 0:
                recent = list(agent.scores_historique)[-100:]
                moy = sum(recent) / len(recent)

                # Ajustement auto du taux d'apprentissage
                agent.entraineur.scheduler.step(moy)
                agent.verifier_stagnation(moy)

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

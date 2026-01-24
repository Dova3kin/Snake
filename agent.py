import torch
import numpy as np
import random
import time
import sys
import pygame
from collections import deque

from game import VectorizedSnakeEnv, Direction, Point, BLOCK_SIZE
from model import ConvNet_QNet, QTrainer
from dashboard import Dashboard
from logger import SimulationLogger

# ============================================================================
# REPRODUCTIBILITÉ (Seeds)
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def log(message):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")


# ============================================================================
# CONFIGURATION
# ============================================================================
N_ENVS = 1000
BATCH_SIZE = 256
MAX_MEMORY = 200_000
LR = 0.0001
GAMMA = 0.99
TRAIN_EVERY = 8

# PER Hyperparamètres
PER_ALPHA = 0.6  # Degré de priorisation (0=uniforme, 1=full priorité)
PER_BETA_START = 0.4  # Importance Sampling initial
PER_BETA_FRAMES = 100_000  # Frames pour atteindre beta=1


# ============================================================================
# PRIORITIZED EXPERIENCE REPLAY (PER)
# ============================================================================


class SumTree:
    """
    Structure de données Sum Tree pour échantillonnage O(log n).
    Chaque nœud parent = somme des enfants.
    Les feuilles contiennent les priorités.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propage le changement de priorité jusqu'à la racine."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrouve l'index de la feuille pour une valeur s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Retourne la somme totale des priorités."""
        return self.tree[0]

    def add(self, priority, data):
        """Ajoute une nouvelle expérience avec sa priorité."""
        idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Met à jour la priorité d'un nœud."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Récupère l'expérience correspondant à la valeur s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Buffer d'expériences avec échantillonnage priorisé.

    - alpha: degré de priorisation (0=uniforme, 1=full priorité)
    - beta: correction d'Importance Sampling (augmente vers 1)
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0
        self.min_priority = 1e-5

    def _get_beta(self):
        """Beta augmente linéairement de beta_start à 1."""
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

    def push(self, experience):
        """Ajoute une expérience avec priorité maximale (sera ajustée après le premier entraînement)."""
        priority = self.max_priority**self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """
        Échantillonne un batch avec priorisation.
        Retourne: (batch, indices, weights)
        """
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size
        beta = self._get_beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            if data is None or (isinstance(data, int) and data == 0):
                # Donnée invalide, réessayer
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calcul des poids d'Importance Sampling
        sampling_probabilities = np.array(priorities) / self.tree.total()
        sampling_probabilities = np.clip(sampling_probabilities, 1e-8, 1.0)

        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalisation

        self.frame += 1
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Met à jour les priorités basées sur les TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class VectorRenderWrapper:
    """Adapte l'environnement vectoriel pour l'affichage Pygame (Visualise l'agent 0)."""

    def __init__(self, vector_env, env_index=0):
        self.env = vector_env
        self.idx = env_index
        self.width = vector_env.w
        self.height = vector_env.h
        self.surface = pygame.Surface((self.width, self.height))
        self.render_mode = True

    def render(self):
        """Dessine l'état du jeu."""
        self.surface.fill((0, 0, 0))

        # Dessin du Serpent avec Gradient
        snake_points = self.snake
        n_points = len(snake_points)
        for i, pt in enumerate(snake_points):
            ratio = 1 - (i / n_points)
            brightness = max(0.3, ratio)
            color = (int(50 * brightness), int(200 * brightness), int(50 * brightness))

            pygame.draw.rect(self.surface, color, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(
                self.surface, (0, 50, 0), (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1
            )

        # Dessin de la Nourriture
        food = self.food
        pygame.draw.rect(
            self.surface, (255, 0, 0), (food.x, food.y, BLOCK_SIZE, BLOCK_SIZE)
        )

        return self.surface

    @property
    def snake(self):
        length = self.env.lengths[self.idx]
        body_array = self.env.bodies[self.idx, :length]
        return [Point(x * BLOCK_SIZE, y * BLOCK_SIZE) for x, y in body_array]

    @property
    def head(self):
        hx, hy = self.env.heads[self.idx]
        return Point(hx * BLOCK_SIZE, hy * BLOCK_SIZE)

    @property
    def food(self):
        fx, fy = self.env.foods[self.idx]
        return Point(fx * BLOCK_SIZE, fy * BLOCK_SIZE)

    @property
    def score(self):
        return self.env.scores[self.idx]

    @property
    def direction(self):
        d_idx = self.env.dirs[self.idx]
        if d_idx == 0:
            return Direction.RIGHT
        if d_idx == 1:
            return Direction.DOWN
        if d_idx == 2:
            return Direction.LEFT
        if d_idx == 3:
            return Direction.UP
        return Direction.RIGHT


class VectorAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # 100% imitation au début (phase d'apprentissage)
        self.epsilon_min = 0.05
        self.epsilon_decay = (
            0.9995  # Décroissance LENTE pour laisser le temps d'apprendre
        )

        # Epsilon Kicker
        self.stagnation_counter = 0
        self.last_mean_score = 0.0
        self.stagnation_threshold = 500

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"✅ Moteur Vectorisé initialisé sur : {self.device}")

        self.model = ConvNet_QNet(output_size=3).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA, device=self.device)

        # Prioritized Experience Replay (PER)
        self.memory = PrioritizedReplayBuffer(
            capacity=MAX_MEMORY,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
        )
        log(f"🧠 PER activé (alpha={PER_ALPHA}, beta_start={PER_BETA_START})")

        self.logger = SimulationLogger()

        self.record = 0
        self.all_scores = deque(maxlen=2000)
        self.start_time = time.time()

    def get_state_tensor(self, states_numpy):
        return torch.tensor(states_numpy, dtype=torch.float).to(self.device)

    def remember_bulk(self, states, actions, rewards, next_states, dones):
        """Stockage en masse des transitions dans le PER buffer."""
        action_one_hots = np.zeros((N_ENVS, 3), dtype=int)
        action_one_hots[np.arange(N_ENVS), actions] = 1

        for i in range(N_ENVS):
            experience = (
                states[i],
                action_one_hots[i],
                rewards[i],
                next_states[i],
                dones[i],
            )
            self.memory.push(experience)

    def train_long_memory(self):
        """Entraînement avec Prioritized Experience Replay."""
        if len(self.memory) > BATCH_SIZE:
            # Échantillonnage priorisé
            mini_batch, indices, weights = self.memory.sample(BATCH_SIZE)

            # Vérification des données valides
            valid_batch = [
                exp
                for exp in mini_batch
                if exp is not None and not isinstance(exp, int)
            ]
            if len(valid_batch) < BATCH_SIZE // 2:
                return  # Pas assez de données valides

            states, actions, rewards, next_states, dones = zip(*valid_batch)

            # Entraînement avec poids d'Importance Sampling
            td_errors = self.trainer.train_step(
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights=weights[: len(valid_batch)],
            )

            # Mise à jour des priorités basée sur les TD-errors
            if td_errors is not None and len(td_errors) > 0:
                self.memory.update_priorities(indices[: len(td_errors)], td_errors)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def check_stagnation(self, current_mean_score):
        """Réaugmente epsilon si le score moyen stagne (Epsilon Kicker)."""
        if current_mean_score <= self.last_mean_score:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_mean_score = current_mean_score

        if self.stagnation_counter >= self.stagnation_threshold and self.epsilon < 0.3:
            old_eps = self.epsilon
            self.epsilon = min(
                0.4, self.epsilon + 0.1
            )  # Réduit de 0.2 à 0.1 pour plus de stabilité
            self.stagnation_counter = 0
            log(f"🔄 EPSILON KICKER: {old_eps:.3f} → {self.epsilon:.3f}")
            return True
        return False


def train_vectorized():
    env = VectorizedSnakeEnv(n_envs=N_ENVS)
    agent = VectorAgent()
    dashboard = Dashboard()
    render_wrapper_0 = VectorRenderWrapper(env, env_index=0)

    t0 = time.time()
    frames = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    last_plot_update = 0
    last_screen_time = time.time()

    states = env.get_states()

    log(f"ℹ️  Lancement de la simulation : {N_ENVS} agents parallèles.")

    while True:
        # --- Interface Utilisateur ---
        events = pygame.event.get()
        action = None
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            act = dashboard.handle_input(event)
            if act:
                action = act

        if dashboard.state != "RUNNING":
            dashboard.update()
            continue

        if action:
            if action == "QUIT":
                pygame.quit()
                sys.exit()
            elif action == "EXPORT":
                agent.logger.export_excel()
            elif isinstance(action, tuple):
                if action[0] == "SAVE":
                    current_time = time.time() - agent.start_time
                    agent.model.save(
                        file_name=action[1],
                        n_games=agent.n_games,
                        total_time=current_time,
                        optimizer_state=agent.trainer.optimizer.state_dict(),
                        epsilon=agent.epsilon,
                        record=agent.record,
                    )
                    log(f"💾 Modèle sauvegardé : {action[1]}")
                elif action[0] == "LOAD":
                    result = agent.model.load(file_name=action[1], device=agent.device)
                    if result is not None:
                        n_games, loaded_time, opt_state, eps, rec = result
                        agent.n_games = n_games
                        agent.start_time = time.time() - loaded_time
                        agent.record = rec
                        if eps is not None:
                            agent.epsilon = eps

                        if opt_state is not None:
                            try:
                                agent.trainer.optimizer.load_state_dict(opt_state)
                            except Exception as e:
                                log(
                                    f"⚠️ Attention : Impossible de charger l'état de l'optimiseur ({e}). Le modèle continuera avec un nouvel optimiseur."
                                )

                        agent.trainer.target_model.load_state_dict(
                            agent.model.state_dict()
                        )
                        log(f"📂 Modèle chargé : {action[1]}")
                    else:
                        log(f"❌ Échec du chargement : {action[1]}")

        # --- Inférence ---
        state_tensor = agent.get_state_tensor(states)

        with torch.no_grad():
            prediction = agent.model(state_tensor)

        # Stratégie Epsilon-Greedy + Imitation Learning (Heuristique A*)
        # Quand imitation_mask=True, on suit l'heuristique, sinon on suit le modèle
        imitation_mask = np.random.random(N_ENVS) < agent.epsilon
        model_actions = torch.argmax(prediction, dim=1).cpu().numpy()
        greedy_actions = env.get_greedy_actions()
        final_moves = np.where(imitation_mask, greedy_actions, model_actions)

        # Injection de bruit aléatoire SÛR (5%) - Exploration sans suicide
        pure_random_mask = np.random.random(N_ENVS) < 0.05
        if np.any(pure_random_mask):
            # Récupérer les actions sûres pour chaque environnement
            safe_random_actions = env.get_safe_random_actions()
            final_moves = np.where(pure_random_mask, safe_random_actions, final_moves)

        # --- Physique ---
        next_states, rewards, dones, scores = env.step(final_moves)

        # --- Entraînement ---
        agent.remember_bulk(states, final_moves, rewards, next_states, dones)

        if agent.n_games > 100:
            if frames % TRAIN_EVERY == 0:
                agent.train_long_memory()
        else:
            agent.train_long_memory()

        states = next_states

        # --- Monitoring ---
        n_dones = np.sum(dones)
        if n_dones > 0:
            agent.n_games += n_dones
            agent.update_epsilon()
            dead_scores = scores[dones]
            for s in dead_scores:
                agent.all_scores.append(s)

        current_max = np.max(scores)
        if current_max > agent.record:
            agent.record = current_max
            log(f"🏆 Nouveau Record : {agent.record}")
            agent.model.save(
                n_games=agent.n_games,
                total_time=time.time() - agent.start_time,
                optimizer_state=agent.trainer.optimizer.state_dict(),
                epsilon=agent.epsilon,
                record=agent.record,
            )

        frames += 1
        if time.time() - t0 > 1.0:
            tps = frames * N_ENVS
            log(
                f"📊 {tps} TPS | Parties : {agent.n_games} | Eps : {agent.epsilon:.3f} | Rec : {agent.record}"
            )

            curr_mean = (
                sum(agent.all_scores) / len(agent.all_scores) if agent.all_scores else 0
            )
            agent.logger.log_stat(
                agent.n_games, agent.epsilon, agent.record, curr_mean, tps
            )

            frames = 0
            t0 = time.time()

        # Screenshots
        if dashboard.auto_screen_active:
            if time.time() - last_screen_time >= dashboard.screen_interval:
                dashboard._take_screenshot()
                last_screen_time = time.time()

        # Rendu
        activations = agent.model.get_activations(state_tensor[0].unsqueeze(0))
        game_surface = render_wrapper_0.render()
        dashboard.update_game(game_surface)
        dashboard.update_nn(agent.model, activations)
        dashboard.update_info(
            agent.n_games, time.time() - agent.start_time, agent.epsilon, agent.record
        )

        # Graphiques (mise à jour périodique)
        if agent.n_games - last_plot_update > 100:
            last_plot_update = agent.n_games
            if len(agent.all_scores) > 0:
                recent = list(agent.all_scores)[-100:]
                avg = sum(recent) / len(recent)

                # Mise à jour du scheduler avec le score moyen actuel
                agent.trainer.scheduler.step(avg)

                agent.check_stagnation(avg)

                plot_scores.append(avg)
                total_score += avg
                mean_score = total_score / len(plot_scores)
                plot_mean_scores.append(mean_score)

                dashboard.update_plots(plot_scores, plot_mean_scores, agent.record)
                dashboard.update_global_plot(list(agent.all_scores))
            dashboard.update()
        else:
            dashboard.update()


if __name__ == "__main__":
    train_vectorized()

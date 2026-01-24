"""
Snake Game - Conçu pour l'apprentissage par renforcement
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Initialisation de Pygame
pygame.init()

# ============================================================================
# CONFIGURATION
# ============================================================================

BLOCK_SIZE = 20
FPS = 60
GAME_SPEED = 500

# Couleurs
WHITE = (255, 255, 255)
BLACK = (20, 20, 30)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
GREEN_DARK = (40, 160, 40)
BLUE = (50, 100, 200)
GRAY = (40, 40, 50)

# Police
FONT = pygame.font.SysFont("arial", 25)
FONT_BIG = pygame.font.SysFont("arial", 50)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


# ============================================================================
# CLASSE PRINCIPALE DU JEU
# ============================================================================


class SnakeGame:
    """
    Classe principale du jeu Snake, adaptable pour le RL.
    """

    def __init__(self, width=640, height=480, render_mode=True, embedded=False):
        self.width = width
        self.height = height
        self.render_mode = render_mode

        # Configuration de l'affichage
        if self.render_mode:
            if embedded:
                self.display = pygame.Surface((self.width, self.height))
            else:
                self.display = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Snake Game - RL Ready")
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None

        # Pré-allocation buffer
        self.state_buffer = np.zeros(
            (3, self.height // BLOCK_SIZE, self.width // BLOCK_SIZE), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Réinitialise le jeu."""
        self.direction = Direction.RIGHT
        self._head_direction = Direction.RIGHT

        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()

        self.frame_iteration = 0
        self.move_counter = 0
        self.frames_per_move = FPS // GAME_SPEED

        self.grid_width = self.width // BLOCK_SIZE
        self.grid_height = self.height // BLOCK_SIZE

        return self.get_state()

    def get_grid_state(self):
        """Retourne l'état du jeu sous forme de grille 2D pour le CNN."""
        self.state_buffer.fill(0)

        # Canal 0: Corps du serpent (1.0)
        for point in self.snake:
            gx = int(point.x // BLOCK_SIZE)
            gy = int(point.y // BLOCK_SIZE)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                self.state_buffer[0, gy, gx] = 1.0

        # Canal 1: Tête + Direction (encodée)
        head_gx = int(self.head.x // BLOCK_SIZE)
        head_gy = int(self.head.y // BLOCK_SIZE)

        if 0 <= head_gx < self.grid_width and 0 <= head_gy < self.grid_height:
            direction_value = 0.0
            if self.direction == Direction.RIGHT:
                direction_value = 0.2
            elif self.direction == Direction.DOWN:
                direction_value = 0.4
            elif self.direction == Direction.LEFT:
                direction_value = 0.6
            elif self.direction == Direction.UP:
                direction_value = 0.8

            self.state_buffer[1, head_gy, head_gx] = direction_value

        # Canal 2: Nourriture
        food_gx = int(self.food.x // BLOCK_SIZE)
        food_gy = int(self.food.y // BLOCK_SIZE)
        if 0 <= food_gx < self.grid_width and 0 <= food_gy < self.grid_height:
            self.state_buffer[2, food_gy, food_gx] = 1.0

        return self.state_buffer.copy()

    def _place_food(self):
        """Place la nourriture aléatoirement."""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        """Retourne l'état (format vecteur simple)."""
        head = self.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger devant
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # Danger à droite
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # Danger à gauche
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Nourriture
            self.food.x < head.x,  # Gauche
            self.food.x > head.x,  # Droite
            self.food.y < head.y,  # Haut
            self.food.y > head.y,  # Bas
        ]

        return np.array(state, dtype=int)

    def _is_collision(self, point=None):
        if point is None:
            point = self.head

        # Collision Murs
        if (
            point.x > self.width - BLOCK_SIZE
            or point.x < 0
            or point.y > self.height - BLOCK_SIZE
            or point.y < 0
        ):
            return True

        # Collision Corps
        if point in self.snake[1:]:
            return True

        return False

    def play_step(self, action=None, events=None):
        """Exécute une frame de jeu."""
        # Gestion des Inputs
        if self.render_mode:
            if events is None:
                events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN and action is None:
                    if (
                        event.key == pygame.K_LEFT
                        and self._head_direction != Direction.RIGHT
                    ):
                        self.direction = Direction.LEFT
                    elif (
                        event.key == pygame.K_RIGHT
                        and self._head_direction != Direction.LEFT
                    ):
                        self.direction = Direction.RIGHT
                    elif (
                        event.key == pygame.K_UP
                        and self._head_direction != Direction.DOWN
                    ):
                        self.direction = Direction.UP
                    elif (
                        event.key == pygame.K_DOWN
                        and self._head_direction != Direction.UP
                    ):
                        self.direction = Direction.DOWN

        self.move_counter += 1
        reward = 0
        game_over = False

        # Logique de mouvement (Headless = instantané, Render = limité par FPS)
        if not self.render_mode or self.move_counter >= self.frames_per_move:
            self.move_counter = 0
            self.frame_iteration += 1

            # Application de l'action IA
            if action is not None:
                self._apply_action(action)

            # Déplacement
            self._move()
            self.snake.insert(0, self.head)

            # Vérification Collision ou Boucle infinie
            if self._is_collision() or (
                action is not None and self.frame_iteration > 100 * len(self.snake)
            ):
                game_over = True
                reward = -10
                return reward, game_over, self.score

            # Manger ou Avancer
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()

        # Mise à jour graphique
        if self.render_mode:
            self._update_ui()
            self.clock.tick(FPS)

        return reward, game_over, self.score

    def _apply_action(self, action):
        """Convertit l'action [Straight, Right, Left] en Direction."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Tout droit
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Droite
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Gauche

        self.direction = new_dir

    def _move(self):
        """Met à jour les coordonnées de la tête."""
        self._head_direction = self.direction
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):
        """Dessine le jeu."""
        self.display.fill(BLACK)

        # Grille
        for x in range(0, self.width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (0, y), (self.width, y))

        # Serpent avec Gradient
        for i, pt in enumerate(self.snake):
            ratio = 1 - (i / len(self.snake))
            brightness = max(0.2, ratio)

            color = (
                int(GREEN[0] * brightness),
                int(GREEN[1] * brightness),
                int(GREEN[2] * brightness),
            )

            pygame.draw.rect(
                self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display,
                BLACK,
                pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4),
                1,
            )

        # Nourriture
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display,
            WHITE,
            pygame.Rect(
                self.food.x + 4, self.food.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8
            ),
        )

        # Score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [10, 10])

        pygame.display.flip()

    def show_game_over(self):
        if not self.render_mode:
            return

        self.display.fill(BLACK)

        text = FONT_BIG.render("GAME OVER", True, RED)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.display.blit(text, text_rect)

        score_text = FONT.render(f"Score final: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(
            center=(self.width // 2, self.height // 2 + 20)
        )
        self.display.blit(score_text, score_rect)

        restart_text = FONT.render("JOUER: Espace | QUITTER: Esc", True, GRAY)
        restart_rect = restart_text.get_rect(
            center=(self.width // 2, self.height // 2 + 70)
        )
        self.display.blit(restart_text, restart_rect)

        pygame.display.flip()


# ============================================================================
# BOUCLE PRINCIPALE (MODE HUMAIN)
# ============================================================================


def main():
    game = SnakeGame()

    while True:
        while True:
            reward, game_over, score = game.play_step()
            if game_over:
                break

        game.show_game_over()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.reset()
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

        print(f"Partie terminée! Score: {score}")


# ============================================================================
# MOTEUR VECTORISÉ (POUR L'ENTRAÎNEMENT IA)
# ============================================================================


class VectorizedSnakeEnv:
    def __init__(self, n_envs=256, width=640, height=480, block_size=BLOCK_SIZE):
        self.n_envs = n_envs
        self.block_size = block_size
        self.w = width
        self.h = height
        self.grid_w = width // block_size
        self.grid_h = height // block_size
        self.max_len = self.grid_w * self.grid_h // 2

        # Directions: 0=Right, 1=Down, 2=Left, 3=Up
        self.move_vecs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)

        self.heads = np.zeros((n_envs, 2), dtype=np.int32)
        self.foods = np.zeros((n_envs, 2), dtype=np.int32)
        self.dirs = np.zeros(n_envs, dtype=np.int32)
        self.scores = np.zeros(n_envs, dtype=np.int32)
        self.dones = np.zeros(n_envs, dtype=bool)
        self.steps_since_apple = np.zeros(n_envs, dtype=np.int32)

        # Corps: Init avec -1
        self.bodies = np.full((n_envs, self.max_len, 2), -1, dtype=np.int32)
        self.lengths = np.full(n_envs, 3, dtype=np.int32)

        self.state_buffer = np.zeros(
            (n_envs, 4, self.grid_h, self.grid_w), dtype=np.float32
        )

        # Pré-calcul du canal Murs (statique)
        self.wall_channel = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.wall_channel[0, :] = 1.0  # Haut
        self.wall_channel[-1, :] = 1.0  # Bas
        self.wall_channel[:, 0] = 1.0  # Gauche
        self.wall_channel[:, -1] = 1.0  # Droite

        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = np.arange(self.n_envs)

        n = len(indices)
        if n == 0:
            return

        # Reset positions (Centre)
        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.heads[indices] = [cx, cy]
        self.dirs[indices] = 0  # Right

        # Reset Corps
        self.bodies[indices, :, :] = -1
        self.bodies[indices, 0] = [cx, cy]
        self.bodies[indices, 1] = [cx - 1, cy]
        self.bodies[indices, 2] = [cx - 2, cy]
        self.lengths[indices] = 3

        self.scores[indices] = 0
        self.dones[indices] = False
        self.steps_since_apple[indices] = 0

        self._spawn_foods(indices)

    def _spawn_foods(self, indices):
        n = len(indices)
        xs = np.random.randint(0, self.grid_w - 1, size=n)
        ys = np.random.randint(0, self.grid_h - 1, size=n)
        self.foods[indices] = np.stack([xs, ys], axis=1)

    def step(self, actions):
        """Action: [0=Straight, 1=Right, 2=Left]"""

        # Calcul distance AVANT mouvement (pour Reward Shaping)
        old_distances = np.abs(self.heads[:, 0] - self.foods[:, 0]) + np.abs(
            self.heads[:, 1] - self.foods[:, 1]
        )

        # Mise à jour Direction
        shifts = np.array([0, 1, -1])[actions]
        self.dirs = (self.dirs + shifts) % 4

        # Calcul Nouvelles Têtes
        current_moves = self.move_vecs[self.dirs]
        new_heads = self.heads + current_moves

        # Collisions Mur
        hit_wall = (
            (new_heads[:, 0] < 0)
            | (new_heads[:, 0] >= self.grid_w)
            | (new_heads[:, 1] < 0)
            | (new_heads[:, 1] >= self.grid_h)
        )

        # Collisions Corps
        hit_body = np.zeros(self.n_envs, dtype=bool)
        for i in range(self.n_envs):
            head = new_heads[i]
            length = self.lengths[i]
            body_part = self.bodies[i, : length - 1]
            if np.any(np.all(head == body_part, axis=1)):
                hit_body[i] = True

        # Manger Pomme
        ate_food = np.all(new_heads == self.foods, axis=1)

        # Famine
        self.steps_since_apple += 1
        self.steps_since_apple[ate_food] = 0
        starved = self.steps_since_apple > 150

        # Reward Shaping (APRÈS mouvement)
        safe_new_heads = np.clip(new_heads, [0, 0], [self.grid_w - 1, self.grid_h - 1])
        new_distances = np.abs(safe_new_heads[:, 0] - self.foods[:, 0]) + np.abs(
            safe_new_heads[:, 1] - self.foods[:, 1]
        )

        distance_reward = 0.3 * (old_distances - new_distances).astype(np.float32)

        # Calcul Rewards (Ratio 1:2 équilibré)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        rewards[ate_food] = 10.0  # Bonus (réduit pour équilibrer)
        rewards += distance_reward  # Shaping
        rewards += -0.01  # Time penalty

        self.dones = hit_wall | hit_body | starved
        rewards[self.dones] = -20.0  # Mort (réduit de -100 à -20)

        # Mise à jour Physique
        self.bodies[:, 1:] = self.bodies[:, :-1]
        self.bodies[:, 0] = new_heads

        self.lengths[ate_food] += 1
        self.scores[ate_food] += 1
        self._spawn_foods(np.where(ate_food)[0])
        self.heads = new_heads

        # Auto-Reset
        final_scores = self.scores.copy()
        final_dones = self.dones.copy()

        if np.any(self.dones):
            self.reset(np.where(self.dones)[0])

        return self.get_states(), rewards, final_dones, final_scores

    def get_states(self):
        """Génère le tenseur (N, 4, H, W) pour l'IA."""
        self.state_buffer.fill(0)
        batch_ids = np.arange(self.n_envs)

        # Canal 3: Murs (broadcast du canal pré-calculé)
        self.state_buffer[:, 3, :, :] = self.wall_channel

        # 1. Nourriture
        fx, fy = self.foods[:, 0], self.foods[:, 1]
        self.state_buffer[batch_ids, 2, fy, fx] = 1.0

        # 2. Têtes (Boussole)
        hx, hy = self.heads[:, 0], self.heads[:, 1]
        hx = np.clip(hx, 0, self.grid_w - 1)
        hy = np.clip(hy, 0, self.grid_h - 1)
        dir_values = (self.dirs + 1) * 0.2
        self.state_buffer[batch_ids, 1, hy, hx] = dir_values

        # 3. Corps (Gradient)
        for i in range(self.n_envs):
            length = self.lengths[i]
            b = self.bodies[i, :length]

            indices = np.arange(length, dtype=np.float32)
            values = 1.0 - (indices / length)

            bx = np.clip(b[:, 0], 0, self.grid_w - 1)
            by = np.clip(b[:, 1], 0, self.grid_h - 1)

            self.state_buffer[i, 0, by, bx] = values

        return self.state_buffer.copy()

    def get_greedy_actions(self):
        """Heuristique vectorisée : Mouvement sûr vers la pomme."""
        safe_mask = np.zeros((self.n_envs, 3), dtype=bool)
        distances = np.full((self.n_envs, 3), np.inf)

        for action in [0, 1, 2]:
            # Simulation direction
            shift = 0 if action == 0 else (1 if action == 1 else -1)
            potential_dirs = (self.dirs + shift) % 4
            vecs = self.move_vecs[potential_dirs]
            next_heads = self.heads + vecs

            # Check collisions
            hit_wall = (
                (next_heads[:, 0] < 0)
                | (next_heads[:, 0] >= self.grid_w)
                | (next_heads[:, 1] < 0)
                | (next_heads[:, 1] >= self.grid_h)
            )

            body_hits = np.any(
                np.all(self.bodies == next_heads[:, np.newaxis, :], axis=2), axis=1
            )
            is_safe = ~hit_wall & ~body_hits

            safe_mask[:, action] = is_safe

            # Calcul distance
            dists = np.abs(next_heads[:, 0] - self.foods[:, 0]) + np.abs(
                next_heads[:, 1] - self.foods[:, 1]
            )
            distances[np.where(is_safe), action] = dists[np.where(is_safe)]

        # Choix de la meilleure action
        best_actions = np.argmin(distances, axis=1)

        # Fallback si coincé
        chosen_is_safe = safe_mask[np.arange(self.n_envs), best_actions]
        unsafe_indices = np.where(~chosen_is_safe)[0]

        if len(unsafe_indices) > 0:
            for i in unsafe_indices:
                safe_acts = np.where(safe_mask[i])[0]
                if len(safe_acts) > 0:
                    best_actions[i] = np.random.choice(safe_acts)
                else:
                    best_actions[i] = np.random.randint(0, 3)  # Mort inévitable

        return best_actions


if __name__ == "__main__":
    main()

"""
Snake Game - Conçu pour l'intégration future d'une IA par renforcement
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
FPS = 60  # Taux de rafraîchissement visuel (fluidité de l'affichage)
GAME_SPEED = 10  # Vitesse du serpent (mouvements par seconde)

# Couleurs
WHITE = (255, 255, 255)
BLACK = (20, 20, 30)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
GREEN_DARK = (40, 160, 40)
BLUE = (50, 100, 200)
GRAY = (40, 40, 50)

# Police
FONT = pygame.font.SysFont('arial', 25)
FONT_BIG = pygame.font.SysFont('arial', 50)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

class Direction(Enum):
    """Énumération des directions possibles"""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


# ============================================================================
# CLASSE PRINCIPALE DU JEU
# ============================================================================

class SnakeGame:
    """
    Classe principale du jeu Snake.
    
    Conçue pour être facilement adaptable à l'apprentissage par renforcement:
    - Méthode `get_state()` pour obtenir l'état du jeu
    - Méthode `play_step()` retourne reward, game_over, score
    - Reset facile avec `reset()`
    """
    
    def __init__(self, width=640, height=480):
        """
        Initialise le jeu Snake.
        
        Args:
            width: Largeur de la fenêtre en pixels
            height: Hauteur de la fenêtre en pixels
        """
        self.width = width
        self.height = height
        
        # Configuration de l'affichage
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game - RL Ready')
        self.clock = pygame.time.Clock()
        
        # Initialisation du jeu
        self.reset()
    
    def reset(self):
        """
        Réinitialise le jeu à son état initial.
        Utile pour l'entraînement RL où on redémarre après chaque épisode.
        
        Returns:
            L'état initial du jeu
        """
        # Direction initiale
        self.direction = Direction.RIGHT
        
        # Position initiale du serpent (au centre)
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        # Score et nourriture
        self.score = 0
        self.food = None
        self._place_food()
        
        # Compteur de frames (pour détecter si le serpent tourne en rond)
        self.frame_iteration = 0
        
        # Compteur pour gérer la vitesse du jeu indépendamment des FPS
        self.move_counter = 0
        self.frames_per_move = FPS // GAME_SPEED  # Nombre de frames entre chaque mouvement
        
        return self.get_state()
    
    def _place_food(self):
        """Place la nourriture à une position aléatoire valide."""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # S'assurer que la nourriture n'est pas sur le serpent
        if self.food in self.snake:
            self._place_food()
    
    def get_state(self):
        """
        Retourne l'état actuel du jeu.
        
        Format adapté pour l'IA par renforcement:
        - Danger immédiat (devant, gauche, droite)
        - Direction actuelle
        - Position relative de la nourriture
        
        Returns:
            numpy array de 11 valeurs booléennes
        """
        head = self.snake[0]
        
        # Points adjacents à la tête
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Direction actuelle
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # Danger devant
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            
            # Danger à droite
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            
            # Danger à gauche
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            
            # Direction de mouvement
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position de la nourriture relative à la tête
            self.food.x < head.x,  # Nourriture à gauche
            self.food.x > head.x,  # Nourriture à droite
            self.food.y < head.y,  # Nourriture en haut
            self.food.y > head.y   # Nourriture en bas
        ]
        
        return np.array(state, dtype=int)
    
    def _is_collision(self, point=None):
        """
        Vérifie s'il y a collision.
        
        Args:
            point: Point à vérifier (par défaut: la tête)
            
        Returns:
            True si collision, False sinon
        """
        if point is None:
            point = self.head
            
        # Collision avec les murs
        if (point.x > self.width - BLOCK_SIZE or point.x < 0 or
            point.y > self.height - BLOCK_SIZE or point.y < 0):
            return True
        
        # Collision avec soi-même
        if point in self.snake[1:]:
            return True
        
        return False
    
    def play_step(self, action=None):
        """
        Exécute une étape du jeu.
        
        Args:
            action: Action à effectuer
                - None: Utilise les entrées clavier (mode humain)
                - [1, 0, 0]: Continuer tout droit
                - [0, 1, 0]: Tourner à droite
                - [0, 0, 1]: Tourner à gauche
        
        Returns:
            tuple: (reward, game_over, score)
                - reward: +10 si mange, -10 si meurt, 0 sinon
                - game_over: True si la partie est terminée
                - score: Score actuel
        """
        # 1. Collecter les entrées utilisateur (à chaque frame pour réactivité)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN and action is None:
                if event.key == pygame.K_LEFT:
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN
        
        # Incrémenter le compteur de mouvement
        self.move_counter += 1
        
        reward = 0
        game_over = False
        
        # Exécuter la logique du jeu seulement quand c'est le moment de bouger
        if self.move_counter >= self.frames_per_move:
            self.move_counter = 0
            self.frame_iteration += 1
            
            # 2. Appliquer l'action de l'IA (si fournie)
            if action is not None:
                self._apply_action(action)
            
            # 3. Déplacer le serpent
            self._move()
            self.snake.insert(0, self.head)
            
            # 4. Vérifier game over
            # Collision ou trop de frames sans manger (évite les boucles infinies)
            if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
                game_over = True
                reward = -10
                return reward, game_over, self.score
            
            # 5. Vérifier si mange la nourriture
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                # Retirer la queue (le serpent ne grandit pas)
                self.snake.pop()
        
        # 6. Mise à jour de l'affichage (à chaque frame pour fluidité)
        self._update_ui()
        self.clock.tick(FPS)
        
        return reward, game_over, self.score
    
    def _apply_action(self, action):
        """
        Applique une action de l'IA.
        
        Args:
            action: Liste de 3 éléments [straight, right, left]
        """
        # Ordre des directions dans le sens horaire
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            # Continuer tout droit
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Tourner à droite
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Tourner à gauche
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
    
    def _move(self):
        """Met à jour la position de la tête selon la direction."""
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
        """Met à jour l'affichage graphique."""
        self.display.fill(BLACK)
        
        # Dessiner la grille (optionnel, pour le style)
        for x in range(0, self.width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (0, y), (self.width, y))
        
        # Dessiner le serpent
        for i, pt in enumerate(self.snake):
            # Corps avec dégradé
            color = GREEN if i == 0 else GREEN_DARK
            pygame.draw.rect(self.display, color,
                           pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Bordure intérieure
            pygame.draw.rect(self.display, BLACK,
                           pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), 1)
        
        # Dessiner la nourriture
        pygame.draw.rect(self.display, RED,
                        pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, WHITE,
                        pygame.Rect(self.food.x + 4, self.food.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        
        # Afficher le score
        text = FONT.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [10, 10])
        
        pygame.display.flip()
    
    def show_game_over(self):
        """Affiche l'écran de game over."""
        self.display.fill(BLACK)
        
        # Texte Game Over
        text = FONT_BIG.render('GAME OVER', True, RED)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.display.blit(text, text_rect)
        
        # Score final
        score_text = FONT.render(f'Score final: {self.score}', True, WHITE)
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.display.blit(score_text, score_rect)
        
        # Instructions
        restart_text = FONT.render('Appuyez sur ESPACE pour rejouer ou ESC pour quitter', True, GRAY)
        restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 70))
        self.display.blit(restart_text, restart_rect)
        
        pygame.display.flip()


# ============================================================================
# BOUCLE PRINCIPALE (MODE HUMAIN)
# ============================================================================

def main():
    """Boucle principale pour jouer en mode humain."""
    game = SnakeGame()
    
    while True:
        # Boucle de jeu
        while True:
            reward, game_over, score = game.play_step()
            
            if game_over:
                break
        
        # Afficher game over
        game.show_game_over()
        
        # Attendre input pour rejouer ou quitter
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
        
        print(f'Partie terminée! Score: {score}')


if __name__ == '__main__':
    main()

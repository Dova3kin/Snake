import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
SIMULATION_BATCH_SIZE = 100  # Nombre de jeux en parallèle pour accélérer la collecte de données

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r)) or 
            (dir_l and game._is_collision(point_l)) or 
            (dir_u and game._is_collision(point_u)) or 
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or 
            (dir_d and game._is_collision(point_l)) or 
            (dir_l and game._is_collision(point_u)) or 
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or 
            (dir_u and game._is_collision(point_l)) or 
            (dir_r and game._is_collision(point_u)) or 
            (dir_l and game._is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_actions_batch(self, states):
        """
        Calculer les actions pour plusieurs états en une seule passe GPU/CPU.
        """
        # random moves: tradeoff exploration / exploitation
        epsilon = 80 - self.n_games
        final_moves = []
        
        # Convertir la liste de numpy arrays en un seul tenseur
        states_tensor = torch.tensor(np.array(states), dtype=torch.float)
        
        # Prédiction par lot
        with torch.no_grad():
            predictions = self.model(states_tensor)
        
        for i in range(len(states)):
            final_move = [0, 0, 0]
            if random.randint(0, 200) < epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                move = torch.argmax(predictions[i]).item()
                final_move[move] = 1
            final_moves.append(final_move)
            
        return final_moves

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    # Création des jeux parallèles
    # Le premier jeu a l'affichage activé pour la démo, les autres sont "headless"
    print(f"Initialisation de {SIMULATION_BATCH_SIZE} jeux parallèles...")
    games = [SnakeGame(render_mode=(i==0)) for i in range(SIMULATION_BATCH_SIZE)]
    # On garde une référence à part pour un jeu visuel optionnel si besoin, 
    # mais pour la vitesse pure on peut tous les mettre en False.
    # Pour voir ce qu'il se passe, on peut activer le rendu sur le jeu 0 tous les X frames.
    
    # États courants de tous les jeux
    current_states = [agent.get_state(g) for g in games]
    
    print("Début de l'entraînement...")
    while True:
        # Obtenir les actions pour tous les jeux en une fois
        final_moves = agent.get_actions_batch(current_states)
        
        new_states = []
        
        for i, game in enumerate(games):
            # Jouer l'étape pour chaque jeu
            reward, done, score = game.play_step(final_moves[i])
            state_new = agent.get_state(game)
            
            if i == 0 and game.render_mode:
                # Si on voulait rendre le jeu 0, mais ici render_mode=False pour tous pour la vitesse
                pass

            # Entraînement court (optionnel par étape, ou on garde juste pour le replay buffer)
            # Pour la performance, on peut skipper train_short_memory ou le faire par batch aussi
            # Ici on le fait un par un, ce qui peut ralentir en Python pur. 
            # Optimisation: stocker dans memory et faire un train_step sur le batch collecté
            
            agent.train_short_memory(current_states[i], final_moves[i], reward, state_new, done)
            agent.remember(current_states[i], final_moves[i], reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                
                # On ne lance le long training et le plot que quand le jeu 0 finit (pour éviter trop de fréquence)
                if i == 0:
                    agent.train_long_memory()

                    if score > record:
                        record = score
                        agent.model.save()

                    print('Game', agent.n_games, 'Score', score, 'Record', record)

                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            
            new_states.append(state_new)
        
        current_states = new_states

if __name__ == '__main__':
    train()

import heapq
import math
from game import BLOCK_SIZE, Point

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def heuristic(node_pos, end_pos):
    # Distance de Manhattan est souvent mieux pour une grille sans diagonales
    # Mais le code utilisateur demandait la distance Euclidienne, donc on garde Euclidienne ou on adapte.
    # Pour Snake (mouvements 4 directions), Manhattan (|dx| + |dy|) est plus logique.
    return abs(node_pos.x - end_pos.x) + abs(node_pos.y - end_pos.y)

def get_neighbors(current_node, game_width, game_height, block_size):
    # 4 directions: haut, bas, gauche, droite (pas de diagonales pour Snake)
    neighbors = [
        (0, -block_size),  # Up
        (0, block_size),   # Down
        (-block_size, 0),  # Left
        (block_size, 0)    # Right
    ]
    
    result = []
    current_x = current_node.position.x
    current_y = current_node.position.y
    
    for new_position in neighbors:
        node_position = Point(current_x + new_position[0], current_y + new_position[1])
        
        # Vérifier si dans les limites
        if (node_position.x > game_width - block_size or node_position.x < 0 or 
            node_position.y > game_height - block_size or node_position.y < 0):
            continue
            
        result.append(node_position)
        
    return result

def a_star_search(game):
    """
    Trouve le chemin le plus court vers la nourriture.
    Retourne le prochain mouvement (Point) ou None si pas de chemin.
    """
    start_node = Node(None, game.head)
    start_node.g = 0
    start_node.h = heuristic(game.head, game.food)
    start_node.f = start_node.g + start_node.h
    
    end_node_position = game.food
    
    open_list = []
    closed_set = set()
    
    heapq.heappush(open_list, start_node)
    
    # Limite pour éviter de trop ralentir si pas de chemin
    max_iterations = 1000 
    iterations = 0
    
    # Créer un set des positions du corps du serpent pour vérification rapide O(1)
    # On exclut la queue car elle va bouger
    snake_body = set(game.snake[:-1]) 
    
    while open_list and iterations < max_iterations:
        iterations += 1
        current_node = heapq.heappop(open_list)
        
        if current_node.position in closed_set:
            continue
            
        closed_set.add(current_node.position)
        
        # Objectif trouvé
        if current_node.position == end_node_position:
            # Reconstruire le chemin
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Retourne le chemin (start -> ... -> end)
        
        # Générer les voisins
        for neighbor_pos in get_neighbors(current_node, game.width, game.height, BLOCK_SIZE):
            if neighbor_pos in closed_set:
                continue
            
            # Vérifier collision avec le corps
            if neighbor_pos in snake_body:
                continue
                
            new_node = Node(current_node, neighbor_pos)
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node.position, end_node_position)
            new_node.f = new_node.g + new_node.h
            
            heapq.heappush(open_list, new_node)
            
    return None

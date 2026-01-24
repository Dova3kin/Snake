import heapq
from game import BLOCK_SIZE, Point

# ============================================================================
# ALGORITHME A* (A-STAR)
# ============================================================================


class Node:
    """Noeud pour l'algorithme A*."""

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
    """Distance de Manhattan."""
    return abs(node_pos.x - end_pos.x) + abs(node_pos.y - end_pos.y)


def get_neighbors(current_node, game_width, game_height, block_size):
    """Retourne les voisins valides (pas hors limites)."""
    neighbors = [
        (0, -block_size),  # Haut
        (0, block_size),  # Bas
        (-block_size, 0),  # Gauche
        (block_size, 0),  # Droite
    ]

    result = []
    current_x = current_node.position.x
    current_y = current_node.position.y

    for new_position in neighbors:
        node_position = Point(current_x + new_position[0], current_y + new_position[1])

        # Vérification limites
        if (
            node_position.x > game_width - block_size
            or node_position.x < 0
            or node_position.y > game_height - block_size
            or node_position.y < 0
        ):
            continue

        result.append(node_position)

    return result


def a_star_search(game):
    """
    Trouve le chemin le plus court vers la nourriture.
    Retourne la liste des Points du chemin ou None.
    """
    start_node = Node(None, game.head)
    start_node.g = 0
    start_node.h = heuristic(game.head, game.food)
    start_node.f = start_node.g + start_node.h

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    max_iterations = 100
    iterations = 0

    # Corps du serpent = obstacles
    snake_body = set(game.snake[:-1])

    while open_list and iterations < max_iterations:
        iterations += 1
        current_node = heapq.heappop(open_list)

        if current_node.position in closed_set:
            continue

        closed_set.add(current_node.position)

        # Objectif trouvé
        if current_node.position == game.food:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        # Voisins
        for neighbor_pos in get_neighbors(
            current_node, game.width, game.height, BLOCK_SIZE
        ):
            if neighbor_pos in closed_set or neighbor_pos in snake_body:
                continue

            new_node = Node(current_node, neighbor_pos)
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node.position, game.food)
            new_node.f = new_node.g + new_node.h

            heapq.heappush(open_list, new_node)

    return None

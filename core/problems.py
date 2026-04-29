import numpy as np
import random

class OptimizationProblem:
    """Clase base abstracta para los problemas."""
    def fitness(self, state): pass
    def get_neighbors(self, state): pass
    def get_random_neighbor(self, state): pass
    def get_initial_state(self): pass

class TSP(OptimizationProblem):
    def __init__(self, n_cities):
        self.n = n_cities
        # Generamos coordenadas (x, y) para poder visualizar el mapa
        self.coords = np.random.rand(n_cities, 2) * 100
        self.dist_matrix = self._create_dist_matrix()

    def _create_dist_matrix(self):
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                matrix[i,j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return matrix

    def fitness(self, state):
        dist = 0
        for i in range(self.n - 1):
            dist += self.dist_matrix[state[i], state[i+1]]
        dist += self.dist_matrix[state[-1], state[0]]
        return dist

    def get_neighbors(self, state):
        neighbors = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                n = list(state)
                n[i], n[j] = n[j], n[i]
                neighbors.append(n)
        return neighbors

    def get_random_neighbor(self, state):
        n = list(state)
        i, j = random.sample(range(self.n), 2)
        n[i], n[j] = n[j], n[i]
        return n

    def get_initial_state(self):
        state = list(range(self.n))
        random.shuffle(state)
        return state

class NQueens(OptimizationProblem):
    def __init__(self, n):
        self.n = n

    def fitness(self, state):
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def get_neighbors(self, state):
        neighbors = []
        for i in range(self.n):
            for val in range(self.n):
                if state[i] != val:
                    n = list(state)
                    n[i] = val
                    neighbors.append(n)
        return neighbors

    def get_random_neighbor(self, state):
        n = list(state)
        row = random.randint(0, self.n - 1)
        col = random.randint(0, self.n - 1)
        n[row] = col
        return n

    def get_initial_state(self):
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
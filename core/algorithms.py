import math
import random
import time

class HillClimbing:
    @staticmethod
    def run(problem):
        current = problem.get_initial_state()
        curr_fit = problem.fitness(current)
        history = [curr_fit]
        
        while True:
            neighbors = problem.get_neighbors(current)
            # Steepest Ascent: evaluamos todos y tomamos el mejor
            next_state = min(neighbors, key=lambda s: problem.fitness(s))
            next_fit = problem.fitness(next_state)
            
            if next_fit >= curr_fit:
                break
            
            current, curr_fit = next_state, next_fit
            history.append(curr_fit)
            
        return current, curr_fit, history

class SimulatedAnnealing:
    @staticmethod
    def run(problem, t_init=100, alpha=0.95, t_min=0.01):
        current = problem.get_initial_state()
        curr_fit = problem.fitness(current)
        
        best_state = current
        best_fit = curr_fit
        
        t = t_init
        history = [curr_fit]
        
        while t > t_min:
            neighbor = problem.get_random_neighbor(current)
            n_fit = problem.fitness(neighbor)
            
            delta = n_fit - curr_fit
            
            # Si es mejor, aceptamos. Si es peor, aceptamos con probabilidad
            if delta < 0 or random.random() < math.exp(-delta / t):
                current, curr_fit = neighbor, n_fit
                
                if curr_fit < best_fit:
                    best_state, best_fit = current, curr_fit
            
            history.append(curr_fit)
            t *= alpha # Enfriamiento geométrico
            
        return best_state, best_fit, history
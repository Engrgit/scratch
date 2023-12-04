# This is a sample Python script.
# This is the code implementation of firefely algorithm
 
import numpy as np


def objective_function(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2


def initialize_fireflies(num_fireflies, num_variables):
    return np.random.rand(num_fireflies, num_variables)


def move_fireflies(current_firefly, other_firefly, alpha, beta):
    r = np.linalg.norm(current_firefly - other_firefly)
    attractiveness = alpha * np.exp(-beta * r ** 2)
    delta = (np.random.rand(len(current_firefly)) - 0.5)
    return current_firefly + attractiveness * delta


def firefly_algorithm(num_fireflies, num_variables, max_generations, beta=1.0, alpha=0.2):
    fireflies = initialize_fireflies(num_fireflies, num_variables)

    for generations in range(max_generations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if objective_function(fireflies[i]) > objective_function(fireflies[j]):
                    fireflies[i] = move_fireflies(fireflies[i], fireflies[j], beta, alpha)


    # optimise for best solutions
    best_index = np.argmin([objective_function(f) for f in fireflies])
    best_solution = fireflies[best_index]

    return best_solution, objective_function(best_solution)


 
if __name__ == '__main__':
    num_fireflies = 20
    num_variables = 4
    max_generation = 50
    max_generations = 50

    best_solution, best_value = firefly_algorithm(num_fireflies, num_variables, max_generations)

    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_value)
 

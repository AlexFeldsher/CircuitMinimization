import Logic
import Search

"""
gates = [Gate('AND', lambda x, y: x and y, 2),
         Gate('OR', lambda x, y: x or y, 2),
         Gate('NOT', lambda x: not x, 1),
         Gate('NAND', lambda x, y: not (x and y), 2),
         Gate('XOR', lambda x, y: (x and not y) or (not x and y), 2)]
"""

gates = [Logic.Gate('AND', lambda x, y: x and y, 2),
         Logic.Gate('OR', lambda x, y: x or y, 2),
         Logic.Gate('NOT', lambda x: not x, 1)]

#gates = [Logic.Gate('NAND', lambda x, y: not (x and y), 2)]

def _example():
    half_adder = {(False, False): False,
                  (False, True): True,
                  (True, False): True,
                  (True, True): False}

    initial_state = Logic.State(gates, half_adder, 2)

    problem = Logic.Problem(initial_state)
    #print("solution with bfs", Search.breadth_first_graph_search(problem).state.state)
    print("solution with simulated annealing", Search.simulated_annealing(problem, Search.exp_schedule(20,0.005,10000)).state)
    #print("solution with hill climbing", Search.hill_climbing(problem).state)
    #print("solution with best first search", Search.best_first_graph_search(problem, Logic.cost2).state.state)


if __name__ == '__main__':
    _example()

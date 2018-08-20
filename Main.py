import Logic
import Search
import math

"""
gates = [Gate('AND', lambda x, y: x and y, 2),
         Gate('OR', lambda x, y: x or y, 2),
         Gate('NOT', lambda x: not x, 1),
         Gate('NAND', lambda x, y: not (x and y), 2),
         Gate('XOR', lambda x, y: (x and not y) or (not x and y), 2)]
"""


# can't use lambda function with multiprocessing

def _and(x, y):
    return x and y


def _or(x, y):
    return x or y


def _not(x):
    return not x


def _nand(x, y):
    return not (x and y)


def _xor(x, y):
    return (x and not y) or (not x and y)


gates = [Logic.Gate('AND', _and, 2),
         Logic.Gate('OR', _or, 2),
         Logic.Gate('NOT', _not, 1)]

gates_nand = [Logic.Gate('NAND', _nand, 2)]


half_adder = {(False, False): False,
                  (False, True): True,
                  (True, False): True,
                  (True, True): False}

mux2_1 = {(False, False, False): False,
          (False, False, True) : False,
          (False, True,  False): True,
          (False, True,  True) : True,
          (True,  False, False): False,
          (True,  False, True) : True,
          (True,  True,  False): False,
          (True,  True,  True) : True}


mux4_1 = {(False, False, False, False, False, False): False,
          (False, False, True, False, False, False): True,
          (False, False, False, True, False, False): False,
          (False, False, True, True, False, False): True,
          (False, False, False, False, True, False): False,
          (False, False, True, False, True, False): True,
          (False, False, False, True, True, False): False,
          (False, False, True, True, True, False): True,

          (False, False, False, False, False, True): False,
          (False, False, True, False, False, True): True,
          (False, False, False, True, False, True): False,
          (False, False, True, True, False, True): True,
          (False, False, False, False, True, True): False,
          (False, False, True, False, True, True): True,
          (False, False, False, True, True, True): False,
          (False, False, True, True, True, True): True,

          (True, False, False, False, False, False): False,
          (False, False, True, False, False, False): False,
          (False, False, False, True, False, False): True,
          (False, False, True, True, False, False): True,
          (False, False, False, False, True, False): False,
          (False, False, True, False, True, False): False,
          (False, False, False, True, True, False): True,
          (False, False, True, True, True, False): True,

          (False, False, False, False, False, True): False,
          (False, False, True, False, False, True): False,
          (False, False, False, True, False, True): True,
          (False, False, True, True, False, True): True,
          (False, False, False, False, True, True): False,
          (False, False, True, False, True, True): False,
          (False, False, False, True, True, True): True,
          (False, False, True, True, True, True): True,

          (False, True, False, False, False, False): False,
          (False, True, True, False, False, False): False,
          (False, True, False, True, False, False): False,
          (False, True, True, True, False, False): False,
          (False, True, False, False, True, False): True,
          (False, True, True, False, True, False): True,
          (False, True, False, True, True, False): True,
          (False, True, True, True, True, False): True,

          (False, True, False, False, False, True): False,
          (False, True, True, False, False, True): False,
          (False, True, False, True, False, True): False,
          (False, True, True, True, False, True): False,
          (False, True, False, False, True, True): True,
          (False, True, True, False, True, True): True,
          (False, True, False, True, True, True): True,
          (False, True, True, True, True, True): True,

          (True, True, False, False, False, False): False,
          (True, True, True, False, False, False): False,
          (True, True, False, True, False, False): False,
          (True, True, True, True, False, False): False,
          (True, True, False, False, True, False): False,
          (True, True, True, False, True, False): False,
          (True, True, False, True, True, False): False,
          (True, True, True, True, True, False): False,

          (True, True, False, False, False, True): True,
          (True, True, True, False, False, True): True,
          (True, True, False, True, False, True): True,
          (True, True, True, True, False, True): True,
          (True, True, False, False, True, True): True,
          (True, True, True, False, True, True): True,
          (True, True, False, True, True, True): True,
          (True, True, True, True, True, True): True}


def _example(truth_table, _gates, n_inputs):


    initial_state = Logic.State(_gates, truth_table, n_inputs, gate_limit=10)

    problem = Logic.Problem(initial_state)
    solution = Search.simulated_annealing(problem, Search.exp_schedule(1, 0.05, 2000))
    print("solution with simulated annealing", solution.state)

    #print("solution with bfs", Search.breadth_first_graph_search(problem).state.state)
    #print("solution with hill climbing", Search.hill_climbing(problem).state)
    #print("solution with best first search", Search.best_first_graph_search(problem, Logic.cost2).state.state)

    counter = 0
    for _input, _output in truth_table.items():
        evaluation = initial_state.evaluate(_input, solution.state)
        counter += 0 if evaluation != _output else 1
        print(_input, "expected:", _output, "got:", evaluation)
    print(counter, "out of", len(truth_table))

if __name__ == '__main__':
    _example(mux2_1, gates, 3)

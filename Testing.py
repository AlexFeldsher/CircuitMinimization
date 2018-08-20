from pyeda.inter import *
import Logic
import Search
import itertools


def _and(x, y):
    return x and y


def _or(x, y):
    return x or y


def _not(x):
    return not x

gates = [Logic.Gate('AND', _and, 2),
         Logic.Gate('OR', _or, 2),
         Logic.Gate('NOT', _not, 1)]


def truth_table_generator(num_of_variables):
    for seq in itertools.product([0,1], repeat=2**num_of_variables):
        truth_table = dict()
        inputs = list(itertools.product([0,1], repeat=num_of_variables))
        for i, output in enumerate(seq):
            truth_table[inputs[i]] = output
        yield truth_table, seq

def ai(truth_table, n_inputs):
    initial_state = Logic.State(gates, truth_table, n_inputs)
    problem = Logic.Problem(initial_state)
    solution = Search.simulated_annealing(problem, Search.exp_schedule(1, 0.05, 2000))
    counter = 0
    for _input, _output in truth_table.items():
        evaluation = initial_state.evaluate(_input, solution.state)
        counter += 0 if evaluation != _output else 1
        print(_input, "expected:", _output, "got:", evaluation)
    print(counter, "out of", len(truth_table))


if __name__ == '__main__':
    for t in truth_table_generator(3):
        print(t)
    X = exprvars('x', 3)
    for t in truth_table_generator(3):
        f = truthtable(X, ''.join(str(i) for i in t[1]))
        print("***************")
        print(f)
        print(truthtable2expr(f).to_cnf())
        print("----------------")
        ai(t[0], 3)


from pyeda.inter import *
from sympy.logic import SOPform
from sympy import symbols
import Logic
import Search
import itertools
import time
import random

DEBUG = True

def log(_method, *args):
    if not DEBUG:
        return
    msg = "{time:.5f}".format(time=time.perf_counter()) + "::" + _method.__qualname__ + "::"
    for arg in args:
        msg += str(arg) + ","
    print(msg)

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
    truth_tables = list()
    for seq in itertools.product([0,1], repeat=2**num_of_variables):
        if 1 not in seq or 0 not in seq:
            continue
        truth_table = dict()
        inputs = list(itertools.product([0,1], repeat=num_of_variables))
        for i, output in enumerate(seq):
            truth_table[inputs[i]] = output
        truth_tables.append((truth_table, seq))
    #log(truth_table_generator, truth_tables)
    return truth_tables


def ai(truth_table, n_inputs, limit=float('inf')):
    log(ai, "(upper gate limit, lower gate limit, height_limit)", limit, limit//3, 5)
    initial_state = Logic.State(gates, truth_table, n_inputs, gate_limit=limit, gate_l_limit=limit//3, height_limit=5)
    problem = Logic.Problem(initial_state)
    solution = Search.simulated_annealing(problem, Search.exp_schedule(1, 0.05, 2000))
    #solution = Search.simulated_annealing(problem, Search.exp_schedule(1, lam, T))
    log(ai,"solution with simulated annealing", solution.state)
    log(ai,"num of gates", solution.state.num_of_gates())
    counter = 0
    for _input, _output in truth_table.items():
        evaluation = initial_state.evaluate(_input, solution.state)
        counter += 0 if evaluation != _output else 1
        #log(ai, _input, "expected:", _output, "got:", evaluation)
    log(ai, counter, "out of", len(truth_table))


def quine_mcluskey(truth_table):
    x,y = symbols('x y')
    minterms = list()
    for key, val in truth_table.items():
        if val == 1:
            minterms.append(list(key))
    sol = SOPform([x,y],minterms)
    return str(expr(str(sol)))


if __name__ == '__main__':
    Logic.DEBUG = False
    Search.DEBUG = False
    truth_tables = truth_table_generator(2)
    #for t in truth_tables:
    #    print("111.111::main::", t)
    X = exprvars('x', 2)
    random.shuffle(truth_tables)
    for t in truth_tables:
        f = truthtable(X, ''.join(str(i) for i in t[1]))
        print("111.111::main::", t)
        print("111.111::main::", f)
        print("111.111::main::product of sums form", truthtable2expr(f).to_cnf())
        print("111.111::main::sum of products form (could be quinemcluskey form)", truthtable2expr(f).to_dnf())
        print("111.111::main::espresso minimization", espresso_exprs(truthtable2expr(f).to_dnf())[0])
        print("111.111::main::quinemcluskey form", quine_mcluskey(t[0]))
        expr_cnf = str(truthtable2expr(f).to_cnf())
        expr_dnf = str(truthtable2expr(f).to_dnf())
        quine_form = str(quine_mcluskey(t[0]))
        limit = min(expr_cnf.count(',') + expr_cnf.count('~'),
                    expr_dnf.count(',') + expr_dnf.count('~'))
                    #quine_form.count(',') + quine_form.count('~'))
        print("111.111::main::limit set to", limit+3)
        ai(t[0], 2, limit+3)

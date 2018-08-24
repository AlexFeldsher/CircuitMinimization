import Logic
import Search
import argparse
import itertools
from pyeda.inter import *


def _and(x, y):
    return x and y


def _or(x, y):
    return x or y


def _not(x):
    return not x


GATES = [Logic.Gate('AND', _and, 2),
         Logic.Gate('OR', _or, 2),
         Logic.Gate('NOT', _not, 1)]


def _create_truthtable(n_vars, truth_table):
    tt = dict()
    for i, seq in enumerate(itertools.product([0,1], repeat=n_vars)):
        tt[seq] = truth_table[i]
    return tt


def main(_args):
    tt = _create_truthtable(_args.n_vars[0], _args.truth_table[0])
    initial_state = Logic.State(GATES, tt, _args.n_vars[0], gate_limit=_args.u_gate_lim[0],
                                gate_l_limit=_args.l_gate_lim[0], height_limit=_args.height_lim[0])
    problem = Logic.Problem(initial_state)
    solution = Search.simulated_annealing(problem, Search.exp_schedule(_args.k[0], _args.lam[0], _args.limit[0]))
    print("\nSolution found with simulated annealing:", solution.state)

    X = exprvars('x', _args.n_vars[0])
    f = truthtable(X, _args.truth_table[0])
    print("Product of sums form", truthtable2expr(f).to_cnf())
    print("Sum of products form", truthtable2expr(f).to_dnf())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circuit minimization using Simulated Annealing')
    parser.add_argument('truth_table', nargs=1, help='Bits representing a truth table, see truthtable function at "https://pyeda.readthedocs.io/en/latest/boolalg.html#boolean-functions"')
    parser.add_argument('n_vars', nargs=1, type=int, help='The number of variables')
    parser.add_argument('k', nargs=1, type=float, help='Parameter k for the scheduler function k*exp(-lam*t)')
    parser.add_argument('lam', nargs=1, type=float, help='Parameter lam for the scheduler function k*exp(-lam*t)')
    parser.add_argument('limit', nargs=1, type=int, help='Limit the number of search iterations')
    parser.add_argument('u_gate_lim', nargs=1, type=int, help='Upper limit for the number of gates')
    parser.add_argument('l_gate_lim', nargs=1, type=int, help='Lower limit for the number of gates')
    parser.add_argument('height_lim', nargs=1, type=int, help='Height limit for the resulting circuit')
    args = parser.parse_args()

    main(args)

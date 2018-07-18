import os
import tempfile
import graphviz
import random
import re
import itertools
import numpy as np
import sympy
from sympy.parsing import sympy_parser
from sympy.printing.dot import dotprint
from sympy.logic import SOPform


class GP:
    BIN_OPS_MAP = {sympy.And: '&', sympy.Or: '|', sympy.Xor: '^'}  # isn't there a sympy mapping somewhere? anyway, extend as needed

    def __init__(self, num_vars=3, bin_ops=(sympy.And, sympy.Or), target_tt=None):
        """
        Create a genetic programming class that will help solve the circuit minimization problem

        :param num_vars: how many vars are we dealing with (i.e. truth table height is 2^num_vars)
        :param bin_ops: not necessarily binary - And(x,y,z) is legal as well. sympy.Not is always used.
        :param target_tt: some truth table, given as a np.array with shape (2**n, ) and dtype=np.bool. if None a random table is generated.
        """
        self.num_vars = num_vars

        # symbols we use in the problem
        self.syms = sympy.symbols('s:'+str(num_vars))  # creates a tuple of (s0, s1, ...s{NUM_VARS-1})

        # operations we use in the problem
        self.bin_ops = bin_ops
        self.ops = bin_ops + (sympy.Not, )

        # FIXME: "private" some of these
        # some precalced stuff for the functions
        self.str_syms = set(map(str, self.syms))
        self.tstr_syms = tuple(self.str_syms)
        self.bin_ops_chars = [self.BIN_OPS_MAP[op] for op in self.bin_ops]
        self.str_bin_ops = set(map(str, self.bin_ops))
        self.str_ops = set(map(str, self.ops))
        self.tstr_bin_ops = tuple(self.str_bin_ops)
        self.tstr_ops = tuple(self.bin_ops)
        self.or_op_regex = '|'.join(self.str_ops)

        # fitness and srepr structure array dtype
        dtype = np.dtype([('fitness', '<f4'), ('srepr', '<U')])  # FIXME: consider setting length to <U, maybe <U3000?
        # population of sreprs and their fitness. this changes throughout the run
        self.population = np.array([], dtype=dtype)
        # population = np.recarray([], dtype=dtype)

        # some truth table, given as a np.array with shape (2**n, )
        self.target_tt = target_tt if target_tt is None else np.random.randint(2, size=2**3, dtype=np.bool)

        self.tt_vars = list(itertools.product([0, 1], repeat=self.num_vars))  # [(0, 0, 0), (0, 0, 1), (0, 1, 0), ...]
        self.tt_vars_lists = list(zip(*self.tt_vars))  # [(0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 1, 1, 0, 0, 1, 1), (0, 1, 0, 1, 0, 1, 0, 1)]

    def simple_random_srepr(self, nsymbols=5):  # FIXME: some other val
        """there's probably some better way."""  # TODO: investigate
        ret = ''
        for i in range(nsymbols):
            if random.random() < 0.5:
                ret += '~'
            ret += random.choice(self.tstr_syms)
            ret += random.choice(self.bin_ops_chars)

        ret += random.choice(self.tstr_syms)
        ret = sympy.srepr(sympy_parser.parse_expr(ret))
        return ret

    @staticmethod
    def find_cpi(s, start):
        """Finds the index 'end' of the closing paren that makes s[start:end+1] legal paren-wise"""
        count = 0
        for i in range(start, len(s)):
            if s[i] == '(':
                count += 1
            elif s[i] == ')':
                count -= 1
                if count == 0:
                    return i

    def mutate(self, srepr):
        """
        Perform a mutation on srepr.
        Currently: pick a node at random. then:
        - if it's a symbol, pick one of these at random:
            - replace it with another symbol at random
            - replace it with a new random tree (hardcoded small depth for now)
        - elif it's a Not, remove it
        - else, it's an other op, which has to be binary. pick one of these at random:
            - replace it with another random binary op
            - remove it (including the whole tree) and pluck in a random symbol

        :param srepr: the result of a call to sympy.srepr,
        such as "Or(Symbol('x'), And(Symbol('y'), Or(Symbol('x'), Not(Symbol('z')))))"
        :return: the mutated srepr
        """

        # *** pick opening paren at random ***

        # using re because it's a bit faster
        op_parens_idxs = [m.start() for m in re.finditer('\(', srepr)]
        ropi = random.choice(op_parens_idxs)  # random opening paren index

        # *** determine type of node (Symbol, And, Or...) ***

        # the node is given before ropi but after one of ' (', so find the prev one
        # of ' (' (or the beginning of the string)
        before_node = [m.start() for m in re.finditer('[ (]\w*$', srepr[:ropi])]
        bni = before_node[-1] if before_node else -1  # 'else' happens if ropi is first '('
        node = srepr[bni+1:ropi]

        if 'Symbol' in node:  # replace with another one randomly
            cpi = self.find_cpi(srepr, ropi)
            symbol = srepr[ropi+2:cpi-1]
            if random.random() < 0.5:
                another = random.choice(tuple(self.str_syms-{symbol}))
                ret = srepr[:ropi+1] + "'" + another + "'" + srepr[cpi:]
            else:
                ret = srepr[:bni+1] + self.simple_random_srepr() + srepr[cpi+1:]
        elif 'Not' in node:  # remove Not
            cpi = self.find_cpi(srepr, ropi)
            ret = srepr[:bni+1] + srepr[ropi+1:cpi] + srepr[cpi+1:]
        else:  # it's a binary op
            if random.random() < 0.5:  # replace with another one randomly
                another = random.choice(tuple(self.str_bin_ops-{node}))
                ret = srepr[:bni+1] + another + srepr[ropi:]
            else:  # replace tree with a random symbol
                cpi = self.find_cpi(srepr, ropi)
                symbol = random.choice(self.tstr_syms)
                ret = srepr[:bni+1] + "Symbol('" + symbol + "')" + srepr[cpi+1:]

        # assert ret != srepr  # might not be true because the generated random tree might be the same
        return ret

    def pick_random_rep(self, s, pick_op_prob=0.9):
        iops = [m.start() for m in re.finditer(self.or_op_regex, s)]  # indices of ops
        isyms = [m.start() for m in re.finditer('Symbol', s)]  # indices of symbols
        pick_from = iops if iops and random.random() < pick_op_prob else isyms
        rep = random.choice(pick_from)

        cpi = self.find_cpi(s, rep)
        return rep, cpi

    def recombine(self, s1, s2):
        """
        1. Pick an op/symbol from each parent, with larger prob for ops
        2. Switch them (resulting in 2 offspring). maybe pick 1?

        :param s1: first srepr parent
        :param s2: second srepr parent
        :return: offspring srepr
        """
        PICK_OP_PROB = 0.9

        rep1, cpi1 = self.pick_random_rep(s1, PICK_OP_PROB)
        rep2, cpi2 = self.pick_random_rep(s2, PICK_OP_PROB)

        off1 = s1[:rep1] + s2[rep2:cpi2+1] + s1[cpi1+1:]
        off2 = s2[:rep2] + s1[rep1:cpi1+1] + s2[cpi2+1:]

        return off1, off2

    def fitness(self, srepr):
        """
        Calculate the fitness of a srepr as a linear combination of:
        - number of agreeing lines with the original truth table
        - number of gates (weight is negative)

        :param srepr: the result of a call to sympy.srepr,
            such as "Or(Symbol('x'), And(Symbol('y'), Or(Symbol('x'), Not(Symbol('z')))))"
        :return: fitness
        """
        WEIGHT_TT = 0.8  # TODO: pick
        WEIGHT_NB = -0.3  # TODO: pick

        # number of agreeing lines
        f = sympy.lambdify(self.syms, srepr)
        tt_f = f(*self.tt_vars_lists)

        '''
        %%timeit
        F = sympy.Lambda(syms, srepr)
        tt_f = np.array([F(*r) for r in tt_vars])
        2.71 ms ± 257 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        
        %%timeit
        f = sympy.lambdify(syms, srepr)
        tt_f = f(*tt_vars_lists)
        1.65 ms ± 73 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        '''

        # number of gates
        nb_gates = len(re.findall(self.or_op_regex, srepr))

        score = WEIGHT_TT * np.count_nonzero(tt_f == tt) + WEIGHT_NB * nb_gates
        return score

    @staticmethod
    def show(srepr, pref='dp_'):
        dp = dotprint(sympy_parser.parse_expr(srepr))
        tfn = tempfile.mktemp(suffix='.png', prefix=pref)
        graphviz.Source(dp, filename=os.path.splitext(tfn)[0], format='png').view()


def tt_to_sympy_minterms(tt):
    """
    Convert a truth table to sympy minterms.
    See http://docs.sympy.org/latest/modules/logic.html

    :param tt: np.array of shape (2**n, ) and dtype bool
    :return: sympy minterms
    """
    num_vars = int(np.log2(tt.size).round())
    trues = np.where(tt)[0]
    minvars = []
    for t in trues:
        minvars.append([int(b) for b in format(t, '0%db' % num_vars)])

    return minvars


if __name__ == '__main__':
    # just for some playing around:
    tt = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.bool)
    g = GP(num_vars=3, bin_ops=(sympy.And, sympy.Or), target_tt=tt)
    fn = g.syms[0] | (g.syms[1] & (~g.syms[2] | g.syms[0]))
    srepr = sympy.srepr(fn)
    mt = tt_to_sympy_minterms(g.target_tt)
    sop_form = SOPform(g.syms, mt)
    # with sympy evaluate(False):  # cancel automatic evaluation

from lru import LRU
import argparse
import datetime
import sys
import os
import time
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
import multiprocessing

global logfile


class Util:
    BIN_OPS_MAP = {sympy.And: '&', sympy.Or: '|',
                   sympy.Xor: '^'}  # isn't there a sympy mapping somewhere? anyway, extend as needed

    def __init__(self, num_vars=3, bin_ops=(sympy.And, sympy.Or), target_tt=None, mutate_prob=0.05, weight_num_agree=20, weight_num_gates=-0.1, nsymbols=5):
        """
        Util class for sake of multiprocessing - avoiding passing the large self.population on each call

        :param num_vars: how many vars are we dealing with (i.e. truth table height is 2^num_vars)
        :param bin_ops: not necessarily binary - And(x,y,z) is legal as well. sympy.Not is always used.
        :param target_tt: some truth table, given as a np.array with shape (2**n, ) and dtype=np.bool. if None a random table is generated.
        :param mutate_prob: probability of child mutation
        :param weight_num_agree: score weight of number of agreeing lines of srepr and target_tt
        :param weight_num_gates: score weight of number of gates
        """
        self.num_vars = num_vars

        # symbols we use in the problem
        self.syms = sympy.symbols('s:' + str(num_vars))  # creates a tuple of (s0, s1, ...s{NUM_VARS-1})

        # operations we use in the problem
        self.bin_ops = bin_ops
        self.ops = bin_ops + (sympy.Not,)

        # FIXME: "private" some of these
        # some precalced stuff for the functions
        self.str_syms = set(map(str, self.syms))
        self.tstr_syms = tuple(self.str_syms)
        self.bin_ops_chars = [self.BIN_OPS_MAP[op] for op in self.bin_ops]
        self.str_bin_ops = set(map(str, self.bin_ops))
        self.str_ops = set(map(str, self.ops))
        self.tstr_bin_ops = tuple(self.str_bin_ops)
        self.tstr_ops = tuple(self.bin_ops)
        self.or_op_regex = re.compile('|'.join(self.str_ops))

        # some truth table, given as a np.array with shape (2**n, )
        self.target_tt = target_tt if target_tt is not None else np.random.randint(2, size=2 ** 3, dtype=np.bool)

        self.tt_vars = list(itertools.product([0, 1], repeat=self.num_vars))  # [(0, 0, 0), (0, 0, 1), (0, 1, 0), ...]
        self.tt_vars_lists = list(
            zip(*self.tt_vars))  # [(0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 1, 1, 0, 0, 1, 1), (0, 1, 0, 1, 0, 1, 0, 1)]

        self.mutate_prob = mutate_prob
        self.weight_num_agree = weight_num_agree
        self.weight_num_gates = weight_num_gates
        self.nsymbols = nsymbols

        # create process pool
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())

    def __getstate__(self):
        """allow pickling (used by multiprocessing)"""
        s = time.time()
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        # print('_GETSTATE UTIL', time.time()-s)
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def simple_random_srepr(self):
        NOT_PROB = 0.5
        PAREN_PROB = 0.2
        ret = ''
        n_paren = 0
        for i in range(self.nsymbols):
            if random.random() < NOT_PROB:
                ret += '~'
            if random.random() < PAREN_PROB:
                ret += '('
                n_paren += 1
            ret += random.choice(self.tstr_syms)
            if n_paren > 0 and random.random() < PAREN_PROB:
                ret += ')'
                n_paren -= 1
            ret += random.choice(self.bin_ops_chars)

        ret += random.choice(self.tstr_syms)
        ret += ')' * n_paren
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
        node = srepr[bni + 1:ropi]

        if 'Symbol' in node:  # replace with another one randomly
            cpi = self.find_cpi(srepr, ropi)
            symbol = srepr[ropi + 2:cpi - 1]
            if random.random() < 0.5:
                another = random.choice(tuple(self.str_syms - {symbol}))
                ret = srepr[:ropi + 1] + "'" + another + "'" + srepr[cpi:]
            else:
                ret = srepr[:bni + 1] + self.simple_random_srepr() + srepr[cpi + 1:]
        elif 'Not' in node:  # remove Not
            cpi = self.find_cpi(srepr, ropi)
            ret = srepr[:bni + 1] + srepr[ropi + 1:cpi] + srepr[cpi + 1:]
        else:  # it's a binary op
            if random.random() < 0.5:  # replace with another one randomly
                another = random.choice(tuple(self.str_bin_ops - {node}))
                ret = srepr[:bni + 1] + another + srepr[ropi:]
            else:  # replace tree with a random symbol
                cpi = self.find_cpi(srepr, ropi)
                symbol = random.choice(self.tstr_syms)
                ret = srepr[:bni + 1] + "Symbol('" + symbol + "')" + srepr[cpi + 1:]

        # assert ret != srepr  # might not be true because the generated random tree might be the same
        return ret

    def pick_random_rep(self, s, pick_op_prob=0.9):
        iops = [m.start() for m in self.or_op_regex.finditer(s)]  # indices of ops
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

        off1 = s1[:rep1] + s2[rep2:cpi2 + 1] + s1[cpi1 + 1:]
        off2 = s2[:rep2] + s1[rep1:cpi1 + 1] + s2[cpi2 + 1:]

        return off1, off2

    def fitness(self, srepr):
        """
        Calculate the fitness of a srepr as a linear combination of:
        - number (or percentage) of agreeing lines with the original truth table
        - number of gates (weight is negative)

        :param srepr: the result of a call to sympy.srepr,
            such as "Or(Symbol('x'), And(Symbol('y'), Or(Symbol('x'), Not(Symbol('z')))))"
        :return: fitness
        """

        # number of agreeing lines
        f = sympy.lambdify(self.syms, srepr)
        tt_f = f(*self.tt_vars_lists)
        # I think it's better to use number of lines instead of percentage since larger tt will also need larger circuits
        nz = np.count_nonzero(tt_f == self.target_tt)
        per = nz / len(tt_f)

        '''
        %%timeit
        F = sympy.Lambda(syms, srepr)
        tt_f = np.array([F(*r) for r in tt_vars])
        2.71 ms  257 µs per loop (mean  std. dev. of 7 runs, 100 loops each)

        %%timeit
        f = sympy.lambdify(syms, srepr)
        tt_f = f(*tt_vars_lists)
        1.65 ms  73 µs per loop (mean  std. dev. of 7 runs, 1000 loops each)
        '''

        # number of gates
        nb_gates = len(self.or_op_regex.findall(srepr))

        score = self.weight_num_agree * nz + self.weight_num_gates * nb_gates
        return score, per, nb_gates

    @staticmethod
    def show(srepr, pref='dp_'):
        dp = dotprint(sympy_parser.parse_expr(srepr))
        tfn = tempfile.mktemp(suffix='.png', prefix=pref)
        graphviz.Source(dp, filename=os.path.splitext(tfn)[0], format='png').view()

    def init_one(self, _=0):
        """just here so we can parallelize population init"""
        some_srepr = self.simple_random_srepr()
        some_fitness, some_accuracy, some_ng = self.fitness(some_srepr)
        return some_fitness, some_accuracy, some_ng, some_srepr

    def create_next_gen(self, parents_sreprs_couple):
        """just here so we can parallelize run"""
        child0, child1 = self.recombine(parents_sreprs_couple[0], parents_sreprs_couple[1])
        if random.random() < self.mutate_prob:
            child0 = self.mutate(child0)
        if random.random() < self.mutate_prob:
            child1 = self.mutate(child1)

        return child0, child1

    def precentage_of_tt(self, srepr):
        f = sympy.lambdify(self.syms, srepr)
        tt_f = f(*self.tt_vars_lists)
        return np.count_nonzero(tt_f == self.target_tt) / len(tt_f)

    def naive_srepr(self):
        mt = tt_to_sympy_minterms(self.target_tt)
        naive_srepr = 'Or('
        for term in mt:
            conj = 'And('
            for i, t in enumerate(term):
                if not t:
                    conj += 'Not('
                conj += sympy.srepr(self.syms[i])
                if not t:
                    conj += ')'
                conj += ','
            conj += ')'
            naive_srepr += conj + ','
        naive_srepr += ')'
        naive_srepr = sympy.srepr(sympy_parser.parse_expr(naive_srepr))  # get rid of commas
        score, per, nb_gates = self.fitness(naive_srepr)

        return score, per, nb_gates, naive_srepr


class GP:
    LOW_FITNESS = -2 ** 31

    def __init__(self, pop_size=5000, size_next_gen=300, lucky_per=0.10, unique_pop=False, add_naive=False, **kwargs):
        """
        Create a genetic programming class that will help solve the circuit minimization problem

        :param pop_size: population max size
        :param size_next_gen: num of offspring in each generation
        :param kwargs: passed on to Util
        """
        self.pop_size = pop_size
        self.size_next_gen = size_next_gen

        self.size_best_parents = (1 - lucky_per) * self.size_next_gen
        if self.size_best_parents % 2 == 1: self.size_best_parents += 1
        self.size_lucky_parents = self.size_next_gen - self.size_best_parents

        self.unique_pop = unique_pop
        self.add_naive = add_naive
        self.util = Util(**kwargs)
        self.cache = LRU(10000)

    def _init_population(self, init_pop_size=1000):
        # fitness and srepr structure array dtype
        pop_dtype = np.dtype([('fitness', '<f4'), ('accuracy', '<f4'), ('numgates', '<i4')])

        # population of sreprs and their fitness. this changes throughout the run
        self.population = np.array([self.LOW_FITNESS] * self.pop_size, dtype=pop_dtype).view(np.recarray)
        self.sreprs = [None] * self.pop_size

        if self.add_naive:
            init_pop_size -= 1
        ret = g.util.pool.map(g.util.init_one, range(init_pop_size))

        for i, r in enumerate(ret):
            r_fitness, r_accuracy, r_numgates, r_srepr = r
            self.population[i] = r_fitness, r_accuracy, r_numgates
            self.cache[r_srepr] = r_fitness, r_accuracy, r_numgates
            self.sreprs[i] = r_srepr

        if self.add_naive:
            r_fitness, r_accuracy, r_numgates, r_srepr = self.util.naive_srepr()
            self.population[i+1] = r_fitness, r_accuracy, r_numgates
            self.cache[r_srepr] = r_fitness, r_accuracy, r_numgates
            self.sreprs[i+1] = r_srepr


    def run(self, num_generations=10, init_pop_size=1000):
        global logfile
        print('init population...', end=' ', flush=True)
        print('init population...', end=' ', flush=True, file=logfile)
        self._init_population(init_pop_size=init_pop_size)
        print('done')
        print('done', file=logfile)

        print('starting search')
        print('starting search', file=logfile)
        for generation in range(num_generations):
            # print current status
            logstr = 'generation {0}/{1}   best fitness: {2}     best accuracy: {3}'.format(
                generation,
                num_generations,
                self.population[self.population.fitness.argmax()],
                self.population[self.population.accuracy.argmax()])

            print(logstr)
            print(logstr, file=logfile)

            # get parents probability distribution - fitter sreprs are better picks as parents
            fitness = self.population.fitness.copy()
            real_min = np.min(fitness[fitness > self.LOW_FITNESS])
            fitness[fitness == self.LOW_FITNESS] = real_min  # so no chance they'll be taken
            nz_fitness = fitness - real_min
            # each parent couple generates 2 children
            best_parents = np.random.choice(np.arange(self.population.shape[0]),
                                            size=(int(self.size_best_parents / 2), 2), p=nz_fitness / nz_fitness.sum())
            lucky_parents = np.random.choice(np.arange(self.population.shape[0]),
                                             size=(int(self.size_lucky_parents / 2), 2))
            parents = np.concatenate((best_parents, lucky_parents))

            parents_sreprs = [(self.sreprs[x[0]], self.sreprs[x[1]]) for x in parents
                              if None not in [self.sreprs[x[0]], self.sreprs[x[1]]]]

            # generate offspring and replace the weak samples in the population
            worst_indices = np.argpartition(self.population.fitness, self.size_next_gen)[:self.size_next_gen]
            next_gen = g.util.pool.map(g.util.create_next_gen, parents_sreprs)
            flat_next_gen = [s for sc in next_gen for s in sc]
            next_gen_noncached_srepr = [srepr for srepr in flat_next_gen if srepr not in self.cache]
            next_gen_cached_srepr = [srepr for srepr in flat_next_gen if srepr in self.cache]
            next_gen_noncached_fitness = g.util.pool.map(g.util.fitness, next_gen_noncached_srepr)

            to_update = dict(zip(next_gen_noncached_srepr, next_gen_noncached_fitness))

            # next_gen_sreprs = flat_next_gen if not self.unique_pop else next_gen_noncached_srepr
            next_gen_fitness = [(srepr,) + to_update[srepr] for srepr in next_gen_noncached_srepr]
            if not self.unique_pop:
                next_gen_fitness += [(srepr,) + self.cache[srepr] for srepr in next_gen_cached_srepr]

            self.cache.update(to_update)

            for wi, children in zip(worst_indices, next_gen_fitness):
                # child0, c0_fitness, c0_accuracy, c0_numgates, child1, c1_fitness, c1_accuracy, c1_numgates = children
                child, c_fitness, c_accuracy, c_numgates = children
                self.population[wi] = (c_fitness, c_accuracy, c_numgates)
                self.sreprs[wi] = child

        print(file=logfile)
        print('Finished!', file=logfile)

    def print_best(self, file=sys.stdout):
        best_fitness_index = self.population.fitness.argmax()
        best_accuracy_index = self.population.accuracy.argmax()
        print('best fitness: {0}     best accuracy: {1}'.format(
            self.population[best_fitness_index],
            self.population[best_accuracy_index]
        ), file=file)
        print(file=file)
        print(self.sreprs[best_fitness_index], file=file)
        print(file=file)
        print(self.sreprs[best_accuracy_index], file=file)


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


parser = argparse.ArgumentParser()

parser.add_argument('--add_naive', type=bool, default=False, help='start with naive solution in population')
parser.add_argument('--pop_size', type=int, default=20000, help='total population size - worst are removed if too many')
parser.add_argument('--init_pop_size', type=int, default=2000, help='initial population size')
parser.add_argument('--size_next_gen', type=int, default=300,
                    help='how many offspring in each generation. MUST BE EVEN')
parser.add_argument('--lucky_per', type=float, default=0.05,
                    help='what percentage of next gen size is taken *uniformly* at random')
parser.add_argument('--num_generations', type=int, default=20000, help='total number of generation to run')
parser.add_argument('--mutate_prob', type=float, default=0.05, help='probability of mutation of children')

parser.add_argument('--weight_num_gates', type=float, default=-0.1, help='score weight of number of gates')
parser.add_argument('--weight_num_agree', type=float, default=10,
                    help='score weight of number of agreeing lines of srepr and target_tt')

parser.add_argument('--unique_pop', type=bool, default=False,
                    help='whether the population does not reinsert existing members')

opt = parser.parse_args()

assert opt.size_next_gen % 2 == 0, 'size_next_gen must be even (each parent couple produces 2 offspring)'

args = vars(opt)


def print_args(args, file=sys.stdout):
    print('------------ Options -------------', file=file)
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)), file=file)
    print('-------------- End ----------------', file=file)


print_args(args)

if __name__ == '__main__':
    global logfile
    # just for some playing around:
    # tt = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.bool)
    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    logdir = os.path.join(dirname, 'logs')
    try:
        os.mkdir(logdir)
    except FileExistsError:
        pass
    logname = os.path.join(logdir, '{0:%Y-%m-%d_%H-%M-%S}.txt'.format(datetime.datetime.now()))
    print(logname)
    logfile = open(logname, 'w')
    print_args(args, logfile)

    # tt = np.array([0, 1, 1, 0, 0, 0, 1, 1] * 128, dtype=np.bool)
#    tt = np.random.randint(2, size=128, dtype=bool)
#    tt = np.random.randint(2, size=8, dtype=bool)
    tt = np.array([False, True, True, True, False, False, True, True])
    print(tt.tolist())
    print(tt.tolist(), file=logfile)
    g = GP(num_vars=int(np.log2(tt.shape[0])),
           unique_pop=opt.unique_pop,
           bin_ops=(sympy.And, sympy.Or),
           target_tt=tt,
           pop_size=opt.pop_size,
           size_next_gen=opt.size_next_gen,
           lucky_per=opt.lucky_per,
           weight_num_gates=opt.weight_num_gates,
           weight_num_agree=opt.weight_num_agree,
           add_naive=opt.add_naive,
    )
    fn = g.util.syms[0] | (g.util.syms[1] & (~g.util.syms[2] | g.util.syms[0]))
    srepr = sympy.srepr(fn)
    mt = tt_to_sympy_minterms(g.util.target_tt)
    sop_form = SOPform(g.util.syms, mt)

    try:
        g.run(num_generations=opt.num_generations, init_pop_size=opt.init_pop_size)
        # g.run()
    except KeyboardInterrupt:
        pass
    g.print_best(logfile)

    # g.util.pool.terminate()  # terminated on this process's death, or so it seems

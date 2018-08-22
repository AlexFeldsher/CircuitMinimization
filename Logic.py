# cython: language_level=3, boundscheck=False
import itertools
import multiprocessing as mp
import time
import types
from copy import deepcopy
from typing import TypeVar, Sequence, Mapping, Union, List, Set
import math
from operator import itemgetter

import Search

DEBUG = True

CircuitInput = int  # For readability
GateInputs = TypeVar('Params', List[Union['Gate', 'CircuitInput']], 'CircuitInput')


def log(_method, *args):
    if not DEBUG:
        return
    msg = "{time:.5f}".format(time=time.perf_counter()) + "::" + _method.__qualname__ + "::"
    for arg in args:
        msg += str(arg) + ","
    print(msg)


# Identity function used for "identity" gates
def _IDENTITY(x):
    return x


class Gate:
    """ A class representing a gate in a circuit
        Basically a node in a graph """

    def __init__(self, gate_type: str, logic: types.FunctionType, n_inputs: int, representation: str = None):
        """ :param gate_type: the name of the gate, i.e. "NAND", "OR", ..
            :param logic: a function that given boolean input will output the result of the gate logic
            :param n_inputs: the number of inputs the gate has
            :param representation: string representation of the graph """
        self._type = gate_type
        self._logic = logic
        self._n_inputs = n_inputs
        self._gate_inputs = None
        self._height = None  # the distance of the gate from the input
        self._representation = representation

    # ------------------------ Special methods ------------------------
    def __repr__(self) -> str:
        if self.gate_inputs is not None:
            self._update_repr()
        return self._representation

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: 'Gate') -> bool:
        return self.__repr__() == repr(other)

    def __lt__(self, other: 'Gate') -> bool:
        # TODO: define what it means circuit1 < circuit2
        return self.height < other.height

    def __copy__(self) -> 'Gate':
        gate = Gate(self.type, self.logic, self.n_inputs)
        gate.gate_inputs = self._gate_inputs
        return gate

    def __deepcopy__(self, memodict={}) -> 'Gate':
        new_params = list()
        if type(self.gate_inputs) is CircuitInput:
            new_params = self.gate_inputs
        else:
            for gate in self.gate_inputs:
                new_params.append(deepcopy(gate))
        gate = Gate(self.type, self.logic, self.n_inputs)
        gate.gate_inputs = new_params
        return gate

    def __hash__(self) -> int:
        """ Sort depends on the hashing of the string representation """
        return hash(self.__repr__())

    def __len__(self) -> int:
        # TODO: why is len define and should it be the number of gates in the circuit?
        return self.height

    def __getitem__(self, item: 'Gate') -> 'Gate':
        # TODO: have to implement slicing for genetic algorithms
        # returns gate == item in the circuit
        if type(item) is Gate:
            for gate in self:
                if gate == item:
                    return gate
        raise KeyError('Gate not found in circuit')

    def __iter__(self) -> 'Gate':
        """ Iterate over the circuit where this gate is the root"""
        yield self
        if type(self.gate_inputs) is not CircuitInput:
            for gate in self.gate_inputs:
                yield from gate

    # -------------- Setters and Getters for class attributes ----------------
    @property
    def gate_inputs(self) -> GateInputs:
        return self._gate_inputs

    @gate_inputs.setter
    def gate_inputs(self, gates: GateInputs) -> None:
        """ Set the gate inputs
            :param gates: the gates to set as input """
        if gates is None:
            return
        if type(gates) is CircuitInput:
            self._gate_inputs = gates
        else:
            self._gate_inputs = list(gates)
        self.update_height()

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, height: int) -> None:
        self._height = height

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def logic(self) -> types.FunctionType:
        return self._logic

    @logic.setter
    def logic(self, logic: types.FunctionType) -> None:
        self._logic = logic

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, gate_type: str) -> None:
        self._type = gate_type

    # ------------------- Public methods ---------------------
    def sort(self):
        """ Defines a standard tree structure to prevent isomorphic circuits to be considered not equal
            Works only because the Gate hash method hashes a string representation of the circuit """
        if type(self.gate_inputs) is CircuitInput:
            return
        for _input in self.gate_inputs:
            _input.sort()
        sorted_inputs = sorted(self.gate_inputs, key=lambda x: repr(x))

        self.gate_inputs = sorted_inputs   # unzips the list
        self._update_repr()

    def inputs_iter(self):
        """ Iterator for the gate inputs """
        # TODO: unnecessary method
        if type(self.gate_inputs) is CircuitInput:
            return self.gate_inputs
        for _input in self.gate_inputs:
            yield _input

    def evaluate(self, bools: Sequence[bool]) -> bool:
        """ Evaluate the gate with the given input
            :param bools: a collection of Booleans
            :return: True or False """
        if len(bools) < self.n_inputs or len(bools) > self.n_inputs:
            raise ValueError('Trying to evaluate with wrong number of parameters')
        return self.logic(*bools)

    def can_replace(self, gate: 'Gate') -> bool:
        """ :return: true if the given gate can replace this gate and vice versa"""
        return gate.n_inputs == self.n_inputs

    def update_height(self) -> int:
        """ updates the height recursively down the tree """
        if type(self.gate_inputs) is CircuitInput:
            self.height = 0
        else:
            self.height = max([p.update_height() for p in self.gate_inputs]) + 1
        return self.height

    def num_of_gates(self) -> int:
        """ The number of unique gates in the circuit """
        if type(self.gate_inputs) is CircuitInput:
            return 0
        set_of_gates = set()
        for gate in self:
            if gate.type != 'ID' and type(gate) is not CircuitInput:
                set_of_gates.add(repr(gate))
        #log(self.num_of_gates, self, set_of_gates, len(set_of_gates))
        return len(set_of_gates)

    # --------------------- Private methods -------------------------------
    def _update_repr(self) -> None:
        """ Sort depends on the identity gate to be invisible in the representation """
        if type(self.gate_inputs) is CircuitInput:
            self._representation = str(self.gate_inputs)
            return
        self._representation = self.type + '('
        for gate in self.gate_inputs:
            self._representation += repr(gate) + ','
        self._representation = self._representation[:-1] + ')'


Gate.IDENTITY_GATE = Gate(gate_type='ID', logic=_IDENTITY, n_inputs=1)     # define Gate class attribute
# must be done outside of the class because it holds an instance of the same class


class State:

    def __init__(self, gates: Sequence[Gate], truth_table: Mapping[Sequence[bool], bool], n_inputs: int,
                 initial_state: Union[Gate, None] = None, gate_limit=float('inf'), gate_l_limit=0, height_limit=float('inf')):
        # log(self.__init__, initial_state, truth_table, n_inputs)
        self.state = initial_state
        self.n_inputs = n_inputs

        self.outputs = set()
        self.get_outputs(self.state)

        self.truth_table = truth_table
        self.gates = gates

        self.gate_limit = gate_limit
        self.gate_l_limit = gate_l_limit
        self.height_limit = height_limit

        # initialize the state with an identity gate for the first circuit input
        if self.state is None:
            gate = Gate.IDENTITY_GATE.__copy__()
            gate.gate_inputs = 0
            self.state = gate

    def __lt__(self, other: 'State') -> bool:
        return self.state.height < other.state.height

    def __eq__(self, other: 'State') -> bool:
        return repr(self.state) == repr(other.state)

    def __hash__(self) -> int:
        return hash(repr(self.state))

    def __copy__(self) -> 'State':
        return State(self.gates, self.truth_table, self.n_inputs, self.state, self.gate_limit, self.gate_l_limit, self.height_limit)

    def get_outputs(self, gate: Gate) -> None:
        """ Every gate output in the given circuit is an 'output' that the next gate can interface with """
        for i in range(self.n_inputs):
            id_gate = Gate.IDENTITY_GATE.__copy__()
            id_gate.gate_inputs = i
            self.outputs.add(id_gate)
        if gate is None:
            return


        self.outputs.add(gate)
        # log(self.get_outputs, gate, gate.height, self.outputs)
        for gate in gate.inputs_iter():
            self.get_outputs(gate)

    def get_actions(self) -> Sequence[Gate]:
        return self.gates

    def get_successors(self, action: Gate) -> Set[Gate]:
        """ Returns a list of successors
            Uses multiprocessing to create circuits simultaneously
            :param action: the gate type we want to add to the current circuit """
        successors = set()
        processes = list()
        queue = mp.Queue()   # shared thread safe data structure
        if self.state.num_of_gates() < self.gate_limit:
            if self.state.height < self.height_limit:
                p1 = mp.Process(target=attach_gate_at_end, args=(self, action, queue))
                processes.append(p1)
                p1.start()
            p2 = mp.Process(target=insert_gate_into_circuit, args=(self, action, queue))
            processes.append(p2)
            p2.start()

        p3 = mp.Process(target=replace_gate_with_action, args=(self, action, queue))
        processes.append(p3)
        p3.start()

        if self.state.num_of_gates() > self.gate_l_limit:
            p4 = mp.Process(target=remove_gate_from_circuit, args=(self, action, queue))
            processes.append(p4)
            p4.start()

        while True in [p.is_alive() for p in processes]:
            while not queue.empty():
                s = queue.get()
                if s.num_of_gates() < self.gate_limit\
                        and (s.num_of_gates() > self.gate_l_limit or s.num_of_gates() > self.state.num_of_gates())\
                        and s.height < self.height_limit:
                    #log(self.get_successors, "height of successor", s.height)
                    #log(self.get_successors, "successor", s)
                    successors.add(s)

        for p in processes:
            p.join()

        # sorting by gate distance from the input (for bfs)
        log(self.get_successors, "(action, height, state)", action.type, self.state.height, self.state)

        return successors

    def apply_action(self, action: Gate) -> 'State':
        state = self.__copy__()
        state.state = action
        state.get_outputs(action)
        # log(self.apply_action, action, state)
        return state

    def evaluate(self, _input: Sequence[bool], gate: Union[Gate, CircuitInput]):
        """ Evaluates the given gate with the given input """
        if type(gate) is not Gate:
            return _input[gate]
        params = [False for _ in range(gate.n_inputs)]
        if type(gate.gate_inputs) is CircuitInput:
            params[0] = self.evaluate(_input, gate.gate_inputs)
        else:
            for i in range(gate.n_inputs):
                params[i] = self.evaluate(_input, gate.gate_inputs[i])
        return gate.evaluate(params)

    def is_goal(self) -> bool:
        """ Returns true if self.state evaluates correct for the entire truth table """
        #log(self.is_goal)
        for _input, output in self.truth_table.items():
            if self.evaluate(_input, self.state) != output:
                #log(self.is_goal, self.evaluate(_input, self.state), "<>", output)
                return False
        return True


class Problem(Search.Problem):

    def __init__(self, initial: State, goal=None):
        log(self.__init__, initial, goal)
        super().__init__(initial, goal)

    def actions(self, state: State) -> Set[Gate]:
        successors = set()
        actions = state.get_actions()
        for action in actions:
            succ = state.get_successors(action)
            successors = successors.union(succ)
        #log(self.actions, state, successors)
        return successors

    def result(self, state: State, action: Gate) -> State:
        res = state.apply_action(action)
        # log(self.result, state, action, res)
        return res

    def goal_test(self, state: State) -> bool:
        res = state.is_goal()
        log(self.goal_test, state.state, res)
        return res

    def value(self, state: State) -> int:
        # simulated annealing loo
        counter = 0
        for _input, output in state.truth_table.items():
            if state.evaluate(_input, state.state) == output:
                counter += 1

        #score = math.sqrt((counter/len(state.truth_table))**2 + (1/state.state.num_of_gates())**2)
        #score = math.exp(counter/(len(state.truth_table)/3)) + (2/(3*state.state.num_of_gates() + 1))
        #score = ((counter - len(state.truth_table)/2)**2)/(len(state.truth_table)**2) - (state.state.num_of_gates()/state.gate_limit)
        #score = math.log(counter/len(state.truth_table)) - math.exp(state.state.num_of_gates()/state.gate_limit)

        log(self.value, state.state, "counter",  counter)
        log(self.value, state.state, "num of gates", state.state.num_of_gates())
        log(self.value, state.state, "height", state.state.height)

        # used to get to the initial state where the defined limitations hold
        if state.state.num_of_gates() < state.gate_l_limit or state.state.height > state.height_limit:
            return -1000

        score = math.exp(counter/len(state.truth_table)) - (state.state.num_of_gates()-state.gate_l_limit)/(state.gate_limit-state.gate_l_limit)
        log(self.value, state.state, "score", score)
        return score


# ------------------- State get_successors helper functions ------------------------------
def attach_gate_at_end(state: State, action: Gate, queue: mp.Queue) -> None:
    # all the possible output combinations where "action" is the top gate
    for combination in itertools.combinations(state.outputs, action.n_inputs - 1):
        # check whether both we have repeating inputs
        if state.state not in combination and len(set(combination)) == len(combination):
            gate = action.__copy__()
            gate.gate_inputs = [state.state, *combination]
            gate.sort()
            queue.put(gate)


def insert_gate_into_circuit(state: State, action: Gate, queue: mp.Queue) -> None:
    # Inserts a gate in the middle of the circuit
    for gate in state.state:
        # can't go behind the input gates (identity gates)
        if gate.type != Gate.IDENTITY_GATE.type:
            root = deepcopy(state.state)  # create a deep copy of the circuit
            gate_copy = root[gate]  # get a pointer to the gate in the copied circuit
            # possible gates to interface with are at height[gate_copy] - 1
            possible_outputs = filter(lambda x: x.height < gate_copy.height, state.outputs)
            new_gate = action.__copy__()  # create the new gate
            # add all the possible combinations to the new gate inputs
            for combination in itertools.combinations(possible_outputs, new_gate.n_inputs):
                new_gate.gate_inputs = combination

                # add the new gate as an input in the gate copy (all possible position)
                for i in range(len(gate_copy.gate_inputs)):
                    tmp = gate_copy.gate_inputs[i]
                    gate_copy.gate_inputs[i] = new_gate
                    root.update_height()
                    root.sort()
                    queue.put(deepcopy(root))

                    gate_copy.gate_inputs[i] = tmp


def replace_gate_with_action(state: State, action: Gate, queue: mp.Queue) -> None:
    # Replacing gates with action
    for gate in state.state:
        # can't replace the inputs (identity gates) and can only replace gates with the same number of inputs
        if gate.type != Gate.IDENTITY_GATE.type and gate.n_inputs == action.n_inputs and gate.type != action.type:
            root = deepcopy(state.state)
            gate_copy = root[gate]
            # Convert the gate to the action gate type
            gate_copy.logic = action.logic
            gate_copy.type = action.type
            root.update_height()
            root.sort()

            queue.put(root)


def remove_gate_from_circuit(state: State, action: Gate, queue: mp.Queue) -> None:
    # Remove gate of type action from the circuit
    for gate in state.state:
        # find a gate that is the same type as action and isn't the last gate in the circuit
        if gate.type != Gate.IDENTITY_GATE.type and gate.type == action.type and gate != state.state:
            prev = None
            root = deepcopy(state.state)  # create a copy of the circuit
            for g in root:
                # if the previous gate is the parent of the gate we are looking for
                if g == gate and type(prev.gate_inputs) is not CircuitInput and g in prev.gate_inputs:
                    prev_inputs = [p for p in prev.gate_inputs if
                                   p != g]  # save the inputs that aren't the gate we are removing
                    possible_outputs = filter(lambda x: x.height < g.height - 1, state.outputs)
                    # create circuits with the gate removed
                    for output in possible_outputs:
                        prev.gate_inputs = prev_inputs + list(output)
                        root.update_height()
                        root.sort()
                        queue.put(deepcopy(root))
                root = deepcopy(state.state)  # create a copy of the circuit
                prev = g
        elif gate == state.state and gate.type == action.type:
            for _input in gate.inputs_iter():
                queue.put(deepcopy(_input))


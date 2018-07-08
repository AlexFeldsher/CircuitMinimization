# cython: language_level=3, boundscheck=False
import itertools
import multiprocessing as mp
import time
import types
from copy import deepcopy
from typing import TypeVar, Sequence, Mapping, Union, List, Set

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


def _id(x):
    return x


class Gate:
    """ A class representing a gate in a circuit
        Basically a node in a graph """

    def __init__(self, name: str, logic: types.FunctionType, n_inputs: int, representation: str = None):
        """ :param name: the name of the gate, i.e. "NAND", "OR", ..
            :param logic: a function that given boolean input will output the result of the gate logic
            :param n_inputs: the number of inputs the gate has
            :param representation: string representation of the graph """
        self.name = name
        self.logic = logic
        self.n_inputs = n_inputs
        self.gate_inputs = None
        self.height = None  # the distance of the gate from the input
        self.representation = representation

    def __repr__(self) -> str:
        if self.gate_inputs is not None:
            self._update_repr()
        return self.representation

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: 'Gate') -> bool:
        # TODO: equality should check if the circuits are identical, AND(NAND(0,1),1) == AND(1,NAND(1,0))
        return self.__repr__() == repr(other)

    def __lt__(self, other: 'Gate') -> bool:
        # TODO: define what it means circuit1 < circuit2
        return self.height < other.height

    def __copy__(self) -> 'Gate':
        gate = Gate(self.name, self.logic, self.n_inputs)
        gate.set_params(self.gate_inputs)
        return gate

    def __deepcopy__(self, memodict={}) -> 'Gate':
        new_params = list()
        if type(self.gate_inputs) is CircuitInput:
            new_params = self.gate_inputs
        else:
            for gate in self.gate_inputs:
                new_params.append(deepcopy(gate))
        gate = Gate(self.name, self.logic, self.n_inputs)
        gate.set_params(new_params)
        return gate

    def __hash__(self) -> int:
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

    def set_params(self, gates: GateInputs) -> None:
        """ Set the gate inputs
            :param gates: the gates to set as input """
        if gates is None:
            return
        if type(gates) is CircuitInput:
            self.gate_inputs = gates
        else:
            self.gate_inputs = list(gates)
        self.update_height()

    def get_params(self) -> GateInputs:
        return self.gate_inputs

    def params_iter(self):
        """ Iterator for the gate inputs """
        if type(self.gate_inputs) is CircuitInput:
            return self.gate_inputs
        for param in self.gate_inputs:
            yield param

    def evaluate(self, bools: Sequence[bool]) -> bool:
        """ Evaluate the gate with the given input
            :param bools: a collection of Booleans
            :return: True or False """
        if len(bools) < self.n_inputs or len(bools) > self.n_inputs:
            raise ValueError('Trying to evaluate with wrong number of parameters')
        return self.logic(*bools)

    def can_replace(self, gate: 'Gate') -> bool:
        """ :return: true if the given gate can replace this gate """
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
            set_of_gates.add(repr(gate))
        return len(set_of_gates)

    def _update_repr(self) -> None:
        if type(self.gate_inputs) is CircuitInput:
            self.representation = str(self.gate_inputs)
            return
        self.representation = self.name + '('
        for gate in self.gate_inputs:
            self.representation += repr(gate) + ','
        self.representation = self.representation[:-1] + ')'


class State:
    identity_gate = Gate(name='I', logic=_id, n_inputs=1)

    def __init__(self, gates: Sequence[Gate], truth_table: Mapping[Sequence[bool], bool], n_inputs: int,
                 initial_state: Union[Gate, None] = None):
        # log(self.__init__, initial_state, truth_table, n_inputs)
        self.state = initial_state
        self.n_inputs = n_inputs

        self.outputs = set()
        self.get_outputs(self.state)

        self.truth_table = truth_table
        self.gates = gates

        if self.state is None:
            gate = self.identity_gate.__copy__()
            gate.set_params(0)
            self.state = gate

    def __lt__(self, other: 'State') -> bool:
        return self.state.height < other.state.height

    def __eq__(self, other: 'State') -> bool:
        return repr(self.state) == repr(other.state)

    def __hash__(self) -> int:
        return hash(repr(self.state))

    def __copy__(self) -> 'State':
        return State(self.gates, self.truth_table, self.n_inputs, self.state)

    def get_outputs(self, gate: Gate) -> None:
        for i in range(self.n_inputs):
            id_gate = self.identity_gate.__copy__()
            id_gate.set_params(i)
            self.outputs.add(id_gate)
        if gate is None:
            return

        self.outputs.add(gate)
        # log(self.get_outputs, gate, gate.height, self.outputs)
        for gate in gate.params_iter():
            self.get_outputs(gate)

    def get_actions(self) -> Sequence[Gate]:
        return self.gates

    def get_successors(self, action: Gate) -> Set[Gate]:
        """ Returns a list of successors """
        successors = set()
        successors.add(self.state)  # add the previous circuit for backtracking

        queue = mp.Queue()   # shared thread safe data structure
        p1 = mp.Process(target=attach_gate_at_end, args=(self, action, queue))
        p2 = mp.Process(target=insert_gate_into_circuit, args=(self, action, queue))
        p3 = mp.Process(target=replace_gate_with_action, args=(self, action, queue))
        p4 = mp.Process(target=remove_gate_from_circuit, args=(self, action, queue))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        # extract successors from queue
        while p1.is_alive() or p2.is_alive() or p3.is_alive() or p4.is_alive():
            while not queue.empty():
                successors.add(queue.get())

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        '''
        # sorting by gate distance from the input (for bfs)
        sorted_list = list(successors.difference({self.state}))
        sorted_list.sort()
        '''
        log(self.get_successors, "(action, height, state, successors)", action.name, self.state.height, self.state,
            successors)

        return successors

    def apply_action(self, action: Gate) -> 'State':
        state = self.__copy__()
        state.state = action
        state.get_outputs(action)
        # log(self.apply_action, action, state)
        return state

    def evaluate(self, _input: Sequence[bool], gate: Union[Gate, CircuitInput]):
        if type(gate) != Gate:
            return _input[gate]
        params = [False for _ in range(gate.n_inputs)]
        if type(gate.gate_inputs) is CircuitInput:
            params[0] = self.evaluate(_input, gate.gate_inputs)
        else:
            for i in range(gate.n_inputs):
                params[i] = self.evaluate(_input, gate.gate_inputs[i])
        return gate.evaluate(params)

    def is_goal(self) -> bool:
        log(self.is_goal)
        for _input, output in self.truth_table.items():
            if self.evaluate(_input, self.state) != output:
                log(self.is_goal, self.evaluate(_input, self.state), "<>", output)
                return False
        return True


def attach_gate_at_end(state: State, action: Gate, queue: mp.Queue) -> None:
    # all the possible output combinations where "action" is the top gate
    for combination in itertools.combinations(state.outputs, action.n_inputs - 1):
        # check whether both we have repeating inputs
        if state.state not in combination and len(set(combination)) == len(combination):
            gate = action.__copy__()
            gate.set_params([state.state, *combination])
            queue.put(gate)


def insert_gate_into_circuit(state: State, action: Gate, queue: mp.Queue) -> None:
    # Inserts a gate in the middle of the circuit
    for gate in state.state:
        # can't go behind the input gates (identity gates)
        if gate.name != state.identity_gate.name:
            root = deepcopy(state.state)  # create a deep copy of the circuit
            gate_copy = root[gate]  # get a pointer to the gate in the copied circuit
            # possible gates to interface with are at height[gate_copy] - 1
            possible_outputs = filter(lambda x: x.height < gate_copy.height - 1, state.outputs)
            new_gate = action.__copy__()  # create the new gate
            # add all the possible combinations to the new gate inputs
            for combination in itertools.combinations(possible_outputs, new_gate.n_inputs):
                new_gate.set_params(combination)

                # add the new gate as an input in the gate copy (all possible position)
                for i in range(len(gate_copy.gate_inputs)):
                    tmp = gate_copy.gate_inputs[i]
                    gate_copy.gate_inputs[i] = new_gate
                    root.update_height()
                    queue.put(deepcopy(root))

                    gate_copy.gate_inputs[i] = tmp


def replace_gate_with_action(state: State, action: Gate, queue: mp.Queue) -> None:
    # Replacing gates with action
    for gate in state.state:
        # can't replace the inputs (identity gates) and can only replace gates with the same number of inputs
        if gate.name != state.identity_gate.name and gate.n_params == action.n_inputs:
            root = deepcopy(state.state)
            gate_copy = root[gate]
            # Convert the gate to the action gate type
            gate_copy.logic = action.logic
            gate_copy.name = action.name
            root.update_height()

            queue.put(root)


def remove_gate_from_circuit(state: State, action: Gate, queue: mp.Queue) -> None:
    # Remove gate of type action from the circuit
    for gate in state.state:
        # find a gate that is the same type as action and isn't the last gate in the circuit
        if gate.name != state.identity_gate.name and gate.name == action.name and gate != state.state:
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
                        queue.put(deepcopy(root))
                root = deepcopy(state.state)  # create a copy of the circuit
                prev = g


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
        log(self.actions, state, successors)
        return successors

    def result(self, state: State, action: Gate) -> State:
        res = state.apply_action(action)
        # log(self.result, state, action, res)
        return res

    def goal_test(self, state: State) -> bool:
        res = state.is_goal()
        log(self.goal_test, state, res)
        return res

    def value(self, state: State) -> int:
        # TODO: define a real state value
        # simulated annealing loo
        counter = 0
        for _input, output in state.truth_table.items():
            if state.evaluate(_input, state.state) == output:
                counter += 1
        return (counter - len(state.truth_table)) * 100 - state.state.num_of_gates()

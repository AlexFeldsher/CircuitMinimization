import unittest
import Logic

Logic.DEBUG = False


def _and(x, y):
    return x and y


def _or(x, y):
    return x or y


def _not(x):
    return not x


class TestGate(unittest.TestCase):

    def test_gate_init(self):
        gate_type = 'testGate'
        gate_logic = _and
        n_inputs = 2
        gate = Logic.Gate(gate_type, gate_logic, n_inputs)

        self.assertEqual(gate.type, gate_type)
        self.assertEqual(gate.logic, gate_logic)
        self.assertEqual(gate.n_inputs, n_inputs)

    def test_equality(self):
        gate_type = 'testGate'
        gate_type2 = 'testGate2'
        gate_logic = _and
        n_inputs = 2
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 1
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 2

        gate1 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate2 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate3 = Logic.Gate(gate_type2, gate_logic, n_inputs)
        gate1.gate_inputs = [id_gate1, id_gate2]
        gate2.gate_inputs = [id_gate1, id_gate2]
        gate3.gate_inputs = [id_gate1, id_gate2]

        self.assertTrue(gate1 == gate2)
        self.assertFalse(gate1 == gate3)
        self.assertFalse(gate2 == gate3)

    def test_height(self):
        gate_type = 'testGate'
        gate_logic = _and
        n_inputs = 2
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 1
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 2

        gate = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate.gate_inputs = [id_gate1, id_gate2]

        gate2 = Logic.Gate("NOT", lambda x: not x, 1)
        gate2.gate_inputs = [gate]

        self.assertEqual(id_gate1.height, 0)
        self.assertEqual(id_gate2.height, 0)

        self.assertEqual(gate.height, 1)

        self.assertEqual(gate2.height, 2)

    def test_ordered_representation(self):
        gate_type = 'testGate'
        gate_logic = _and
        n_inputs = 2
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 1
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 2

        gate1 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate2 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate1.gate_inputs = [id_gate1, id_gate2]
        gate2.gate_inputs = [id_gate2, id_gate1]

        gate1.sort()
        gate2.sort()

        self.assertEqual(repr(gate1), repr(gate2))

        gate3 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate4 = Logic.Gate(gate_type, gate_logic, n_inputs)

        gate3.gate_inputs = [gate1, gate2]
        gate4.gate_inputs = [gate2, gate1]

        gate3.sort()
        gate4.sort()

        self.assertEqual(repr(gate3), repr(gate4))

    def test_num_of_gates(self):
        gate_type = 'testGate'
        gate_logic = _and
        n_inputs = 2
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 1
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 2

        gate1 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate2 = Logic.Gate(gate_type, gate_logic, n_inputs)
        gate1.gate_inputs = [id_gate1, id_gate2]
        gate2.gate_inputs = [id_gate2, id_gate1]

        gate1.sort()
        gate2.sort()

        self.assertEqual(gate1.num_of_gates(), 3)
        self.assertEqual(gate2.num_of_gates(), 3)

        gate3 = Logic.Gate(gate_type, gate_logic, n_inputs)

        gate3.gate_inputs = [gate1, gate2]

        gate3.sort()

        self.assertEqual(gate3.num_of_gates(), 4)


class TestState(unittest.TestCase):

    half_adder = {(False, False): False,
                  (False, True): True,
                  (True, False): True,
                  (True, True): False}

    and_gate = Logic.Gate("And", _and, 2)
    or_gate = Logic.Gate("Or", _or, 2)
    not_gate = Logic.Gate("Not", _not, 1)

    gates = [and_gate, or_gate, not_gate]

    def __init__(self, *args, **kwargs):
        super(TestState, self).__init__(*args, **kwargs)
        self.state = Logic.State(gates=self.gates, truth_table=self.half_adder, n_inputs=2)

    def test_init(self):
        state = Logic.State(gates=self.gates, truth_table=self.half_adder, n_inputs=2)
        self.assertEqual(state.state.type, Logic.Gate.IDENTITY_GATE.type)
        self.assertEqual(state.n_inputs, 2)
        self.assertEqual(state.gates, self.gates)
        self.assertEqual(state.truth_table, self.half_adder)

    def test_get_actions(self):
        actions = self.state.get_actions()
        self.assertEqual(actions, self.gates)

    def test_get_outputs(self):
        state = self.state.__copy__()

        # create Not(Or(And(0,1), 0))
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 0
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 1

        gate1 = self.and_gate.__copy__()
        gate1.gate_inputs = [id_gate1, id_gate2]
        gate1.sort()

        gate2 = self.or_gate.__copy__()
        gate2.gate_inputs = [id_gate1, gate1]
        gate2.sort()

        gate3 = self.not_gate.__copy__()
        gate3.gate_inputs = [gate2]
        gate3.sort()

        state.get_outputs(gate3)

        expected = {id_gate1, id_gate2, gate1, gate2, gate3}

        self.assertEqual(expected, state.outputs)

    def test_get_successors(self):
        state = self.state.__copy__()
        successors = state.get_successors(self.and_gate.__copy__())
        succ_repr = sorted([repr(s) for s in successors])
        expected = sorted(['And(0,1)'])
        self.assertEqual(expected, succ_repr)

        gate = next(filter(lambda x: x.type == 'And', successors))
        state.state = gate
        state.get_outputs(gate)
        successors = state.get_successors(self.and_gate.__copy__())
        succ_repr = sorted([repr(s) for s in successors])
        expected = sorted(['0', '1', 'And(0,And(0,1))', 'And(1,And(0,1))'])
        self.assertEqual(expected, succ_repr)

        successors = state.get_successors(self.or_gate.__copy__())
        succ_repr = sorted([repr(s) for s in successors])
        expected = sorted(['Or(0,1)', 'And(0,Or(0,1))', 'And(1,Or(0,1))', 'Or(0,And(0,1))', 'Or(1,And(0,1))'])
        self.assertEqual(expected, succ_repr)

    def test_evaluate(self):
        # create state with And(1,Or(0,1))
        state = self.state.__copy__()
        successors = state.get_successors(self.and_gate.__copy__())
        gate = next(filter(lambda x: x.type == 'And', successors))
        state.state = gate
        state.get_outputs(gate)
        successors = state.get_successors(self.or_gate.__copy__())
        gate2 = next(filter(lambda x: x.__repr__() == 'And(1,Or(0,1))', successors))
        state.state = gate2
        state.get_outputs(gate2)

        expected = {(False, False): False,
                    (False, True): True,
                    (True, False): False,
                    (True, True): True}

        for _input in self.half_adder.keys():
            self.assertEqual(state.evaluate(_input, state.state), expected[_input])

    def test_is_goal(self):
        # create state with And(1,Or(0,1))
        state = self.state.__copy__()
        successors = state.get_successors(self.and_gate.__copy__())
        gate = next(filter(lambda x: x.type == 'And', successors))
        state.state = gate
        state.get_outputs(gate)
        successors = state.get_successors(self.or_gate.__copy__())
        gate2 = next(filter(lambda x: x.__repr__() == 'And(1,Or(0,1))', successors))
        state.state = gate2
        state.get_outputs(gate2)

        self.assertFalse(state.is_goal())

        # create state And(Not(And(0,1)), Or(0,1)) -- Half adder implementation
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 0
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 1
        and1 = self.and_gate.__copy__()
        and1.gate_inputs = [id_gate1, id_gate2]

        or1 = self.or_gate.__copy__()
        or1.gate_inputs = [id_gate1, id_gate2]

        not1 = self.not_gate.__copy__()
        not1.gate_inputs = [and1]

        and2 = self.and_gate.__copy__()
        and2.gate_inputs = [not1, or1]
        and2.sort()

        state = self.state.__copy__()
        state.state = and2
        state.get_outputs(and2)

        self.assertTrue(state.is_goal())

    def test_problem_value(self):
        # create state with And(1,Or(0,1))
        state = self.state.__copy__()
        successors = state.get_successors(self.and_gate.__copy__())
        gate = next(filter(lambda x: x.type == 'And', successors))
        state.state = gate
        state.get_outputs(gate)
        successors = state.get_successors(self.or_gate.__copy__())
        gate2 = next(filter(lambda x: x.__repr__() == 'And(1,Or(0,1))', successors))
        state.state = gate2
        state.get_outputs(gate2)
        problem = Logic.Problem(state)
        val1 = problem.value(state)

        # create state And(Not(And(0,1)), Or(0,1)) -- Half adder implementation
        id_gate1 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate1.gate_inputs = 0
        id_gate2 = Logic.Gate.IDENTITY_GATE.__copy__()
        id_gate2.gate_inputs = 1
        and1 = self.and_gate.__copy__()
        and1.gate_inputs = [id_gate1, id_gate2]
        or1 = self.or_gate.__copy__()
        or1.gate_inputs = [id_gate1, id_gate2]
        not1 = self.not_gate.__copy__()
        not1.gate_inputs = [and1]
        and2 = self.and_gate.__copy__()
        and2.gate_inputs = [not1, or1]
        and2.sort()
        state = self.state.__copy__()
        state.state = and2
        state.get_outputs(and2)
        problem = Logic.Problem(state)
        val2 = problem.value(state)

        self.assertGreater(val2, val1)


if __name__ == '__main__':
    unittest.main()

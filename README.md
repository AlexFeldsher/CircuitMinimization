# CircuitMinimization
AI final project

This solution uses a graph representation of a circuit, where each node is a Logic gate.
Each gate is a root of it's own subtree, and his children are the gates that connected to its input.

### Limitation
* Currently only works with a single output.
* No obvious way to support genetic algorithms, since connecting parts of different graphs could be messy.

### Features
* Supports a random amount and types of gates, with any number of inputs.

        Gate(name='AND', logic=lambda x,y: x and y, n_params=2)
        Gate(name='NOT', logic=lambda x: not x, n_params=1)
    
* Works with simulated annealing

### Circuit who's thy neighbor?
* All the circuits with an additional gate.
* All the circuits with a single gate changed.
* All the circuits with a single gate removed.

### Circuit what's thy value?
* The circuit with the lowest number of gates that solves the truth table should have the heighst value

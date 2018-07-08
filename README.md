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

## Simulated annealing schedule function
Currently using the function provided in the aima library.

    def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
        return lambda t: (k * math.exp(-lam * t) if t < limit else 0)
* High k value - large cicuits are explored and takes longer to converge (also requires higher limit).
* limit - the number of iterations performed.
* lam ??
* Coverging to the optimal solution also depend on the circuit value.

## TODO
* Define a value to a circuit.
* Identify equivalent circuits. Will significantly reduce the search space.

        AND(0,OR(1,0)) == AND(OR(0,1),0)
        
* Find the optimal simulated annealing schedule function.
* Find a correct way to implement slicing in the Gate class for genetic algorithms.


## Run
    python3 Main.py

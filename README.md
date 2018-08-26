# CircuitMinimization
AI final project

### Genetic Programming Implementation
Use pip to install dependencies by executing:
```
pip install -r requirements.txt 
```

Run `gp.py`, see list of available variables by executing:
```
python gp.py -h
```
### Simulated Annealing Implementation
requirements are:

    pyeda==0.28.0
    typing==3.6.4
```
python3 simulated_annealing.py -h

usage: simulated_annealing.py [-h]
                              truth_table n_vars k lam limit u_gate_lim
                              l_gate_lim height_lim

Circuit minimization using Simulated Annealing

positional arguments:
  truth_table  Bits representing a truth table, see truthtable function at
               "https://pyeda.readthedocs.io/en/latest/boolalg.html#boolean-
               functions"
  n_vars       The number of variables
  k            Parameter k for the scheduler function k*exp(-lam*t)
  lam          Parameter lam for the scheduler function k*exp(-lam*t)
  limit        Limit the number of search iterations
  u_gate_lim   Upper limit for the number of gates
  l_gate_lim   Lower limit for the number of gates
  height_lim   Height limit for the resulting circuit

optional arguments:
  -h, --help   show this help message and exit
```
example:
```
python3 simulated_annealing.py 0101 2 1 0.05 200 5 0 4
    
0101 represents the truth table:
        x y out
        0 0  0
        0 1  1
        1 0  0
        1 1  1
2 variables (x,y), schedule function = 1*exp(-0.05*t), 200 iterations limit
upper gate limit is 5, lower gate limit is 0, and height limit is 4
```

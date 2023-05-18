# U-GA
A Python implementation of the Unscented Genetic Algorithm (U-GA).

Usage:

python3 u_ga.py -h -n [1:inf] -b [1:inf] -p [1:inf] -f function name
-h: help
-n: number of generations
-b: number of bits
-p: population size
-f: funtion to be optimized (imported from test_functions.py)

Available Functions:

Function 1: one_max <br>
Function 2: zero_max
Function 3: sq01
Function 4: quad01
Function 5: lines01
Function 6: lines02
Function 7: trap01
Function 8: trap02
Function 9: mxn
Function 10: 2x4+12
Function 11: beale
Function 12: chung-reynolds
Function 13: dixon-price
Function 14: el-attar-vidyasagar-dutta
Function 15: matyas
Function 16: powell-singular
Function 17: powell-singular-2
Function 18: powell-sum
Function 19: rosenbrock
Function 20: rotated-ellipse
Function 21: rotated-ellipse-2
Function 22: rump
Function 23: scahffer-1
Function 24: scahffer-2
Function 25: scahffer-3
Function 26: scahffer-4
Function 27: sphere
Function 28: schwefel-1-2
Function 29: schwefel-2-21
Function 30: schwefel-2-22
Function 31: step-1
Function 32: step-2
Function 33: step-3
Function 34: stepint
Function 35: type-1-deceiptive
Function 36: type-2-deceiptive
Function 37: type-3-deceiptive


Reference:

Aguilar-Rivera, A. (2023). The unscented genetic algorithm for fast solution of GA-hard optimization problems. Applied Soft Computing, 139, 110260.

Benchmarks Reference:

M. Jamil, X.-S. Yang, A literature survey of benchmark functions for global optimization problems, arXiv preprint arXiv:1308.4008 (2013).

A. R. Al-Roomi, Unconstrained Single-Objective Benchmark Functions Repository (2015). URL https://www.al-roomi.org/benchmarks/unconstrained

# Knapsack Genetic Solver 

This project provides a class solving the Knapsack Problem using a Genetic Algorithm. 

Created for the Intelligent Systems course.

Made by:

	Brenda Eloisa Sánchez Pichardo - A01801759
	Valeria Jassive Ramírez Macías - A01636965
	Diana Patricia Madera Espíndola - A01025835
	Zoe Caballero Domíguez - A01747247


**Required libraries:**

	* matplotlib
	* argparse
    * Numpy
    * Random


## Usage

By default, the program runs all the instances. Make sure you have all of them in a folder called 'KPInstances'. Also, please do not remove the 'Graph Results'.

Run the script with the following syntax:

```python knapsackSolver.py [optional arguments]```

To run a specific instance you can do so using:

```python knapsackSolver.py --file "your_file_path" [optional arguments]```


## Arguments

**--file**: String. Path to the knapsack instance file.

**--plot_objective**: String. File name for the objective function plot. Default: "objective_function.png"

**--plot_fitness**: String. File name for the fitness plot. "Default: fitness.png"


### Arguments for the Genetic Algorithm

We tunned the defaults after running parameter optimization using the Optuna library. Since not all values work best with all the instances, we decided on those that gave good results for most of them.

**--population_size**: Int. Size of the population for the genetic algorithm.Default: 550

**--tournament_selection**: Int. Number of tournament selections. Default: 2

**--probability_crossover**: Float. Crossover probability (range: 0-1). Default: 0.7

**--probability_mutation**: Float. Mutation probability. If not provided, defaults to 1/n_items.

**--generations**: Int. Number of generations for the algorithm. Default: 500


## Example

Solve a knapsack problem with custom parameters:

```python knapsackSolver.py --file "KPInstances/ks_4_0" --population_size 300 --generations 200```

or 

```python knapsackSolver.py  --population_size 300 --generations 200```

## Example Output

	(11, [(4, 8.0), (5, 10.0), (8, 15.0), (3, 4.0)])

	Best Solution:
	[0 0 1 1]

	Results:
	(19.0, 11)

	Generating plots...
	Objective function plot saved as objective_function.png
	Fitness plot saved as fitness.png


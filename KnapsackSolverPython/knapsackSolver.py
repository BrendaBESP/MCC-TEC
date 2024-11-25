import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse


def check_file_path(path):
    """
    Helper function to check if the pathfile is valid.
    """
    try:
      if len(path) == 0:
        raise Exception("The pathfile is empty")
      with open(path, "r") as f:
        pass
    except Exception as e:
      print(
        "There is an error with the pathfile. Please enter a valid path using the argument '--file' or run all instances with '--run_all'"
      )
      exit(1)

class KnapsackGeneticSolver:
    """
    KnapsackGeneticSolver class to solve the knapsack problem using a genetic algorithm.
    Takes in a path to a file with the knapsack problem instance.
    """
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.max_weight = None
        self.items = None
        self.problem = self.load(instance_path)
        self.n_items = self.get_n_items()

    def load(self, fileName):
        """
        Loads a filename to create knapsack problem instance.
        The function returns a tuple, where the first element represents W and,
        the second, a list of the items within the instance. Each item is represented
        as a tuple (weight, profit).
        """
        f = open(fileName, "r")
        lines = f.readlines()
        line = lines[0].split(",")
        nbItems = int(line[0].strip())
        maxWeight = int(line[1].strip())
        items = [None] * nbItems
        for i in range(0, nbItems):
            line = lines[i + 1].split(",")
            weight = int(line[0].strip())
            profit = float(line[1].strip())
            items[i] = (weight, profit)
        self.max_weight = maxWeight
        self.items = items
        return (maxWeight, items)

    def print_problem(self):
        """
        Prints the instance problem in the form (weight, items)
        """
        print(self.problem)

    def get_n_items(self):
        """
        Returns the number of items in the instance
        """
        return len(self.problem[1])

    def _create(self):
        """
        Creates a random individual of the size of the problem
        """
        return np.random.randint(0, 2, self.n_items)

    def _mutate(self, individual, pm):
        """
        Mutates an individual with a probability of pm.
        pm = prob(mutation)
        """
        for i in range(len(individual)):
            if random.random() < pm:
                individual[i] = 1 - individual[i]

    def _combine(self, parentA, parentB, pc):
        """
        Combines two parents with a probability of pc.
        pc = prob(crossover)
        """
        if random.random() < pc:
            cPoint = np.random.randint(1, len(parentA))
            offspringA = np.concatenate((parentA[0:cPoint], parentB[cPoint:]))
            offspringB = np.concatenate((parentB[0:cPoint], parentA[cPoint:]))
            return offspringA, offspringB
        
        return parentA, parentB

    def _evaluate(self, individual):
        """
        Evaluates an individual based on the profit. If the weight of the items
        exceeds the max weight, we ignore the rest of the items.
        """
        profit = 0
        weight = 0
        for item in range(len(individual)):
            if individual[item] == 1:
                weight += self.items[item][0]
                if weight > self.max_weight:
                    return profit
                else:
                    profit += self.items[item][1]
        return profit

    def _select(self, population, size, evaluations):
        """
        Selects the best individual from two random individuals in the population.
        """
        bestID = np.random.randint(0, len(population))
        for i in range(size - 1):
            rivalID = np.random.randint(0, len(population))
            if evaluations[rivalID] > evaluations[bestID]:
                bestID = rivalID

        return np.copy(population[bestID])

    def _ga(self, pSize, tournament_selection, pc, pm, generations):
        """
        Runs the genetic algorithm.
        """
        best_profit = [None] * generations
        worst_profit = [None] * generations
        average_profit = [None] * generations

        population = [None] * pSize
        evaluations = [None] * pSize
        for i in range(pSize):
            population[i] = self._create()
            evaluations[i] = self._evaluate(population[i])

        for i in range(generations):
            newPopulation = [None] * pSize
            newEvaluations = [None] * pSize
            for j in range(pSize // 2):
                parentA = self._select(population, tournament_selection, evaluations)
                parentB = self._select(population, tournament_selection, evaluations)
                offspringA, offspringB = self._combine(parentA, parentB, pc)
                self._mutate(offspringA, pm)
                self._mutate(offspringB, pm)
                newPopulation[2 * j] = offspringA
                newPopulation[2 * j + 1] = offspringB
                newEvaluations[2 * j] = self._evaluate(offspringA)
                newEvaluations[2 * j + 1] = self._evaluate(offspringB)
            population = newPopulation
            evaluations = newEvaluations
            best_profit[i] = max(newEvaluations)
            worst_profit[i] = min(newEvaluations)
            average_profit[i] = sum(newEvaluations) / len(newEvaluations)

        bestID = 0
        for i in range(1, pSize):
            if evaluations[i] > evaluations[bestID]:
                bestID = i
        history = {
            "Best": best_profit,
            "Worst": worst_profit,
            "Average": average_profit,
        }
        return population[bestID], history

    def solve(
        self,
        population_size=500,
        tournament_selection=3,
        probability_crossover=0.9,
        probability_mutation=0.25,
        generations=500,
    ):
        """
        Returns the solution using a genetic algorithm.
        population_size = size of the population
        probability_crossover = probability of crossover
        probability_mutation = probability of mutation. It is recommended to
          set it as 1/n, where n is the number of items in the problem
        generations = number of generations
        """
        return self._ga(
            population_size,
            tournament_selection,
            probability_crossover,
            probability_mutation,
            generations,
        )

    def get_results(self, individual):
        """
        Returns the profit and the weight of the individual.
        """
        profit = 0
        weight = 0
        for item in range(len(individual)):
            if individual[item] == 1:
                weight += self.items[item][0]
                if weight > self.max_weight:
                    weight -= self.items[item][0]
                    return profit, weight
                else:
                    profit += self.items[item][1]
        return profit, weight

    def plot_objective_function(self, hist, name):
        """
        Plots the objective function through the generations.
        """
        num_plots = len(hist)
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6), sharex=True)

        colors = ["blue", "#FF8902", "#009E73"]

        for i, (key, values) in enumerate(hist.items()):
            axes[i].plot(values, marker="o", color=colors[i])
            axes[i].set_ylabel(key)
            axes[i].grid(True)

        plt.xlabel("Generation")
        plt.suptitle("Evolution of objective function through generations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(name)

    def plot_fitness(self, hist, name):
        """
        Plots the fitness through the generations.
        """  
        plt.figure(figsize=(8, 6))
        for key in hist:
            plt.plot(hist[key], label=key, marker="o")

        plt.xlabel("Generation")
        plt.ylabel("Profit")
        plt.title("Evolution of the generations fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(name)


def main():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description="Knapsack Genetic Solver")
    parser.add_argument("--file", type=str, help="Path to the knapsack problem file")
    parser.add_argument(
        "--population_size",
        type=int,
        default=550,
        help="Population size for the solver",
    )
    parser.add_argument(
        "--tournament_selection",
        type=int,
        default=2,
        help="Number of tournament selections",
    )
    parser.add_argument(
        "--probability_crossover",
        type=float,
        default=0.7,
        help="Probability of crossover (0-1)",
    )
    parser.add_argument(
        "--probability_mutation",
        type=float,
        help="Probability of mutation (default: 1/n_items)",
    )
    parser.add_argument(
        "--generations", type=int, default=500, help="Number of generations"
    )
    parser.add_argument(
        "--plot_objective",
        type=str,
        default="objective_function.png",
        help="File name for objective function plot",
    )
    parser.add_argument(
        "--plot_fitness",
        type=str,
        default="fitness.png",
        help="File name for fitness plot",
    )
    args = parser.parse_args()

    run_all = True
    if args.file:
        run_all = False
        
    # If the flag is set to run all instances, we iterate over all of them
    if run_all:
        file_names = [
            "ks_4_0",
            "ks_19_0",
            "ks_30_0",
            "ks_40_0",
            "ks_45_0",
            "ks_50_0",
            "ks_50_1",
            "ks_60_0",
            "ks_82_0",
            "ks_100_0",
            "ks_100_1",
            "ks_100_2",
            "ks_106_0",
            "ks_200_0",
            "ks_200_1",
            "ks_300_0",
            "ks_400_0",
            "ks_500_0",
            "ks_1000_0",
            "ks_10000_0",
        ]

        # Iterate over each file
        for filename in file_names:
            knapsack_solver = KnapsackGeneticSolver(f"KPInstances/{filename}")
            mutation = args.probability_mutation or (1 / knapsack_solver.n_items)

            best_solution, hist = knapsack_solver.solve(
                population_size=args.population_size,
                tournament_selection=args.tournament_selection,
                probability_crossover=args.probability_crossover,
                probability_mutation=mutation,
                generations=args.generations,
            )

            # Display results
            print(f"~~~~~~Solving instance {filename}~~~~~~")
            print("\nBest Solution:")
            print(best_solution)
            print("\nResults:")
            print(knapsack_solver.get_results(best_solution))

            # Plot the results
            print("\nGenerating plots...")
            knapsack_solver.plot_objective_function(
                hist, f"GraphResults/{filename}_{args.plot_objective}"
            )
            knapsack_solver.plot_fitness(hist, f"GraphResults/{filename}_{args.plot_fitness}")
            print(f"Objective function plot saved as {args.plot_objective}")
            print(f"Fitness plot saved as {args.plot_fitness}")
    else:
        # Check if the pathfile is valid
        check_file_path(args.file)

        # Initialize the solver with the given file
        knapsack_solver = KnapsackGeneticSolver(args.file)
        knapsack_solver.print_problem()

        # Compute default mutation probability if not provided
        mutation_probability = args.probability_mutation or (
            1 / knapsack_solver.n_items
        )

        # Solve the knapsack problem
        best_solution, hist = knapsack_solver.solve(
            population_size=args.population_size,
            tournament_selection=args.tournament_selection,
            probability_crossover=args.probability_crossover,
            probability_mutation=mutation_probability,
            generations=args.generations,
        )

        # Display results
        print("\nBest Solution:")
        print(best_solution)
        print("\nResults:")
        print(knapsack_solver.get_results(best_solution))

        # Plot the results
        print("\nGenerating plots...")
        knapsack_solver.plot_objective_function(hist, f"GraphResults/{args.plot_objective}")
        knapsack_solver.plot_fitness(hist, f"GraphResults/{args.plot_fitness}")
        print(f"Objective function plot saved as {args.plot_objective}")
        print(f"Fitness plot saved as {args.plot_fitness}")


if __name__ == "__main__":
    main()

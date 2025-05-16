from pulp import *
import sys

def read_data(filename):
    """
    Reads data from a specified text file.

    Args:
        filename (str): Path to the input file.

    Returns:
        list: Parsed data as a list of integer lists.
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append([int(x) for x in line.strip().split()])
    return data

def solve_single_knapsack(nb_items, demand_size, weight_item, cost_item, mode):
    """
    Solves the single knapsack minimization problem (primal or dual).

    Args:
        nb_items (int): Number of items.
        demand_size (list): Required weight threshold (length 1).
        weight_item (list): Weights of items.
        cost_item (list): Costs of items.
        mode (int): Mode of solving (0 = primal integer, 1 = relaxed, 2 = dual).

    Returns:
        tuple: (status, objective_value, selected_items or dual variables).
    """
    if mode in [0, 1]:  # Primal mode
        prob = LpProblem("Single_minKP", LpMinimize)
        category = 'Binary' if mode == 0 else 'Continuous'
        x = LpVariable.dicts("present", range(nb_items), 0, 1, cat=category)

        # Objective function
        prob += lpSum(cost_item[j] * x[j] for j in range(nb_items))

        # Constraint: meet demand
        prob += lpSum(weight_item[j] * x[j] for j in range(nb_items)) >= demand_size[0]

        prob.solve()

        selected_items = [(j, weight_item[j], cost_item[j]) for j in range(nb_items) if value(x[j]) > 1e-5]
        return LpStatus[prob.status], value(prob.objective), selected_items

    else:  # Dual mode
        prob = LpProblem("Dual_minKP", LpMaximize)
        v_1 = LpVariable("v_1", lowBound=0)
        u = LpVariable.dicts("u_j", range(nb_items), lowBound=0)

        # Objective function
        prob += demand_size[0] * v_1 - lpSum(u[j] for j in range(nb_items))

        for j in range(nb_items):
            prob += weight_item[j] * v_1 - u[j] <= cost_item[j]

        prob.solve()

        return (
            LpStatus[prob.status],
            value(prob.objective),
            {"v1": [value(v_1)], "uj": [value(u[j]) for j in range(nb_items)]}
        )

def solve_multi_knapsack(nb_knapsacks, nb_items, demand_size, weight_item, cost_item, mode):
    """
    Solves the multiple knapsack minimization problem (primal or dual).

    Args:
        nb_knapsacks (int): Number of knapsacks.
        nb_items (int): Number of items.
        demand_size (list): Required weights per knapsack.
        weight_item (list): Weights of items.
        cost_item (list): Costs of items.
        mode (int): Solving mode (0 = primal integer, 1 = relaxed, 2 = dual).

    Returns:
        tuple: (status, objective_value, result_data).
    """
    if mode in [0, 1]:  # Primal mode
        prob = LpProblem("Multi_minKP", LpMinimize)
        category = 'Binary' if mode == 0 else 'Continuous'

        x = LpVariable.dicts("present", ((i, j) for i in range(nb_knapsacks) for j in range(nb_items)), 0, 1, cat=category)

        # Objective function
        prob += lpSum(cost_item[j] * x[i, j] for i in range(nb_knapsacks) for j in range(nb_items))

        # Constraints: each knapsack meets its demand
        for i in range(nb_knapsacks):
            prob += lpSum(weight_item[j] * x[i, j] for j in range(nb_items)) >= demand_size[i]

        # Each item is assigned to at most one knapsack
        for j in range(nb_items):
            prob += lpSum(x[i, j] for i in range(nb_knapsacks)) <= 1

        prob.solve()

        selected_items = {
            i: [(j, weight_item[j], cost_item[j]) for j in range(nb_items) if value(x[i, j]) > 1e-5]
            for i in range(nb_knapsacks)
        }
        return LpStatus[prob.status], value(prob.objective), selected_items

    else:  # Dual mode
        prob = LpProblem("Dual_multi_minKP", LpMaximize)
        vi = LpVariable.dicts("vi", range(nb_knapsacks), lowBound=0)
        vj = LpVariable.dicts("vj", range(nb_items), lowBound=0)

        # Objective function
        prob += lpSum(demand_size[i] * vi[i] for i in range(nb_knapsacks)) - lpSum(vj[j] for j in range(nb_items))

        # Constraints
        for i in range(nb_knapsacks):
            for j in range(nb_items):
                prob += weight_item[j] * vi[i] - vj[j] <= cost_item[j]

        prob.solve()

        return (
            LpStatus[prob.status],
            value(prob.objective),
            {"vi": [value(vi[i]) for i in range(nb_knapsacks)], "vj": [value(vj[j]) for j in range(nb_items)]}
        )

def print_single_knapsack_results(status, objective_value, data):
    """Displays the results of the single knapsack problem."""
    print("\nStatus:", status)
    print(f"Optimal Total Cost: {objective_value}")

    if isinstance(data, dict) and "v1" in data and "uj" in data:
        print("\nDual variables (v1 for the knapsack):")
        for val in data["v1"]:
            print(f"  v1 = {val:.4f}")

        print("\nDual variables (uj for each item):")
        for j, val in enumerate(data["uj"]):
            print(f"  u{j} = {val:.4f}")
    else:
        print("\nOptimal items selected:")
        total_weight = 0
        for j, weight, cost in data:
            print(f"Item {j}: weight = {weight}, cost = {cost}")
            total_weight += weight
        print(f"\nTotal weight of selected items: {total_weight}")

def print_multi_knapsack_results(status, objective_value, data):
    """Displays the results of the multiple knapsack problem."""
    print("\nStatus:", status)
    print(f"Objective value: {objective_value}")

    if isinstance(data, dict) and "vi" in data and "vj" in data:
        print("\nDual variables (vi for each knapsack):")
        for i, val in enumerate(data["vi"]):
            print(f"  v{i} = {val:.4f}")

        print("\nDual variables (vj for each item):")
        for j, val in enumerate(data["vj"]):
            print(f"  v{j} = {val:.4f}")
    else:
        print("\nOptimal items selected by knapsack:")
        for knapsack, items in data.items():
            print(f"\nKnapsack {knapsack}:")
            total_weight = 0
            for j, weight, cost in items:
                print(f"  Item {j}: weight = {weight}, cost = {cost}")
                total_weight += weight
            print(f"  Total weight in knapsack {knapsack}: {total_weight}")

def main():
    """Main function to solve knapsack problems based on file input and mode."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename> <mode: 0|1|2>")
        sys.exit(1)

    fileName = sys.argv[1]
    mode = int(sys.argv[2])

    if mode not in [0, 1, 2]:
        print("Invalid mode. Use 0 for primal integer, 1 for relaxation, or 2 for dual.")
        sys.exit(1)

    data = read_data(fileName)
    nb_knapsacks = data[0][0]
    nb_items = data[1][0]
    demand_size = data[2]
    weight_item = data[3]
    cost_item = data[4]

    print(f"Problem instance: {fileName}")
    print(f"Number of knapsacks: {nb_knapsacks}")
    print(f"Number of items: {nb_items}")

    if nb_knapsacks == 1:
        status, obj_val, selected = solve_single_knapsack(nb_items, demand_size, weight_item, cost_item, mode)
        print_single_knapsack_results(status, obj_val, selected)
    else:
        status, obj_val, selected = solve_multi_knapsack(nb_knapsacks, nb_items, demand_size, weight_item, cost_item, mode)
        print_multi_knapsack_results(status, obj_val, selected)

if __name__ == "__main__":
    main()
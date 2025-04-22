from pulp import *
import sys

def read_data(filename):
    """
    Read data from a txt file
    
    Args:
        filename (str): Path to the input file
        
    Returns:
        list: Processed data from the file
    """
    alist = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            alist.append([int(x) for x in line.strip().split()])
    return alist
 
def solve_single_knapsack(nb_items, demand_size, weight_item, cost_item, mode):
    if mode in [0, 1]:  # Primal
        prob = LpProblem("Single_minKP", LpMinimize)
        cat = 'Binary' if mode == 0 else 'Continuous'
        x = LpVariable.dicts("present", range(nb_items), 0, 1, cat=cat)

        prob += lpSum(cost_item[j] * x[j] for j in range(nb_items))
        prob += lpSum(weight_item[j]*x[j] for j in range(nb_items)) >= demand_size[0]

        prob.solve()

        selected_items = [(j, weight_item[j], cost_item[j])
                          for j in range(nb_items) if value(x[j]) > 1e-5]
        return LpStatus[prob.status], value(prob.objective), selected_items

    else:  # Dual
        prob = LpProblem("Dual_minKP", LpMaximize)

        v_1 = LpVariable("v_1", lowBound=0)
        u = LpVariable.dicts("u_j", range(nb_items), lowBound=0)

        # Objective: maximize dual function
        prob += demand_size[0]*v_1 - lpSum(u[j] for j in range(nb_items))

        for j in range(nb_items):
            prob += weight_item[j]*v_1 - u[j] <= cost_item[j]

        prob.solve()

        return (
            LpStatus[prob.status],
            value(prob.objective),
            {
                "v1": [value(v_1)],
                "uj": [value(u[j]) for j in range(nb_items)]
            }
        )
            
        
        

def solve_multi_knapsack(nb_knapsacks, nb_items, demand_size, weight_item, cost_item, mode):
    """
    Solve the multiple knapsack minimization problem
    
    Args:
        nb_knapsacks (int): Number of knapsacks
        nb_items (int): Number of items
        demand_size (list): Minimum required weights for each knapsack
        weight_item (list): Weights of items
        cost_item (list): Costs of items
        
    Returns:
        tuple: (status, objective_value, selected_items_by_knapsack)
    """
    if mode in [0, 1]: # Primal
        # Create the model
        prob = LpProblem("Multi_minKP", LpMinimize)

        cat = 'Binary' if mode == 0 else 'Continuous'
    
        # Create variables
        x = LpVariable.dicts("present", ((i,j) for i in range(nb_knapsacks) for j in range(nb_items)),lowBound=0, upBound=1, cat=cat)
        
        # Objective: minimize total cost
        prob += lpSum(cost_item[j] * x[i,j] for i in range(nb_knapsacks) for j in range(nb_items))
        
        # Constraint: Need to select enough items to respect the demanding size for each knapsack
        for i in range(nb_knapsacks):
            prob += lpSum(weight_item[j]*x[i,j] for j in range(nb_items)) >= demand_size[i]
        
        # Constraint: An item is present in at most one knapsack
        for j in range(nb_items):
            prob += lpSum(x[i,j] for i in range(nb_knapsacks)) <= 1
        
        # Solve the problem
        prob.solve()
        
        # Return results
        selected_items_by_knapsack = {}
        for i in range(nb_knapsacks):
            selected_items_by_knapsack[i] = [(j, weight_item[j], cost_item[j]) 
                                            for j in range(nb_items) if value(x[i,j]) > 0]
        return LpStatus[prob.status], value(prob.objective), selected_items_by_knapsack
    else: # Dual
         # Create the model
        prob = LpProblem("Dual_multi_minKP", LpMaximize)
        # Create variables
        vi = LpVariable.dicts("vi", (i for i in range(nb_knapsacks)),lowBound=0)
        vj = LpVariable.dicts("vj", (j for j in range(nb_items)),lowBound=0)
        
        # Objective: minimize total cost
        prob += lpSum(demand_size[i] * vi[i] for i in range(nb_knapsacks)) - lpSum(vj[j] for j in range(nb_items))
        
        for i in range(nb_knapsacks):
            for j in range(nb_items):
                prob += weight_item[j]*vi[i] - vj[j] <= cost_item[j]
        
        # Solve the problem
        prob.solve()
        
        # Return results
        return (
            LpStatus[prob.status],
            value(prob.objective),
            {
                "vi": [value(vi[i]) for i in range(nb_knapsacks)],
                "vj": [value(vj[j]) for j in range(nb_items)]
            }
        )


    
def print_single_knapsack_results(status, objective_value, data):
    """Print results for single knapsack problem"""
    print("\nStatus:", status)
    print(f"Optimal Total Cost: {objective_value}")
    
    print("\nOptimal items selected:")
       # Si on est en mode dual, data contient les vecteurs vi et vj
    if isinstance(data, dict) and "v1" in data and "uj" in data:
        print("\nDual variables (v1 for the knapSack):")
        for i, val in enumerate(data["v1"]):
            print(f"  v1 = {val:.4f}")
        print("\nDual variables (vj for each item):")
        for j, val in enumerate(data["uj"]):
            print(f"  u{j} = {val:.4f}")
    else:
        total_weight = 0
        for j, weight, cost in data:
            print(f"Item {j}: weight = {weight}, cost = {cost}")
            total_weight += weight
        print(f"\nTotal weight of selected items: {total_weight}")

def print_multi_knapsack_results(status, objective_value, data):
    """Print results for multiple knapsack problem (primal OR dual)"""
    print("\nStatus:", status)
    print(f"Objective value: {objective_value}")

    # Si on est en mode dual, data contient les vecteurs vi et vj
    if isinstance(data, dict) and "vi" in data and "vj" in data:
        print("\nDual variables (vi for each knapsack):")
        for i, val in enumerate(data["vi"]):
            print(f"  v{i} = {val:.4f}")
        print("\nDual variables (vj for each item):")
        for j, val in enumerate(data["vj"]):
            print(f"  v{j} = {val:.4f}")
    else:
        # Mode primal : data est selected_items_by_knapsack
        print("\nOptimal items selected by knapsack:")
        for knapsack, items in data.items():
            print(f"\nKnapsack {knapsack}:")
            total_weight = 0
            for j, weight, cost in items:
                print(f"  Item {j}: weight = {weight}, cost = {cost}")
                total_weight += weight
            print(f"  Total weight in knapsack {knapsack}: {total_weight}")

def main():
    # Set up argument parser
    fileName = sys.argv[1]
    mode = int(sys.argv[2])
    
    # Check mode
    if mode not in [0, 1, 2]:
        print("Invalid mode. Use 0 for primal entier, 1 for relaxation, or 2 for dual.")
        sys.exit(1)
    
    # Read data
    data_list = read_data(fileName)
    nb_knapsacks = data_list[0][0]
    nb_items = data_list[1][0]
    demand_size = data_list[2]
    weight_item = data_list[3]
    cost_item = data_list[4]
    
    print(f"Problem instance: {fileName}")
    print(f"Number of knapsacks: {nb_knapsacks}")
    print(f"Number of items: {nb_items}")
    
    if nb_knapsacks == 1:
        status, objective_value, selected_items = solve_single_knapsack(
            nb_items, demand_size, weight_item, cost_item, mode)
        print_single_knapsack_results(status, objective_value, selected_items)
    else:
        status, objective_value, selected_items_by_knapsack = solve_multi_knapsack(
            nb_knapsacks, nb_items, demand_size, weight_item, cost_item, mode)
        print_multi_knapsack_results(status, objective_value, selected_items_by_knapsack)

if __name__ == "__main__":
    main()
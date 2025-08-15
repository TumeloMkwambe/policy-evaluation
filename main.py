import copy
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Gridworld:
    NEIGHBOURING_CELLS = [[0,-1], [1,0], [0,1], [-1,0]]
    
    def __init__(self, row_dim, col_dim, goal_state, discount_factor):
        '''
        Args:
            row_dim (int): number of rows in grid
            col_dim (int): number of columns in grid
            goal_state (list): terminal state for grid world

        Attributes:
            grid (np.array): grid world states
            values (np.array): values aligned according to position of state in grid
            discount_rate (int): discount rate
        '''

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.discount_factor = discount_factor
        self.goal_state = goal_state
        self.grid = np.zeros((row_dim, col_dim), dtype=np.int32)
        self.values = np.zeros((row_dim, col_dim), dtype=np.float32)

    def isValid(self, cell):
        '''
        Args:
            cell (list): coordinates of state
        
        Objective:
            checks if state is valid and within grid world
        '''
        
        valid = 0 <= cell[0] < self.row_dim and 0 <= cell[1] < self.col_dim
        return valid

    def pe_loop(self, type_):
        '''
        Args:
            type_ (string): determines policy evaluation version to implement (in-place or two-array)
            
        Objective:
            Implementation of the policy evaluation algorithm; two-arrays or in-place version depending on the type_ argument.
            
        **Note:
            Rewards of -1 on all transitions
            Policy is uniform random with 4 possible actions hence π(a|s) = 0.25 for all a given s.
            Transitions are deterministic hence impossible states given action are not explored since p(s',r|s,a) = 0.
            Near walls, certain actions will lead to a no-change of state, hence we'll use that same state's value for calculating its new value.
        '''

        if type_ == 'in-place':
            old_values = self.values
        elif type_ == 'two-array':
            old_values = copy.copy(self.values)

        max_change = 0
        for i in range(self.row_dim):
            for j in range(self.col_dim):
                if([i, j] != self.goal_state):
                    new_value = 0
                    for neigh in self.NEIGHBOURING_CELLS: 
                        neigh_cell = [a + b for a, b in zip([i, j], neigh)] 
                        if self.isValid(neigh_cell):
                            next_i, next_j = neigh_cell
                        else:
                            next_i, next_j = i, j # if at far end then state does not change
                        new_value += 0.25 * (-1 + self.discount_factor * old_values[next_i, next_j])
                    max_change = max(max_change, abs(self.values[i, j] - new_value))
                    self.values[i, j] = new_value
        return max_change

    def policy_evaluation(self, type_, threshold):
        max_change = threshold + 1
        iteration = 0
        while max_change > threshold:
            max_change = self.pe_loop(type_)
            iteration += 1
        return iteration

if __name__ == "__main__":
    discount_rates = np.logspace(-0.2, 0, num=20)
    conv_iters = {
        'two_array': [],
        'in_place': []
    }

    for i in range(20):
        two_array = Gridworld(4, 4, [0,0], discount_rates[i])
        in_place = Gridworld(4, 4, [0,0], discount_rates[i])
        ta_iterations = two_array.policy_evaluation('two-array', 1e-2)
        ip_iterations = in_place.policy_evaluation('in-place', 1e-2)
        conv_iters['two_array'].append(ta_iterations)
        conv_iters['in_place'].append(ip_iterations)

        if discount_rates[i] == 1.0:
            ta_heatmap_ax = sns.heatmap(two_array.values, annot=True, cmap="viridis")
            ip_heatmap_ax = sns.heatmap(in_place.values, annot=True, cmap="viridis")
            ta_heatmap_ax.figure.savefig('two_array_heatmap.png')
            ip_heatmap_ax.figure.savefig('in_place_heatmap.png')

    plt.figure(figsize=(8, 5))
    plt.plot(discount_rates, conv_iters['in_place'], marker='o', label="In-place update")
    plt.plot(discount_rates, conv_iters['two_array'], marker='s', label="Two-array update")
    plt.xlabel("Discount Factor (γ)")
    plt.ylabel("Iterations/Sweeps to Convergence")
    plt.title("Policy Evaluation: In-place vs Two-array Convergence Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
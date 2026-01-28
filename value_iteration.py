from typing import Tuple

import numpy as np
import math


class ValueIteration:

    def __init__(self, theta=0.0001, discount_factor=1.0):
        self.theta = theta
        self.discount_factor = discount_factor

    def calculate_q_values(self, current_capital: int, value_function: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Helper function to calculate the value for all action in a given state. Compute Q(s,a) for all valid stakes a at the given capital s.
        Args:
            current_capital: The gambler’s capital. Integer. (state)
            value_function: The vector that contains values at each state. (the recursive value function)
            rewards: The reward vector. (the immediate reward according to the gambler's problem definition)
        Returns:
            A vector containing the expected value of each action in THIS state.
            Its length equals to the number of actions.
            q_values: np.ndarray of length (max_stake + 1), where index == stake amount. q_values[0] is invalid and set to -inf so argmax never picks it.
        """
        # Implement the calculation of Q-values for all actions in the given state, and return them as a numpy array.
        s = int(current_capital)
        if s <= 0 or s >= 100:
            # Terminal states: no meaningful actions
            return np.array([float("-inf")], dtype=float)

        max_stake = min(s, 100 - s)
        q_values = np.full(max_stake + 1, float("-inf"), dtype=float)  # stake 0 invalid

        # Probabilities given in the assignment (two fair dice)
        p_lose_full = 5.0 / 12.0     # sum < 7
        p_win_full = 1.0 / 6.0       # sum == 7
        p_lose_half = 5.0 / 12.0     # sum > 7

        gamma = self.discount_factor

        for a in range(1, max_stake + 1):
            lose_half = int(math.ceil(a / 2.0))

            s1 = max(0, s - a)              # lose full stake
            s2 = min(100, s + a)            # win full stake
            s3 = max(0, s - lose_half)      # lose half stake (ceil)

            q = (
                p_lose_full * (rewards[s1] + gamma * value_function[s1])
                + p_win_full * (rewards[s2] + gamma * value_function[s2])
                + p_lose_half * (rewards[s3] + gamma * value_function[s3])
            )
            q_values[a] = q

        return q_values


    def value_iteration_for_gamblers(self) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Perform value iteration for the gambler's problem. Run value iteration until convergence for the modified Gambler's problem.
        Returns:
            policy:  np.ndarray (size 101) best stake for each capital 0..100
            V:       np.ndarray (size 101) value estimates for each capital 0..100
        """
        # Implement the value iteration algorithm for the gambler's problem, returning the optimal policy and value function.
        # States: 0..100 (0 and 100 are terminal)
        V = np.zeros(101, dtype=float)

        # Rewards: only reaching 100 gives reward 100, everything else 0
        rewards = np.zeros(101, dtype=float)
        rewards[100] = 100.0

        while True:
            delta = 0.0

            # Only update non-terminal states
            for s in range(1, 100):
                q_values = self.calculate_q_values(s, V, rewards)
                new_v = float(np.max(q_values))

                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v

            if delta < self.theta:
                break

        # Extract optimal policy
        policy = np.zeros(101, dtype=int)
        for s in range(1, 100):
            q_values = self.calculate_q_values(s, V, rewards)
            best_action = int(np.argmax(q_values))  # stake amount
            policy[s] = best_action

        # Terminal states: stake 0
        policy[0] = 0
        policy[100] = 0

        return policy, V

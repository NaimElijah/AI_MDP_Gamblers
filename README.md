# 🎰 Gambler’s Problem – Value Iteration (MDP)

This project implements **Value Iteration** to solve a modified version of the **Gambler’s Problem**, modeled as a **Markov Decision Process (MDP)**.

The goal is to compute:

* The **optimal policy** (how much to stake at each capital level)
* The **optimal value function** (expected return for each capital)

The implementation follows classical **Dynamic Programming / Reinforcement Learning** principles.

---

## 📌 Problem Description

A gambler starts with an initial capital between **0 and 100** and repeatedly places bets.

### States

* Capital levels: **0 to 100**
* States **0** and **100** are terminal

### Actions

* At capital `s`, the gambler may stake any integer amount
  `a ∈ {1, 2, ..., min(s, 100 − s)}`

### Transition Probabilities

Each bet outcome depends on rolling **two fair dice**:

| Outcome                   | Probability | Capital Change |
| ------------------------- | ----------- | -------------- |
| Lose full stake (sum < 7) | 5/12        | −a             |
| Win full stake (sum = 7)  | 1/6         | +a             |
| Lose half stake (sum > 7) | 5/12        | −ceil(a/2)     |

### Rewards

* Reaching capital **100** gives a reward of **100**
* All other transitions give **0**

### Objective

Maximize the expected total reward using **Value Iteration**.

---

## 🧠 Solution Approach

The solution uses **Value Iteration**:

1. Initialize the value function `V(s)` for all states
2. Iteratively update values using the Bellman optimality equation:

   ```
   V(s) = max_a Q(s, a)
   ```
3. Stop when changes fall below a small threshold `θ`
4. Extract the optimal policy from the converged value function

The discount factor is set to:

```
γ = 1.0
```

---

## 📂 Project Structure

```
AI_MDP_Gamblers/
├── gamblers_experiment.py   # Runs value iteration and plots results
├── value_iteration.py       # Core value iteration and Q-value logic
```

---

## ▶️ How to Run

### Requirements

* Python 3.x
* NumPy
* Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the experiment:

```bash
python gamblers_experiment.py
```

---

## 📊 Output

The program prints:

* The **optimal policy** (stake at each capital)
* The **optimal value function**

It also generates a plot:

* **Capital vs Value Estimate**

This visualizes how the expected return evolves as the gambler’s capital increases.

---

## 🧪 Key Files Explained

### `value_iteration.py`

* Implements:

  * `calculate_q_values(s, V, rewards)`
  * `value_iteration_for_gamblers()`
* Encapsulates the MDP logic and convergence loop

### `gamblers_experiment.py`

* Runs the value iteration
* Prints policy and value function
* Plots the value function using Matplotlib

---

## 📚 Concepts Demonstrated

* Markov Decision Processes (MDP)
* Value Iteration
* Bellman Optimality Equation
* Policy extraction
* Probabilistic transitions
* Dynamic programming in reinforcement learning

---

## 🚀 Possible Extensions

* Change reward structure
* Add policy iteration
* Compare convergence speed for different `θ`
* Add Monte Carlo or TD learning versions
* Visualize policy (stake size vs capital)

---

## 📖 References

* Classical Gambler’s Problem (MDP)

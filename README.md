# Mimicking Human Behavior in the Stochastic Prisoner's Dilemma

This repository contains the implementation for **"Human Behavior Patterns in the Stochastic Prisoner's Dilemma"**, a research project conducted at Yale University by **Evangelia Chalioti** and **Himnish Hunma**.

The project investigates whether reinforcement learning agents can reproduce behavioral patterns observed in human subjects in an **Iterated Stochastic Prisoner's Dilemma (ISPD)**.

Specifically, we study whether Q-learning agents can learn to:

- cooperate when future outcomes appear favorable
- defect when future outcomes appear risky
- update beliefs after uncertainty resolves
- mimic optimism/pessimism patterns observed in human gameplay

---

## Motivation

Human players in repeated strategic environments often make decisions based on expectations about uncertain future outcomes. Prior work shows that in stochastic repeated Prisoner's Dilemma environments:

- players are more likely to cooperate when the probability of a less favorable future environment is low
- players become more likely to defect when future outcomes appear riskier

This repository explores whether these behavioral tendencies can emerge in **teacherless reinforcement learning agents** trained through repeated interaction.

If successful, such agents can:

- generate synthetic behavioral data at lower cost than human experiments
- serve as better approximations of real\-world human strategic behavior
- improve modeling of markets, negotiations, and multi\-agent systems

---

## The Stochastic Prisoner's Dilemma

Unlike the standard Prisoner's Dilemma, this environment contains **two possible payoff matrices**.

### Dilemma A (less cooperative)

|            | Cooperate | Defect |
|------------|-----------|--------|
| Cooperate  | (32, 32)  | (10, 52) |
| Defect     | (52, 10)  | (24, 24) |

### Dilemma B (more cooperative)

|            | Cooperate | Defect |
|------------|-----------|--------|
| Cooperate  | (62, 62)  | (10, 82) |
| Defect     | (82, 10)  | (24, 24) |

### Stochastic Process

At **period 0**:

- Dilemma A is selected with probability `0.5`
- Dilemma B is selected with probability `0.5`

At **period 1**:

- A permanent environment is selected using probability `p_A`

Where:

- `p_A` = probability that Dilemma A persists
- `1 - p_A` = probability that Dilemma B persists

After period 1:

- If A is selected → all future rounds use A
- If B is selected → all future rounds use B

This setup amplifies the role of future uncertainty while keeping the environment computationally tractable.

---

## Research Question

Can reinforcement learning agents reproduce two human behavioral traits observed in prior behavioral economics experiments?

### Trait 1: First\-period cooperation

Agents should:

- cooperate when `p_A` is low
- defect when `p_A` is high

### Trait 2: Belief updating

After uncertainty resolves:

- agents should revise beliefs
- switch behavior accordingly

Example:

- cooperate initially → defect if A is realized
- defect initially → cooperate if B is realized

---

## Reinforcement Learning Setup

We use a modified version of an open\-source Iterated Prisoner's Dilemma framework containing **230\+ strategies**.

The environment was modified to support:

- stochastic payoff transitions
- effectively infinite play
- richer state representations

### State Space

Each agent observes:

- expected cumulative rewards
- opponent action history (past 12 moves)
- opponent cooperation frequency
- payoff matrix metadata
- stochastic environment parameters (`p_A`)

### Action Space

Agents choose between:

- `C` → Cooperate
- `D` → Defect

### Reward Function

Reward = cumulative discounted payoff earned across repeated interactions.

### Q\-Learning Update Rule

\[ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right] \]

Where:

- `α = 0.9` → learning rate
- `γ = 0.33` → discount factor
- `ε = 0.9` → exploration parameter

---

## Training Procedure

Each training iteration consists of:

- 20 matches against 20 randomly sampled strategies

The agent retains learned Q\-values across iterations.

Training setup:

- 60 total iterations
- degenerate strategies removed (e.g. always defect, always cooperate)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run training:

```bash
python random_trainer.py
```
This will:
test the agent across 11 probability values: [0, 0.1, 0.2, ..., 1.0]
run 50 test matches per probability value
output cooperation rates and belief-updating statistics

Output includes:
First time coop: Number of first-period cooperative moves
First time defections: Number of first-period defection moves
Switching A to B: Transitions from cooperation to defection
Switching B to A: Transitions from defection to cooperation

## Results
**Period 0 Behavior**
The model successfully reproduces human-like strategic optimism:
low p_A → majority cooperation
high p_A → majority defection

**Period 1 Behavior**
Agents update their beliefs after uncertainty resolves:
cooperate → defect transitions increase as p_A rises
initial actions do not eliminate this trend

The superposition of two traits was however not observed with the training hyperparameters used. Future work will explore whether more extensive training or alternative algorithms can elicit this behavior.
## Citation
If you use this code in your research, please cite:

```
Chalioti, E., & Hunma, H. (2024). Mimicking Human Behavior in the Stochastic Prisoner’s Dilemma
```

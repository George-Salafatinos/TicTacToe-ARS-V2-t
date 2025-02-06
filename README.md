# TicTacToe ARS Implementation

This repository implements the Augmented Random Search (ARS) algorithm for training a TicTacToe agent, building on my previous implementation exploring REINFORCE and PPO algorithms.

## What is ARS?

ARS is described in [Mania et al. 2018](https://arxiv.org/abs/1803.07055) as a simple but effective alternative to policy gradient methods. Instead of using backpropagation, it:
1. Takes random directions in parameter space
2. Tests policy performance in both positive and negative directions 
3. Updates the policy using the reward differences

Key Features:
- Uses a linear policy (no hidden layers)
- Normalizes states using running statistics
- Straightforward implementation without complex optimization

## Implementation Details

### Core Components:
- Linear policy matrix M that maps states to move probabilities
- State normalization using running mean (μ) and covariance (Σ)
- Optional rollout averaging to reduce variance

### Hyperparameters:
- N: Number of random directions sampled per iteration
- b: Number of top-performing directions used for updates
- ν: Standard deviation of exploration noise
- α: Step size for parameter updates

## How to Use

The project uses a Flask web interface where you can:
1. Train new models with configurable parameters
2. Play against trained models
3. Save and load models
4. View training progress

## Observations

Some interesting findings from experiments:
- State normalization can help but requires careful tuning
- Policy can learn decent play in ~500 episodes but training is unstable
- Results suggest good policies exist but are hard to find reliably

## Future Work

Potential improvements:
- Add intermediate rewards (e.g. for two-in-rows)
- Try different state representations
- Implement more sophisticated exploration strategies
- Add proper co-play support

## Requirements
- Python 3.8+
- Flask
- NumPy
- Matplotlib

## Acknowledgments 

This builds on my previous work exploring REINFORCE and PPO for TicTacToe, adapting those ideas to the simpler ARS framework.
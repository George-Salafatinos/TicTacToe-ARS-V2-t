import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
import numpy as np
import json

from utils.tic_tac_toe import TicTacToeEnv, available_moves

class Model:
    def __init__(self, n=9, p=9):
        super().__init__()
        self.n = n
        self.p = p
        self.M = np.zeros((n, p)) # alg 2 line 2
        self.mu = np.zeros(9) # alg 2 line 2
        self.Sigma = np.identity(9) # alg 2 line 2
        self.step_number = 0 # alg 2 line 2
        self.nu = .1 # hyperparameter
        self.delta_k= 0 # positive or negative, randomly selected alg 2
    def predict(self, x):
        # Add small constant to prevent division by zero
        sigma_diag = np.diag(self.Sigma)
        sigma_diag = np.where(sigma_diag < 1e-8, 1.0, sigma_diag)
        normalized_x = np.diag(1.0 / np.sqrt(sigma_diag))@(x-self.mu)
        pi = (self.M + self.nu*self.delta_k)@normalized_x
        # Clip to prevent extreme values
        pi = np.clip(pi, -10, 10)
        return pi

########################################################
# board_to_tensor_for_o / board_to_tensor_for_x
########################################################

def board_to_state_vector(board):
    """
    If O is the agent:
    O => +1, X => -1, '' => 0
    """
    return np.array([1 if cell == 'O' else (-1 if cell == 'X' else 0) for cell in board])

########################################################
# Minimal moving average
########################################################

def moving_average(values, window_size=50):
    averaged = []
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_vals = values[start_index:i+1]
        avg = sum(window_vals) / len(window_vals)
        averaged.append(avg)
    return averaged


########################################################
# soft max
########################################################


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / (np.sum(exp_logits)+1e-10)

########################################################
# get_mask
########################################################
def get_mask(available_moves):
    mask = np.zeros(9)
    for move in available_moves:
        mask[move] = 1
    return mask

########################################################
# online_stats
########################################################
class OnlineVectorStats:
    def __init__(self, initial_mean=None, initial_cov=None, initial_count=0):
        """
        Initialize with optional known statistics
        
        Parameters:
        initial_mean: vector of means (p-dimensional)
        initial_cov: covariance matrix (p x p)
        initial_count: number of samples used for initial statistics
        """
        self.mean = initial_mean
        self.cov = initial_cov
        self.count = initial_count
        
    def update(self, new_data):
        """
        Update statistics with new batch of vector data
        
        Parameters:
        new_data: numpy array of shape (n_samples, n_features) where each row is a vector
        
        Returns:
        tuple of (new_mean_vector, new_covariance_matrix, new_count)
        """
        new_data = np.asarray(new_data)
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        
        new_count = len(new_data)
        
        if self.count == 0 or self.mean is None:
            # First batch
            new_mean = np.mean(new_data, axis=0)
            new_cov = np.cov(new_data.T, ddof=1) if new_count > 1 else np.zeros((new_data.shape[1], new_data.shape[1]))
            self.mean = new_mean
            self.cov = new_cov
            self.count = new_count
            return new_mean, new_cov, new_count
            
        # Compute new count
        combined_count = self.count + new_count
        
        # Update mean vector
        new_batch_mean = np.mean(new_data, axis=0)
        delta = new_batch_mean - self.mean
        new_mean = self.mean + (new_count * delta) / combined_count
        
        # Update covariance matrix
        if combined_count > 1:
            # Compute current batch covariance
            new_batch_cov = np.cov(new_data.T, ddof=1) if new_count > 1 else np.zeros_like(self.cov)
            
            # Compute combined covariance using vectorized operations
            old_SS = self.cov * (self.count - 1)  # Convert back to sum of squares
            new_SS = new_batch_cov * (new_count - 1)  # Convert back to sum of squares
            
            # Additional term for mean adjustment
            adjustment = np.outer(delta, delta) * (self.count * new_count) / combined_count
            
            # Combined sum of squares
            combined_SS = old_SS + new_SS + adjustment
            
            # New covariance
            new_cov = combined_SS / (combined_count - 1)
        else:
            new_cov = np.zeros_like(self.cov)
            
        self.mean = new_mean
        self.cov = new_cov
        self.count = combined_count
        
        return new_mean, new_cov, combined_count

########################################################
# train_reinforce
########################################################

def play_game(opponent, model_o, model_x=None):
    env = TicTacToeEnv()
    done = False
    reward_o = 0
    reward_x = 0
    board_state_vectors_o = []
    board_state_vectors_x = []

    def get_move(state, model, available_moves_list):
        try:
            prefs = model.predict(state)
            # Set large negative number for unavailable moves
            for i in range(9):
                if i not in available_moves_list:
                    prefs[i] = -1e6
            # Apply softmax
            probs = softmax(prefs)
            # If we get NaN, fall back to random
            if np.any(np.isnan(probs)):
                return random.choice(available_moves_list)
            return np.random.choice(9, p=probs)
        except:
            return random.choice(available_moves_list)

    while not done:
        # X moves first
        if opponent == "random":
            moves_x = available_moves(env.board)
            if moves_x:
                x_move = random.choice(moves_x)
                _, rew_x, done_x, info_x = env.step(x_move)
                if info_x.get("winner") == 'X':
                    #reward_o = -1  # Changed to -1 for loss
                    break
                if done_x:
                    reward_o = 0.5  # Draw
                    break

        elif opponent == "self-play":
            moves_x = available_moves(env.board)
            if moves_x:
                x_move = get_move(board_to_state_vector(env.board), model_x, moves_x)
                _, rew_x, done_x, info_x = env.step(x_move)
                if info_x.get("winner") == 'X':
                    reward_x = 1
                    #reward_o = -1
                    break
                if done_x:
                    reward_x = 0.5
                    reward_o = 0.5
                    break

        if env.is_done():
            break

        # O's Turn
        board_state_vectors_o.append(board_to_state_vector(env.board))
        moves_o = available_moves(env.board)
        if not moves_o:
            reward_o = 0.5  # Draw
            break

        o_move = get_move(board_to_state_vector(env.board), model_o, moves_o)
        _, rew_o, done_o, info_o = env.step(o_move)
        
        if info_o.get("winner") == 'O':
            reward_o = 1
            break
        elif done_o:
            reward_o = 0.5
            break

    return {
        "reward_o": reward_o,
        "reward_x": reward_x,
        "winner": info_o.get("winner") if 'info_o' in locals() else None,
        "board_state_vectors_o": board_state_vectors_o,
        "board_state_vectors_x": board_state_vectors_x
    }

def train_ARS_V2t(
    steps=10,
    lr=0.01,
    gamma=0.99,
    opponent="random",
    model_name="unnamed",
):
    # only supports random and self-play at the moment

    # Create the policy net for O
    model_o = Model(n=9, p=9)

    # If co-play, we need a separate net for X (not supported yet)
    if opponent == "co-play":
        model_x = Model(n=9, p=9)
    else:
        model_x = None

    scores_o = []   # O's sliding average of wins
    losses_o = []   # O's policy losses
    total_wins_o = 0.0  # count how many times O wins
    N = 15 # alg 2
    b = 15 # alg 2
    alpha = lr # step size
    episode_sampling_size = 10 # will run the trajectory with the same parameters this many times due to game randomness

    for episode in range(steps):
        # Store deltas and rewards in parallel lists
        delta_ks = []  # All deltas (positive and negative)
        rewards = []   # Corresponding rewards
        unique_deltas = []  # Original deltas for sorting

        # Generate deltas
        for k in range(N):
            delta_k = np.random.standard_normal(size=(9, 9))
            unique_deltas.append(delta_k)
            delta_ks.extend([delta_k, -delta_k])
            rewards.extend([0, 0])

        # Collect rewards for each delta
        for i, delta_k in enumerate(delta_ks):
            reward_averaged = 0
            for game in range(episode_sampling_size):
                model_o.delta_k = delta_k
                result = play_game(opponent, model_o, model_x)
                reward_o = result["reward_o"]
                board_state_vectors_o = result["board_state_vectors_o"]

                # add reward
                reward_averaged += reward_o/episode_sampling_size

                #update stats: alg 2
                initial_stats = OnlineVectorStats(initial_mean=model_o.mu, 
                                                initial_cov=model_o.Sigma, 
                                                initial_count=model_o.step_number)
                model_o.mu, model_o.Sigma, model_o.step_number = initial_stats.update(board_state_vectors_o)

            rewards[i] = reward_averaged

        ########################################################
        # End of episode => compute returns & do updates, ARS_V2t
        ########################################################

        # alg 2 line 6: get max rewards for each unique delta
        max_rewards = {i: max(rewards[2*i], rewards[2*i+1]) 
                      for i in range(len(unique_deltas))}
        
        # Sort unique deltas by their max rewards
        sorted_indices = sorted(range(len(unique_deltas)), 
                              key=lambda x: max_rewards[x], 
                              reverse=True)
        
        # Get top rewards for sigma_R calculation
        top_rewards = []
        update_sum = np.zeros_like(model_o.M)
        
        # Process top b directions
        for i in sorted_indices[:b]:
            pos_reward = rewards[2*i]
            neg_reward = rewards[2*i+1]
            delta = unique_deltas[i]
            top_rewards.extend([pos_reward, neg_reward])
            update_sum += (pos_reward - neg_reward) * delta

        sigma_R = np.std(top_rewards, ddof=1) / np.sqrt(len(top_rewards))
        
        # Update M
        model_o.M = model_o.M + alpha/(b*sigma_R) * update_sum

        policy_loss = 0  # not exactly defined here
        total_wins_o+= max_rewards[sorted_indices[0]]
        avg_score_o = total_wins_o / (episode + 1)
        scores_o.append(avg_score_o)
        losses_o.append(policy_loss)
        if episode % 200 == 0:
            print(episode, model_o.mu, model_o.Sigma)
            

    # sliding-average the scores from O's perspective
    scores_o = moving_average(scores_o, window_size=50)

    # Save O's net to disk
    timestamp = int(time.time())
    filename = f"{model_name}_ars_{timestamp}.npz"
    model_path = os.path.join("saved_models", filename)
    np.savez(model_path, 
             M=model_o.M,
             mu=model_o.mu,
             Sigma=model_o.Sigma,
             step_number=model_o.step_number)

    model_data = {
        "algorithm": "ars",
        "model_path": model_path
    }
    return model_data, (scores_o, losses_o)

########################################################

def predict_ars_v2t(board, model_data):
    """
    Inference for O using ARS model.
    """
    # Load the saved model
    model_path = model_data["model_path"]
    saved_model = np.load(model_path)
    
    # Create a new model and load its parameters
    model = Model(n=9, p=9)
    model.M = saved_model['M']
    model.mu = saved_model['mu']
    model.Sigma = saved_model['Sigma']
    model.step_number = saved_model['step_number']
    model.delta_k = 0  # No exploration during inference
    
    # Convert board to state vector
    state = board_to_state_vector(board)
    
    # Get action preferences
    action_prefs = model.predict(state)
    
    # Get available moves and mask
    moves = available_moves(board)
    if not moves:
        return None
        
    mask = get_mask(moves)
    
    # Apply mask BEFORE softmax - set unavailable moves to large negative number
    action_prefs = action_prefs - 1e6 * (1 - mask)  # This makes unavailable moves have very negative preference
    probs = softmax(action_prefs)
    
    # Sample move
    return np.random.choice(9, p=probs)
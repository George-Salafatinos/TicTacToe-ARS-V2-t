import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os

from utils.tic_tac_toe import TicTacToeEnv, available_moves

########################################################
# CustomPolicyNet that supports arbitrary hidden_sizes
########################################################

class CustomPolicyNet(nn.Module):
    """
    A flexible policy network for Tic-Tac-Toe, allowing different hidden layer sizes.
    Example:
        hidden_sizes=[32] -> (9 -> 32 -> 9).
        hidden_sizes=[32, 32] -> (9 -> 32 -> 32 -> 9), etc.
    """
    def __init__(self, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]

        layers = []
        input_dim = 9  # 9 board cells
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 9))  # final output: 9 logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

########################################################
# board_to_tensor_for_o / board_to_tensor_for_x
########################################################

def board_to_tensor_for_o(board):
    """
    If O is the agent:
    O => +1, X => -1, '' => 0
    """
    arr = []
    for cell in board:
        if cell == 'O':
            arr.append(1)
        elif cell == 'X':
            arr.append(-1)
        else:
            arr.append(0)
    return torch.tensor(arr, dtype=torch.float)

def board_to_tensor_for_x(board):
    """
    If X is the agent:
    X => +1, O => -1, '' => 0
    """
    arr = []
    for cell in board:
        if cell == 'X':
            arr.append(1)
        elif cell == 'O':
            arr.append(-1)
        else:
            arr.append(0)
    return torch.tensor(arr, dtype=torch.float)

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
# train_reinforce
########################################################

def train_reinforce(
    steps=10,
    lr=0.01,
    gamma=0.99,
    model_name="unnamed",
    opponent="random",
    hidden_sizes=None
):
    """
    We train an agent O. Opponent can be:
      - 'random'   => X is random
      - 'self-play' => X is the same net as O (common approach: but here we keep it random for X in minimal code)
      - 'co-play'  => X is a *separate* net also being trained at the same time.
    hidden_sizes: a list of integers for layer sizes, e.g. [32], [32, 32], etc.

    Returns:
      model_data (dict) => { "algorithm": "reinforce", "model_path": ... }
      (scores, losses)  => final arrays for each episode (from O's perspective).
    """
    if hidden_sizes is None:
        hidden_sizes = [32]

    # Create the policy net for O
    policy_o = CustomPolicyNet(hidden_sizes=hidden_sizes)
    optimizer_o = optim.Adam(policy_o.parameters(), lr=lr)


    # If co-play, we need a separate net for X
    if opponent == "co-play":
        policy_x = CustomPolicyNet(hidden_sizes=hidden_sizes)
        optimizer_x = optim.Adam(policy_x.parameters(), lr=lr)
    else:
        policy_x = None
        optimizer_x = None

    env = TicTacToeEnv()
    scores_o = []   # O's sliding average of wins
    losses_o = []   # O's policy losses
    total_wins_o = 0.0  # count how many times O wins

    for episode in range(steps):
        # We'll track O's transitions:
        log_probs_o = []
        rewards_o = []

        # If co-play, track X's transitions too:
        log_probs_x = []
        rewards_x = []

        env.reset()
        done = False

        while not done:
            # X moves first
            if opponent == "random":
                # random X
                moves_x = available_moves(env.board)
                if moves_x:
                    x_move = random.choice(moves_x)
                    _, rew_x, done_x, info_x = env.step(x_move)
                    if info_x.get("winner") == 'X':
                        # O loses => reward_o=-1
                        rewards_o.append(-1)
                        log_probs_o.append(torch.tensor(0.0))
                        break
                    if done_x:
                        # draw
                        rewards_o.append(0)
                        log_probs_o.append(torch.tensor(0.0))
                        break

            elif opponent == "self-play":
                # Minimal code previously had random X as well.
                # We'll keep it random unless you prefer the same net for X:
                moves_x = available_moves(env.board)
                if moves_x:
                    x_move = random.choice(moves_x)
                    _, rew_x, done_x, info_x = env.step(x_move)
                    if info_x.get("winner") == 'X':
                        rewards_o.append(-1)
                        log_probs_o.append(torch.tensor(0.0))
                        break
                    if done_x:
                        rewards_o.append(0)
                        log_probs_o.append(torch.tensor(0.0))
                        break

            elif opponent == "co-play":
                # co-play => X is a separate net being trained.
                state_x = board_to_tensor_for_x(env.board).unsqueeze(0)
                logits_x = policy_x(state_x)
                moves_x = available_moves(env.board)
                if not moves_x:
                    # it would be a draw
                    rewards_o.append(0)
                    log_probs_o.append(torch.tensor(0.0))

                    rewards_x.append(0)
                    log_probs_x.append(torch.tensor(0.0))
                    break

                mask_x = torch.zeros(9)
                for mx in moves_x:
                    mask_x[mx] = 1
                masked_x = logits_x + (mask_x.unsqueeze(0) - 1) * 1e6
                dist_x = torch.distributions.Categorical(logits=masked_x[0])
                action_x = dist_x.sample()
                log_prob_x = dist_x.log_prob(action_x)

                _, rew_x, done_x, info_x = env.step(action_x.item())

                # If X wins => O gets -1, X gets +1
                if info_x.get("winner") == 'X':
                    rewards_o.append(-1)
                    log_probs_o.append(torch.tensor(0.0))
                    rewards_x.append(1)
                    log_probs_x.append(log_prob_x)
                    break
                elif done_x:
                    # draw
                    rewards_o.append(0)
                    log_probs_o.append(torch.tensor(0.0))
                    rewards_x.append(0)
                    log_probs_x.append(log_prob_x)
                    break
                else:
                    # game continues
                    log_probs_x.append(log_prob_x)
                    rewards_x.append(0)

            if env.is_done():
                break

            # O's turn
            state_o = board_to_tensor_for_o(env.board).unsqueeze(0)
            logits_o_ = policy_o(state_o)
            moves_o = available_moves(env.board)
            if not moves_o:
                # draw
                rewards_o.append(0)
                log_probs_o.append(torch.tensor(0.0))
                if opponent == "co-play":
                    # X's final reward is 0 as well
                    rewards_x.append(0)
                    log_probs_x.append(torch.tensor(0.0))
                break

            mask_o = torch.zeros(9)
            for mo in moves_o:
                mask_o[mo] = 1
            masked_logits_o = logits_o_ + (mask_o.unsqueeze(0) - 1) * 1e6
            dist_o = torch.distributions.Categorical(logits=masked_logits_o[0])
            action_o = dist_o.sample()
            log_prob_o_ = dist_o.log_prob(action_o)

            _, rew_o, done_o, info_o = env.step(action_o.item())
            if info_o.get("winner") == 'O':
                # O wins
                rewards_o.append(1)
                log_probs_o.append(log_prob_o_)
                if opponent == "co-play":
                    # X gets -1
                    rewards_x.append(-1)
                    log_probs_x.append(torch.tensor(0.0))
                break
            elif done_o:
                # draw
                rewards_o.append(0)
                log_probs_o.append(log_prob_o_)
                if opponent == "co-play":
                    rewards_x.append(0)
                    log_probs_x.append(torch.tensor(0.0))
                break
            else:
                # game continues
                log_probs_o.append(log_prob_o_)
                rewards_o.append(0)
                if opponent == "co-play":
                    # no immediate reward for X
                    # do nothing additional here
                    pass

        ########################################################
        # End of episode => compute returns & do updates
        ########################################################
        # O's update
        returns_o = []
        G_o = 0
        for r in reversed(rewards_o):
            G_o = r + gamma * G_o
            returns_o.insert(0, G_o)
        returns_o = torch.tensor(returns_o, dtype=torch.float)
        if len(returns_o) > 1:
            returns_o = (returns_o - returns_o.mean()) / (returns_o.std() + 1e-8)

        policy_loss_o = torch.tensor(0.0)
        for lp, R in zip(log_probs_o, returns_o):
            policy_loss_o = policy_loss_o + (-lp * R)

        optimizer_o.zero_grad()
        policy_loss_o.backward()
        optimizer_o.step()

        # X's update if co-play
        if opponent == "co-play":
            returns_x = []
            G_x = 0
            for r in reversed(rewards_x):
                G_x = r + gamma * G_x
                returns_x.insert(0, G_x)
            returns_x = torch.tensor(returns_x, dtype=torch.float)
            if len(returns_x) > 1:
                returns_x = (returns_x - returns_x.mean()) / (returns_x.std() + 1e-8)

            policy_loss_x = torch.tensor(0.0)
            for lp, R in zip(log_probs_x, returns_x):
                policy_loss_x = policy_loss_x + (-lp * R)

            optimizer_x.zero_grad()
            policy_loss_x.backward()
            optimizer_x.step()

        # final reward for O
        final_r_o = rewards_o[-1] if len(rewards_o) > 0 else 0
        if final_r_o > 0:
            total_wins_o += 1
        avg_score_o = total_wins_o / (episode + 1)
        scores_o.append(avg_score_o)
        losses_o.append(policy_loss_o.item())

    # sliding-average the scores from O's perspective
    scores_o = moving_average(scores_o, window_size=50)

    # Save O's net to disk
    timestamp = int(time.time())
    filename = f"{model_name}_reinforce_{timestamp}.pth"
    model_path = os.path.join("saved_models", filename)
    torch.save(policy_o.state_dict(), model_path)

    model_data = {
        "algorithm": "reinforce",
        "model_path": model_path
    }
    return model_data, (scores_o, losses_o)

########################################################

def predict_reinforce(board, model_data):
    """
    Inference for O.
    """
    policy_o = CustomPolicyNet(hidden_sizes=[32])  # or read from model_data if you store hidden_sizes
    model_path = model_data["model_path"]
    policy_o.load_state_dict(torch.load(model_path))
    policy_o.eval()

    state_o = board_to_tensor_for_o(board).unsqueeze(0)
    logits_o = policy_o(state_o)
    moves_o = available_moves(board)
    if not moves_o:
        return None

    mask_o = torch.zeros(9)
    for mo in moves_o:
        mask_o[mo] = 1
    masked_logits_o = logits_o + (mask_o.unsqueeze(0) - 1) * 1e6
    dist_o = torch.distributions.Categorical(logits=masked_logits_o[0])
    action_o = dist_o.sample()
    return action_o.item()

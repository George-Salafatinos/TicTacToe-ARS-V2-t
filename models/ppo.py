import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from utils.tic_tac_toe import TicTacToeEnv, available_moves

class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def board_to_tensor_for_o(board):
    arr = []
    for cell in board:
        if cell == 'O':
            arr.append(1)
        elif cell == 'X':
            arr.append(-1)
        else:
            arr.append(0)
    return torch.tensor(arr, dtype=torch.float)

def collect_trajectory(env, policy, gamma, opponent="random"):
    """
    Collect one 'episode' of experience for agent O.
    Return lists: states, actions, rewards, log_probs, final_return.
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    done = False

    while not done:
        # X moves first
        if opponent == "random":
            moves_x = available_moves(env.board)
            if moves_x:
                x_act = random.choice(moves_x)
                _, rx, done_x, info_x = env.step(x_act)
                if info_x.get("winner") == 'X':
                    # O lost
                    rewards.append(-1)
                    states.append(None)
                    actions.append(None)
                    log_probs.append(None)
                    break
                if done_x:
                    rewards.append(0)
                    states.append(None)
                    actions.append(None)
                    log_probs.append(None)
                    break
        elif opponent == "self-play":
            # Minimally, still random for X
            moves_x = available_moves(env.board)
            if moves_x:
                x_act = random.choice(moves_x)
                _, rx, done_x, info_x = env.step(x_act)
                if info_x.get("winner") == 'X':
                    rewards.append(-1)
                    states.append(None)
                    actions.append(None)
                    log_probs.append(None)
                    break
                if done_x:
                    rewards.append(0)
                    states.append(None)
                    actions.append(None)
                    log_probs.append(None)
                    break

        if env.is_done():
            break

        # O's turn
        s_tensor = board_to_tensor_for_o(env.board)
        moves_o = available_moves(env.board)
        if not moves_o:
            rewards.append(0)
            states.append(None)
            actions.append(None)
            log_probs.append(None)
            break

        with torch.no_grad():
            logits = policy(s_tensor.unsqueeze(0))
            mask = torch.zeros(9)
            for mo in moves_o:
                mask[mo] = 1
            masked_logits = logits + (mask.unsqueeze(0) - 1) * 1e6
            dist = torch.distributions.Categorical(logits=masked_logits[0])
            action = dist.sample()
            logp = dist.log_prob(action)

        states.append(s_tensor)
        actions.append(action)
        log_probs.append(logp)

        _, r_o, d_o, i_o = env.step(action.item())
        if i_o.get("winner") == 'O':
            rewards.append(1)
            break
        elif d_o:
            # draw
            rewards.append(0)
            break
        else:
            rewards.append(0)

        if env.is_done():
            break

    # Convert to returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return states, actions, log_probs, returns

def train_ppo(steps=10, lr=0.001, gamma=0.99, model_name="unnamed", opponent="random"):
    """
    Very minimal PPO-like training for O. For demonstration only.
    We'll do 1 episode per update, no batch, just to keep it short.
    """
    policy = PPONet()
    old_policy = PPONet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    eps_clip = 0.2  # typical PPO clip range
    scores = []
    losses = []
    env = TicTacToeEnv()
    total_score = 0.0

    for episode in range(steps):
        env.reset()
        old_policy.load_state_dict(policy.state_dict())
        states, actions, log_probs_old, returns = collect_trajectory(env, old_policy, gamma, opponent)

        # Filter out any None entries from partial data
        valid_data = [(s, a, lp, rt) for (s,a,lp,rt) in zip(states, actions, log_probs_old, returns)
                      if s is not None and a is not None and lp is not None]
        if len(valid_data) == 0:
            # means we ended quickly (X might have won)
            final_r = returns[-1] if len(returns) else 0
            if final_r > 0:
                total_score += 1
            avg_score = total_score / (episode + 1)
            scores.append(avg_score)
            losses.append(0.0)
            continue

        s_tensors = torch.stack([d[0] for d in valid_data])
        a_tensors = torch.stack([d[1] for d in valid_data])
        logp_olds = torch.stack([d[2] for d in valid_data]).detach()
        R_vals = torch.tensor([d[3] for d in valid_data], dtype=torch.float)

        if len(R_vals) > 1:
            R_vals = (R_vals - R_vals.mean()) / (R_vals.std() + 1e-8)

        # PPO update
        logits = policy(s_tensors)
        masks = []
        for i, st in enumerate(s_tensors):
            # figure out valid moves from st
            env_state = st.view(-1).numpy()
            # not exactly the same as env's board, but we can do a quick mask
            # we know: O->+1, X->-1, ''->0
            # let's do it the simpler way: if env_state[i]==0, it's valid
            # but we don't have the exact index
            # To keep it minimal, skip exact mask & just compute log prob of chosen action
            pass
        # We'll do a simpler approach: we compute log prob of chosen action from logits
        dist_new = torch.distributions.Categorical(logits=logits)
        logp_news = dist_new.log_prob(a_tensors)

        ratio = torch.exp(logp_news - logp_olds)  # ratio for PPO
        surr1 = ratio * R_vals
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * R_vals
        policy_loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Score tracking
        final_r = returns[-1] if len(returns) else 0
        if final_r > 0:
            total_score += 1
        avg_score = total_score / (episode + 1)
        scores.append(avg_score)
        losses.append(policy_loss.item())

    # Save model
    timestamp = int(time.time())
    filename = f"{model_name}_ppo_{timestamp}.pth"
    model_path = os.path.join("saved_models", filename)
    torch.save(policy.state_dict(), model_path)

    model_data = {
        "algorithm": "ppo",
        "model_path": model_path
    }
    # Return (scores, losses)
    return model_data, (scores, losses)

def predict_ppo(board, model_data):
    policy = PPONet()
    policy.load_state_dict(torch.load(model_data["model_path"]))
    policy.eval()

    s_tensor = board_to_tensor_for_o(board)
    moves = available_moves(board)
    if not moves:
        return None

    with torch.no_grad():
        logits = policy(s_tensor.unsqueeze(0))
        mask = torch.zeros(9)
        for m in moves:
            mask[m] = 1
        masked_logits = logits + (mask.unsqueeze(0) - 1) * 1e6
        dist = torch.distributions.Categorical(logits=masked_logits[0])
        action = dist.sample()
    return action.item()

import random
from utils.tic_tac_toe import TicTacToeEnv, available_moves

def moving_average(values, window_size=50):
    """Compute a rolling average for the given list of values."""
    averaged = []
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_vals = values[start_index:i+1]
        avg = sum(window_vals) / len(window_vals)
        averaged.append(avg)
    return averaged

def train_random_search(steps=10):
    env = TicTacToeEnv()
    avg_scores = []
    best_score = -999
    best_params = {"random_seed": 42}
    total_score = 0.0

    for i in range(steps):
        candidate_seed = random.randint(0, 100000)
        random.seed(candidate_seed)
        env.reset()
        done = False
        while not done:
            moves = available_moves(env.board)
            if not moves:
                break
            action = random.choice(moves)
            next_state, reward, done, info = env.step(action)

        winner = info.get("winner", None)
        score = 1 if winner == 'X' else 0
        total_score += score
        current_avg = total_score / (i + 1)
        avg_scores.append(current_avg)

        if score > best_score:
            best_score = score
            best_params = {"random_seed": candidate_seed}

    # Apply a rolling average to smooth the plot
    avg_scores = moving_average(avg_scores, window_size=50)

    trained_model = {
        "algorithm": "random-search",
        "best_params": best_params,
        "best_score": best_score
    }
    return trained_model, avg_scores

def predict_random_search(board, model_data):
    best_seed = model_data.get("best_params", {}).get("random_seed", 42)
    random.seed(best_seed)
    moves = available_moves(board)
    if not moves:
        return None
    return random.choice(moves)

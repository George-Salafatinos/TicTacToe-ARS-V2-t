import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_curve(metrics, title="Training Curve"):
    fig, ax = plt.subplots()
    ax.plot(metrics, marker='o', linestyle='-', color='blue', label='Score')
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1])
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    buf.close()
    return img_base64

def _sliding_average(values, window_size=50):
    averaged = []
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        subset = values[start_index:i+1]
        avg = sum(subset) / len(subset)
        averaged.append(avg)
    return averaged

def plot_score_and_loss(scores, losses, title="RL Training"):
    """
    Two-subplot figure:
      Top: Score (scores)
      Bottom: Policy Loss (losses), after sliding average
    """
    # Also smooth the losses for plotting
    smooth_losses = _sliding_average(losses, window_size=50)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))
    fig.suptitle(title)

    # Score subplot
    ax1.plot(scores, marker='o', color='blue', label='Avg Score')
    ax1.set_ylabel("Score")
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True)

    # Loss subplot
    ax2.plot(smooth_losses, marker='o', color='red', label='Policy Loss (smoothed)')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    plt.close(fig)
    buf.close()
    return img_base64

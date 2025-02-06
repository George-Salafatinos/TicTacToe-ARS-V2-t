#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from statistics import mean

# Import your revised "reinforce" with co-play logic
from models.reinforce import train_reinforce

def run_single_experiment(steps, lr, gamma, hidden_sizes=None):
    if hidden_sizes is None:
        hidden_sizes = [32]
    model_name = "temp_sweep_coplay_" + str(time.time())

    model_data, (scores, losses) = train_reinforce(
        steps=steps,
        lr=lr,
        gamma=gamma,
        model_name=model_name,
        opponent='co-play',  # <--- co-play specified here
        hidden_sizes=hidden_sizes
    )
    final_acc = scores[-1] if len(scores) > 0 else 0.0
    return final_acc, scores

def multi_run_final_accuracy(steps, lr, gamma, hidden_sizes=None, n_runs=10):
    finals = []
    for _ in range(n_runs):
        final_acc, _ = run_single_experiment(steps, lr, gamma, hidden_sizes)
        finals.append(final_acc)
    return mean(finals)

def multi_run_average_scores(steps, lr, gamma, hidden_sizes=None, n_runs=10):
    all_runs_scores = []
    min_length = None
    for _ in range(n_runs):
        _, scores = run_single_experiment(steps, lr, gamma, hidden_sizes)
        if min_length is None or len(scores) < min_length:
            min_length = len(scores)
        all_runs_scores.append(scores)
    for i in range(len(all_runs_scores)):
        all_runs_scores[i] = all_runs_scores[i][:min_length]
    avg_scores = []
    for ep in range(min_length):
        vals = [all_runs_scores[i][ep] for i in range(len(all_runs_scores))]
        avg_scores.append(mean(vals))
    return avg_scores

def main():
    if not os.path.exists("hyperparam_plots_coplay"):
        os.makedirs("hyperparam_plots_coplay")

    n_runs = 10

    # 1) final avg accuracy vs LR (steps=800, gamma=0.85)
    lrs_1 = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    results_1 = []
    for lr in lrs_1:
        avg_acc = multi_run_final_accuracy(steps=800, lr=lr, gamma=0.85, n_runs=n_runs)
        results_1.append(avg_acc)
    plt.figure()
    plt.plot(lrs_1, results_1, marker='o')
    plt.title(f"(Co-Play) Final Avg Accuracy vs LR (steps=800, gamma=0.85)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Avg Accuracy")
    plt.xscale("log")
    plt.ylim([0,1])
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_1_lr_gamma085.png")
    plt.close()

    # 2) final avg accuracy vs LR (steps=800, gamma=0.99)
    lrs_2 = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    results_2 = []
    for lr in lrs_2:
        avg_acc = multi_run_final_accuracy(steps=800, lr=lr, gamma=0.99, n_runs=n_runs)
        results_2.append(avg_acc)
    plt.figure()
    plt.plot(lrs_2, results_2, marker='o')
    plt.title(f"(Co-Play) Final Avg Accuracy vs LR (steps=800, gamma=0.99)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Avg Accuracy")
    plt.xscale("log")
    plt.ylim([0,1])
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_2_lr_gamma099.png")
    plt.close()

    # 3) final avg accuracy vs gamma (steps=800, lr=0.01)
    gammas_3 = [0.5, 0.75, 0.85, 0.9, 0.95, 0.99]
    results_3 = []
    for g in gammas_3:
        avg_acc = multi_run_final_accuracy(steps=800, lr=0.01, gamma=g, n_runs=n_runs)
        results_3.append(avg_acc)
    plt.figure()
    plt.plot(gammas_3, results_3, marker='o')
    plt.title(f"(Co-Play) Final Avg Accuracy vs Gamma (steps=800, lr=0.01)")
    plt.xlabel("Gamma")
    plt.ylabel("Final Avg Accuracy")
    plt.ylim([0,1])
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_3_gamma_lr001.png")
    plt.close()

    # 4) final avg accuracy vs gamma (steps=800, lr=0.08)
    gammas_4 = [0.5, 0.75, 0.85, 0.9, 0.95, 0.99]
    results_4 = []
    for g in gammas_4:
        avg_acc = multi_run_final_accuracy(steps=800, lr=0.08, gamma=g, n_runs=n_runs)
        results_4.append(avg_acc)
    plt.figure()
    plt.plot(gammas_4, results_4, marker='o')
    plt.title(f"(Co-Play) Final Avg Accuracy vs Gamma (steps=800, lr=0.08)")
    plt.xlabel("Gamma")
    plt.ylabel("Final Avg Accuracy")
    plt.ylim([0,1])
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_4_gamma_lr008.png")
    plt.close()

    # 5) final avg accuracy vs hidden layer size (steps=800, lr=0.01, gamma=0.85)
    hidden_sizes_list_5 = [16, 32, 64, 128, 256]
    results_5 = []
    for hs in hidden_sizes_list_5:
        avg_acc = multi_run_final_accuracy(
            steps=800, lr=0.01, gamma=0.85,
            hidden_sizes=[hs],
            n_runs=n_runs
        )
        results_5.append(avg_acc)
    plt.figure()
    plt.plot(hidden_sizes_list_5, results_5, marker='o')
    plt.title(f"(Co-Play) Final Avg Accuracy vs Hidden Layer Size\n(steps=800, lr=0.01, gamma=0.85)")
    plt.xlabel("Hidden Layer Nodes")
    plt.ylabel("Final Avg Accuracy")
    plt.ylim([0,1])
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_5_hidden_sizes.png")
    plt.close()

    # 6) multi-line chart for net variants (9->32->9, 9->32->32->9, 9->32->32->32->9)
    # steps=1200, lr=0.01, gamma=0.85
    arch_list_6 = [
        [32],         # 9->32->9
        [32, 32],     # 9->32->32->9
        [32, 32, 32]  # 9->32->32->32->9
    ]
    plt.figure()
    for arch in arch_list_6:
        avg_scores = multi_run_average_scores(
            steps=1200, lr=0.01, gamma=0.85, hidden_sizes=arch, n_runs=n_runs
        )
        x_vals = np.arange(len(avg_scores))
        label_str = f"{[9]+arch+[9]}"
        plt.plot(x_vals, avg_scores, label=label_str)
    plt.title(f"(Co-Play) Score vs Time (Averaged Over {n_runs} Runs)\nNet Archs (1200 steps, lr=0.01, gamma=0.85)")
    plt.xlabel("Episode")
    plt.ylabel("Score (sliding avg for O)")
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.savefig("hyperparam_plots_coplay/hyper_6_arch.png")
    plt.close()

    print(f"(Co-Play) All hyperparameter plots (with {n_runs} runs each) saved in 'hyperparam_plots_coplay' folder.")

if __name__ == "__main__":
    main()

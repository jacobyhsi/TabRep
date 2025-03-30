import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def summarise_mle_results():
    # steps = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1000]
    # steps = list(range(2000, 100001, 2000))
    steps = [500, 1000] + list(range(2000, 30001, 2000))
    # methods = ['tabrep_ddpm']
    # methods = ['tabrep_ddpm', 'tabrep_flow', 'tabsyn', 'tabddpm']
    # dataset = 'adult'
    dataset = 'adult'
    methods = ['tabrep_ddpm', 'tabrep_flow', 'tabsyn', 'tabddpm']
    colors = ['blue', 'red', 'orange', 'green']
    method2name = {
        'tabrep_ddpm': 'TabRep-DDPM',
        'tabrep_flow': 'TabRep-Flow',
        'tabsyn': 'TabSYN',
        'tabddpm': 'TabDDPM',
    }

    results = []
    for method in methods:
        aurocs = []
        for step in steps:
            try:
                with open(f'mle/{dataset}/{method}_{step}.json') as f:
                    scores = json.load(f)
                    auroc = scores['best_auroc_scores']['XGBClassifier']['roc_auc']
                    aurocs.append(auroc)
            except FileNotFoundError:
                # print(f'File not found: mle/diabetes/{method}_{step}.json')
                aurocs.append(None)  # Append None for missing files
        results.append(aurocs)

    # plot

    # Main plot
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, method in enumerate(methods):
        ax.plot(steps, results[i], label=method2name[method], color=colors[i])

    ax.set_xlabel(r'Epoch ($\times$ 1K)', fontsize=30)
    ax.set_ylabel('AUC', fontsize=30)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(framealpha=0.4, frameon=True, fontsize=24, loc='center right')
    ax.set_ylim(0.30, 0.95)
    # xtick_labels = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
    xtick_labels_in_k = [".5", 1, 5, 10, 15, 20, 25, 30]
    xtick_positions = [300, 1200, 5000, 10000, 15000, 20000, 25000, 30000]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{label}' for label in xtick_labels_in_k], fontsize=17)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=26)

    # Inset zoom region
    zoom_start = 500
    zoom_end = 10000
    zoom_indices = [i for i, step in enumerate(steps) if zoom_start <= step <= zoom_end]

    axins = ax.inset_axes([0.1, 0.6, 0.35, 0.2])  # [x, y, width, height]
    for i, method in enumerate(methods):
        axins.plot([steps[j] for j in zoom_indices], 
                [results[i][j] for j in zoom_indices], 
                label=method2name[method], 
                color=colors[i])

    axins.set_xlim(500, 9000)
    axins.set_ylim(0.855, 0.918)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="black")

    print('saving fig')
    plt.savefig(f'training_progress_adult.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)

if __name__ == '__main__':
    summarise_mle_results()

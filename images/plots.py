import matplotlib.pyplot as plt
import numpy as np

layers = list(range(13))
r2_scores = [0.420, 0.448, 0.487, 0.510, 0.529, 0.548, 0.547, 0.538, 0.541, 0.550, 0.550, 0.554, 0.556]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

ax.plot(layers, r2_scores, marker='o', color="#9EB9F3", linewidth=2.5, markersize=6, label='R² score')

ax.scatter([12], [0.556], color='#F6CF71', s=100, zorder=5, label='$L_{best}$')

ax.set_title('Layer-wise Probing', fontsize=12, pad=15, fontweight='bold')
ax.set_xlabel('Hidden layer $L$', fontsize=10)
ax.set_ylabel('$R^2$', fontsize=10)
ax.set_xticks(layers)
ax.set_ylim(0.40, 0.58)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('images/probing_r2.png')
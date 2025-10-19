import matplotlib.pyplot as plt
import numpy as np

# Create figure with your actual data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Left: Performance comparison
platforms = ['IQM\nEmerald', 'Rigetti\nAnkaa-3', 'IonQ\nForte-1']
performance = [94.1, 71.0, 100.0]
theoretical = [39.6, 39.6, 39.6]
colors = ['#2E86AB', '#A23B72', '#F18F01']

x = np.arange(len(platforms))
width = 0.35

bars1 = ax1.bar(x - width/2, performance, width, label='Optimized (γ=0.25, β=1.25)', 
                color=colors, alpha=0.8)
bars2 = ax1.bar(x + width/2, theoretical, width, label='Theoretical (γ=π/4, β=π/8)', 
                color='gray', alpha=0.5)

ax1.set_ylabel('Approximation Ratio (%)')
ax1.set_title('Cross-Platform QAOA Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(platforms)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 110)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

# Middle: Architecture comparison
arch_data = {
    'Superconducting': [94.1, 71.0],
    'Trapped Ion': [100.0]
}

positions = [1, 2]
for i, (arch, values) in enumerate(arch_data.items()):
    for val in values:
        ax2.scatter(positions[i], val, s=200, alpha=0.6, 
                   color=['#2E86AB', '#A23B72'][i])
    ax2.scatter(positions[i], np.mean(values), s=300, 
               color=['#2E86AB', '#A23B72'][i], 
               marker='_', linewidths=3)

ax2.set_xticks(positions)
ax2.set_xticklabels(['Superconducting\n(n=2)', 'Trapped Ion\n(n=1)'])
ax2.set_ylabel('Approximation Ratio (%)')
ax2.set_title('Performance by Architecture')
ax2.set_ylim(60, 105)
ax2.grid(True, alpha=0.3, axis='y')

# Right: Improvement factors
improvements = [94.1/39.6, 71.0/39.6, 100.0/39.6]
bars3 = ax3.bar(platforms, improvements, color=colors, alpha=0.8)
ax3.set_ylabel('Improvement Factor')
ax3.set_title('Improvement over Theoretical Parameters')
ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax3.set_ylim(0, 3)

for bar, imp in zip(bars3, improvements):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{imp:.1f}×', ha='center', va='bottom')

plt.suptitle('QAOA Parameter Optimization: Cross-Platform Validation', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('cross_platform_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'cross_platform_results.png'")
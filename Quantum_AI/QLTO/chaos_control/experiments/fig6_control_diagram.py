"""
Figure 6: Control System Block Diagram

Clean schematic of the quantum chaos control loop.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

from core import FIGURES_DIR


def main():
    print("Generating Control System Diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    
    # Block definitions: (x, y, width, height, label, color)
    blocks = [
        (0.5, 2, 2, 1.2, 'VQA\nOptimizer', '#AED6F1'),
        (3.5, 2, 2, 1.2, 'Trajectory\nBuffer', '#FCF3CF'),
        (6.5, 2, 2.2, 1.2, 'QFT Period\nDetector', '#ABEBC6'),
        (9.5, 2, 2, 1.2, 'Learning Rate\nController', '#F5B7B1'),
    ]
    
    for x, y, w, h, label, color in blocks:
        rect = Rectangle((x, y), w, h, fill=True, 
                         facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=11, 
               ha='center', va='center', fontweight='bold')
    
    # Forward arrows
    arrows = [
        (2.5, 2.6, 3.5, 2.6, 'θ(n)'),
        (5.5, 2.6, 6.5, 2.6, ''),
        (8.7, 2.6, 9.5, 2.6, 'Period k'),
    ]
    
    for x1, y1, x2, y2, label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            ax.text((x1 + x2)/2, y1 + 0.3, label, fontsize=10, ha='center')
    
    # Feedback arrow (curved)
    ax.annotate('', xy=(0.5, 2), xytext=(9.5, 2),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red',
                              connectionstyle='arc3,rad=-0.4'))
    ax.text(5, 0.7, 'Feedback: Adjust γ', fontsize=12, ha='center', 
           color='red', fontweight='bold')
    
    # Control law box
    control_text = "if period ≥ 4:\n    γ ← γ × 0.9\nelif period = 1:\n    γ ← γ × 1.05"
    ax.text(10.5, 4.2, control_text, fontsize=9, family='monospace',
           va='top', ha='center',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_title('Quantum-Assisted Chaos Control Loop', 
                 fontsize=14, fontweight='bold', y=0.95)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig6_control_diagram.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()

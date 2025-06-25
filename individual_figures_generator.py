import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class IndividualFigureGenerator:
    """
    Generate individual publication-ready figures from attention sink analysis results
    """
    
    def __init__(self, json_file_path: str, output_dir: str = "./individual_figures"):
        self.json_file = Path(json_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the analysis results
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Set publication style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight'
        })
        
    def figure1_positional_dominance(self):
        """
        Figure 1: Positional Dominance of P1 in GPT-2
        Bar chart showing attention received by each sink position (P1-P4)
        """
        # Extract position analysis data
        position_data = np.array(self.data['attention_analysis']['position_analysis'])
        avg_by_position = position_data.mean(axis=0)
        std_by_position = position_data.std(axis=0)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        positions = range(1, len(avg_by_position) + 1)
        bars = ax.bar(positions, avg_by_position, 
                     yerr=std_by_position, 
                     capsize=5, 
                     alpha=0.8,
                     color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_by_position)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_by_position[i] + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Sink Token Position')
        ax.set_ylabel('Average Attention Received')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'P{i}' for i in positions])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(avg_by_position) + max(std_by_position) + 0.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "figure1_positional_dominance.pdf"
        plt.savefig(output_path)
        print(f"Figure 1 saved: {output_path}")
        plt.show()
        
    def figure2_layerwise_attention(self):
        """
        Figure 2: Layer-wise P1 Attention Dynamics in GPT-2
        Line plot showing attention to sinks across layers
        """
        layer_averages = self.data['attention_analysis']['layer_averages']
        layers = range(1, len(layer_averages) + 1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(layers, layer_averages, 'o-', linewidth=2.5, markersize=7, 
               color='#d62728', markerfacecolor='white', markeredgecolor='#d62728', 
               markeredgewidth=2)
        
        # Highlight peak layer
        peak_layer_idx = np.argmax(layer_averages)
        peak_layer = peak_layer_idx + 1
        peak_value = layer_averages[peak_layer_idx]
        
        ax.plot(peak_layer, peak_value, 'o', markersize=10, 
               color='gold', markeredgecolor='#d62728', markeredgewidth=2,
               label=f'Peak: Layer {peak_layer}')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Attention to Sink Tokens')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "figure2_layerwise_p1_attention.pdf"
        plt.savefig(output_path)
        print(f"Figure 2 saved: {output_path}")
        plt.show()
        
    def figure3_headwise_attention(self):
        """
        Figure 3: Diversity of Head Contributions to P1 Sink Attention in GPT-2
        Heatmap showing attention patterns across heads and layers
        """
        head_data = np.array(self.data['attention_analysis']['head_patterns'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(head_data, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(head_data.shape[1]))
        ax.set_xticklabels([f'H{i+1}' for i in range(head_data.shape[1])])
        ax.set_yticks(range(head_data.shape[0]))
        ax.set_yticklabels([f'L{i+1}' for i in range(head_data.shape[0])])
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Layer')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention to Sink Tokens')
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        output_path = self.output_dir / "figure3_headwise_attention.pdf"
        plt.savefig(output_path)
        print(f"Figure 3 saved: {output_path}")
        plt.show()
        
    def figure4_representation_norms(self):
        """
        Figure 4: P1 Representation Norm Dynamics in GPT-2
        Line plot showing how representation norms evolve across layers
        """
        norm_data = np.array(self.data['representation_analysis']['norm_analysis'])
        layers = range(1, len(norm_data) + 1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot each sink position, with emphasis on P1
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        linewidths = [3, 2, 2, 2]  # P1 gets thicker line
        markers = ['o', 's', '^', 'D']
        
        for sink_idx in range(min(norm_data.shape[1], 4)):  # Limit to 4 sinks
            ax.plot(layers, norm_data[:, sink_idx], 
                   color=colors[sink_idx], 
                   linewidth=linewidths[sink_idx],
                   marker=markers[sink_idx],
                   markersize=6,
                   label=f'P{sink_idx+1}' + (' (Primary Sink)' if sink_idx == 0 else ''),
                   alpha=0.9 if sink_idx == 0 else 0.7)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('L2 Norm of Representation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "figure4_p1_norm_dynamics.pdf"
        plt.savefig(output_path)
        print(f"Figure 4 saved: {output_path}")
        plt.show()
        
    def generate_all_figures(self):
        """
        Generate all individual figures
        """
        print("Generating individual publication figures...")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
        
        self.figure1_positional_dominance()
        print()
        
        self.figure2_layerwise_attention()
        print()
        
        self.figure3_headwise_attention()
        print()
        
        self.figure4_representation_norms()
        print()
        
        print("=" * 50)
        print("All figures generated successfully!")
        print(f"Files saved in: {self.output_dir}")
        
        # Print summary statistics for reference
        self._print_summary_stats()
        
    def _print_summary_stats(self):
        """
        Print key statistics for reference in paper writing
        """
        print("\nKey Statistics for Paper:")
        print("-" * 30)
        
        # Position dominance
        position_data = np.array(self.data['attention_analysis']['position_analysis'])
        avg_by_position = position_data.mean(axis=0)
        
        print(f"P1 attention: {avg_by_position[0]:.3f}")
        print(f"P2 attention: {avg_by_position[1]:.3f}")
        print(f"P3 attention: {avg_by_position[2]:.3f}")
        print(f"P4 attention: {avg_by_position[3]:.3f}")
        
        # P1 dominance ratio
        p1_dominance = avg_by_position[0] / sum(avg_by_position[1:])
        print(f"P1 dominance ratio: {p1_dominance:.1f}x")
        
        # Layer dynamics
        layer_averages = self.data['attention_analysis']['layer_averages']
        peak_layer = np.argmax(layer_averages) + 1
        peak_value = max(layer_averages)
        
        print(f"Peak attention layer: {peak_layer}")
        print(f"Peak attention value: {peak_value:.3f}")
        print(f"Sequence length: {self.data['attention_analysis']['sequence_length']}")

def main():
    """
    Main function to generate individual figures
    Usage: Update the json_file_path to point to your saved results
    """
    
    # Update this path to point to your saved JSON results
    json_file_path = "./attention_sink_analysis/results_gpt2.json"
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"Error: JSON file not found at {json_file_path}")
        print("Please update the path to point to your saved analysis results.")
        print("The file should be generated by running the comprehensive analysis first.")
        return
    
    # Generate figures
    generator = IndividualFigureGenerator(json_file_path)
    generator.generate_all_figures()

if __name__ == "__main__":
    main()
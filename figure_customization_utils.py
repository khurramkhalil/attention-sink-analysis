import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class FigureStyler:
    """
    Utility class for consistent figure styling across all publication figures
    """
    
    @staticmethod
    def setup_publication_style():
        """Set up matplotlib for publication-quality figures"""
        plt.style.use('default')
        
        # Publication settings
        plt.rcParams.update({
            # Font settings
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            
            # Figure settings
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Line and marker settings
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 0.5,
            
            # Grid settings
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            
            # Spine settings
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Color cycle (professional colors)
            'axes.prop_cycle': plt.cycler('color', [
                '#d62728',  # Red
                '#1f77b4',  # Blue  
                '#ff7f0e',  # Orange
                '#2ca02c',  # Green
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
            ])
        })
    
    @staticmethod
    def clean_axes(ax):
        """Remove top and right spines for cleaner look"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)
        
    @staticmethod
    def add_significance_bars(ax, x_positions, heights, significance_levels, y_offset=0.02):
        """Add significance bars above bars in bar plots"""
        for i, (x, h, sig) in enumerate(zip(x_positions, heights, significance_levels)):
            if sig:
                ax.text(x, h + y_offset, sig, ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)

# Enhanced individual figure generator with better styling
class EnhancedFigureGenerator:
    """
    Enhanced version with additional customization options
    """
    
    def __init__(self, json_file_path: str, output_dir: str = "./publication_figures"):
        self.json_file = Path(json_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        import json
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Setup styling
        FigureStyler.setup_publication_style()
        
    def figure1_positional_dominance_enhanced(self, show_stats=True, color_scheme='journal'):
        """
        Enhanced Figure 1: P1 Positional Dominance with statistical annotations
        """
        position_data = np.array(self.data['attention_analysis']['position_analysis'])
        avg_by_position = position_data.mean(axis=0)
        std_by_position = position_data.std(axis=0)
        sem_by_position = std_by_position / np.sqrt(position_data.shape[0])
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Color schemes
        if color_scheme == 'journal':
            colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']  # Red, Blue, Orange, Green
        elif color_scheme == 'grayscale':
            colors = ['#2c2c2c', '#5a5a5a', '#888888', '#b6b6b6']
        else:
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        
        positions = range(1, len(avg_by_position) + 1)
        bars = ax.bar(positions, avg_by_position, 
                     yerr=sem_by_position,  # Use SEM for error bars
                     capsize=5, 
                     alpha=0.8,
                     color=colors[:len(positions)],
                     edgecolor='black',
                     linewidth=0.5)
        
        # Highlight P1 with special styling
        bars[0].set_alpha(1.0)
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
        
        # Add value labels
        for i, (bar, val, sem) in enumerate(zip(bars, avg_by_position, sem_by_position)):
            height = bar.get_height() + sem + 0.005
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        # Styling
        ax.set_xlabel('Initial Token Position', fontweight='bold')
        ax.set_ylabel('Average Attention Received', fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'P{i}' for i in positions])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(avg_by_position) + max(sem_by_position) + 0.05)
        
        FigureStyler.clean_axes(ax)
        
        # Add statistics text box if requested
        if show_stats:
            stats_text = f'P1 Dominance: {avg_by_position[0]/sum(avg_by_position[1:]):.1f}×\nN = {position_data.shape[0]} layers'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                   fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "figure1_positional_dominance.pdf"
        plt.savefig(output_path)
        print(f"Enhanced Figure 1 saved: {output_path}")
        plt.show()
    
    def figure2_layerwise_with_confidence(self, add_trend_line=False):
        """
        Figure 2: Layer-wise attention with confidence intervals
        """
        layer_averages = self.data['attention_analysis']['layer_averages']
        layers = np.array(range(1, len(layer_averages) + 1))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Main line plot
        line = ax.plot(layers, layer_averages, 'o-', linewidth=2.5, markersize=8, 
                      color='#d62728', markerfacecolor='white', 
                      markeredgecolor='#d62728', markeredgewidth=2, 
                      label='Sink Attention')[0]
        
        # Highlight peak
        peak_idx = np.argmax(layer_averages)
        peak_layer = layers[peak_idx]
        peak_value = layer_averages[peak_idx]
        
        ax.plot(peak_layer, peak_value, 'o', markersize=12, 
               color='gold', markeredgecolor='#d62728', markeredgewidth=2,
               zorder=5, label=f'Peak (Layer {peak_layer})')
        
        # Add trend line if requested
        if add_trend_line:
            # Fit polynomial trend
            z = np.polyfit(layers, layer_averages, 3)
            p = np.poly1d(z)
            ax.plot(layers, p(layers), '--', alpha=0.5, color='gray', 
                   label='Trend')
        
        # Styling
        ax.set_xlabel('Transformer Layer', fontweight='bold')
        ax.set_ylabel('Average Attention to Sink Tokens', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim(0.5, len(layer_averages) + 0.5)
        
        FigureStyler.clean_axes(ax)
        
        # Add annotations
        ax.annotate(f'Peak: {peak_value:.3f}', 
                   xy=(peak_layer, peak_value), 
                   xytext=(peak_layer + 1, peak_value + 0.05),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        output_path = self.output_dir / "figure2_layerwise_attention.pdf"
        plt.savefig(output_path)
        print(f"Enhanced Figure 2 saved: {output_path}")
        plt.show()
    
    def generate_all_enhanced(self):
        """Generate all figures with enhanced styling"""
        print("Generating enhanced publication figures...")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
        
        self.figure1_positional_dominance_enhanced(show_stats=True)
        print()
        
        self.figure2_layerwise_with_confidence(add_trend_line=False)
        print()
        
        print("Enhanced figures generated!")

# Quick usage example
def generate_custom_figures(json_path, style='enhanced'):
    """
    Convenience function to generate figures with different styles
    """
    if style == 'enhanced':
        generator = EnhancedFigureGenerator(json_path)
        generator.generate_all_enhanced()
    else:
        # Use basic generator from previous artifact
        from individual_figures_generator import IndividualFigureGenerator
        generator = IndividualFigureGenerator(json_path)
        generator.generate_all_figures()

if __name__ == "__main__":
    # Example usage
    json_file_path = "./attention_sink_analysis/results_gpt2.json"
    
    if Path(json_file_path).exists():
        generate_custom_figures(json_file_path, style='enhanced')
    else:
        print(f"Please update the path: {json_file_path}")
        print("Run your comprehensive analysis first to generate the JSON file.")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class AAAIFigureGenerator:
    """
    Generate individual, publication-ready figures for AAAI paper
    """
    
    def __init__(self, json_file_path: str, posthoc_file_path: str = None):
        """Load analysis results"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        # Load post-hoc results if available
        self.posthoc_data = None
        if posthoc_file_path and Path(posthoc_file_path).exists():
            with open(posthoc_file_path, 'r') as f:
                self.posthoc_data = json.load(f)
        
        # Extract plot data
        self.df = self._extract_plot_data()
        
        # Set publication style
        self._set_publication_style()
    
    def _extract_plot_data(self):
        """Extract data for plotting"""
        plot_data = []
        
        for family, models in self.data['organized_results'].items():
            for model_name, categories in models.items():
                for category, samples in categories.items():
                    for sample in samples:
                        if 'attention_analysis' in sample and 'position_analysis' in sample['attention_analysis']:
                            pos_data = np.array(sample['attention_analysis']['position_analysis'])
                            avg_by_position = pos_data.mean(axis=0)
                            
                            plot_data.append({
                                "Family": family,
                                "Model": model_name.split('/')[-1],
                                "Category": category,
                                "Position_1": avg_by_position[0] if len(avg_by_position) > 0 else np.nan,
                                "Position_2": avg_by_position[1] if len(avg_by_position) > 1 else np.nan,
                                "Position_3": avg_by_position[2] if len(avg_by_position) > 2 else np.nan,
                                "Position_4": avg_by_position[3] if len(avg_by_position) > 3 else np.nan,
                            })
        
        return pd.DataFrame(plot_data)
    
    def _set_publication_style(self):
        """Set publication-quality matplotlib style"""
        # Use IEEE style parameters
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman'],
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'patch.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        })
        
        # Set professional color palette
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Professional magenta
            'accent': '#F18F01',       # Professional orange
            'success': '#C73E1D',      # Professional red
            'family': ['#2E86AB', '#A23B72', '#F18F01'],  # For families
            'category': ['#C73E1D', '#A23B72', '#2E86AB', '#F18F01', '#6A994E']  # For categories
        }
    
    def generate_figure1_positional_dominance(self, save_path: str = "./figure1_positional_dominance.pdf"):
        """
        Figure 1: Overall Positional Dominance of P1
        Bar chart showing attention to P1 vs P2-P4
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Calculate means for each position
        position_cols = ["Position_1", "Position_2", "Position_3", "Position_4"]
        position_means = self.df[position_cols].mean()
        position_stds = self.df[position_cols].std()
        
        # Create bar chart
        x_pos = np.arange(1, 5)
        bars = ax.bar(x_pos, position_means, yerr=position_stds, 
                     capsize=4, alpha=0.8, 
                     color=[self.colors['primary'], self.colors['secondary'], 
                           self.colors['accent'], self.colors['success']],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, position_means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + position_stds.iloc[i] + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Attention Score', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['P1', 'P2', 'P3', 'P4'])
        ax.set_ylim(0, max(position_means + position_stds) * 1.15)
        
        # Add statistical annotation for P1 dominance
        if self.posthoc_data and 'position_dominance_tests' in self.posthoc_data.get('statistical_analysis', {}):
            # Add significance stars above P1
            ax.text(1, position_means.iloc[0] + position_stds.iloc[0] + 0.05, 
                   '***', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Figure 1 saved: {save_path}")
    
    def generate_figure2_family_effects(self, save_path: str = "./figure2_family_effects.pdf"):
        """
        Figure 2: P1 Attention Strength Across Model Families (with Post-Hoc Results)
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        
        # Calculate family statistics
        families = sorted(self.df['Family'].unique())
        family_means = []
        family_stds = []
        family_ns = []
        
        for family in families:
            family_data = self.df[self.df['Family'] == family]['Position_1']
            family_means.append(family_data.mean())
            family_stds.append(family_data.std())
            family_ns.append(len(family_data))
        
        # Create bar chart
        x_pos = np.arange(len(families))
        bars = ax.bar(x_pos, family_means, yerr=family_stds, 
                     capsize=5, alpha=0.8, 
                     color=self.colors['family'][:len(families)],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, mean_val, std_val, n) in enumerate(zip(bars, family_means, family_stds, family_ns)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Add sample size
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'n={n}', ha='center', va='center', fontsize=9, alpha=0.7)
        
        # Add statistical significance annotations
        if (self.posthoc_data and 'family_analysis' in self.posthoc_data and 
            'pairwise_comparisons' in self.posthoc_data['family_analysis']):
            
            pairwise = self.posthoc_data['family_analysis']['pairwise_comparisons']
            y_offset = max(family_means) + max(family_stds) + 0.03
            
            # Add significance lines and stars
            sig_pairs = [(k, v) for k, v in pairwise.items() if v.get('significant', False)]
            
            for i, (comparison, result) in enumerate(sig_pairs):
                f1, f2 = comparison.split('_vs_')
                if f1 in families and f2 in families:
                    x1, x2 = families.index(f1), families.index(f2)
                    
                    # Draw significance line
                    line_y = y_offset + i * 0.04
                    ax.plot([x1, x2], [line_y, line_y], 'k-', linewidth=1)
                    ax.plot([x1, x1], [line_y - 0.01, line_y + 0.01], 'k-', linewidth=1)
                    ax.plot([x2, x2], [line_y - 0.01, line_y + 0.01], 'k-', linewidth=1)
                    
                    # Add significance level
                    p_val = result.get('p_value', 1.0)
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    ax.text((x1 + x2) / 2, line_y + 0.01, sig_text, 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Model Family', fontsize=12, fontweight='bold')
        ax.set_ylabel('P1 Attention Strength', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f.upper() for f in families])
        ax.set_ylim(0, max(family_means) + max(family_stds) + 0.15)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Figure 2 saved: {save_path}")
    
    def generate_figure3_category_effects(self, save_path: str = "./figure3_category_effects.pdf"):
        """
        Figure 3: P1 Attention Strength Across Text Categories (with Post-Hoc Results)
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        
        # Calculate category statistics
        categories = sorted(self.df['Category'].unique())
        cat_means = []
        cat_stds = []
        cat_ns = []
        
        for category in categories:
            cat_data = self.df[self.df['Category'] == category]['Position_1']
            cat_means.append(cat_data.mean())
            cat_stds.append(cat_data.std())
            cat_ns.append(len(cat_data))
        
        # Create bar chart
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, cat_means, yerr=cat_stds, 
                     capsize=5, alpha=0.8, 
                     color=self.colors['category'][:len(categories)],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, mean_val, std_val, n) in enumerate(zip(bars, cat_means, cat_stds, cat_ns)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Add sample size
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'n={n}', ha='center', va='center', fontsize=8, alpha=0.7)
        
        # Add key statistical significance annotations (top 3 most significant)
        if (self.posthoc_data and 'category_analysis' in self.posthoc_data and 
            'pairwise_comparisons' in self.posthoc_data['category_analysis']):
            
            pairwise = self.posthoc_data['category_analysis']['pairwise_comparisons']
            
            # Find most significant comparisons
            sig_pairs = [(k, v) for k, v in pairwise.items() if v.get('significant', False)]
            sig_pairs.sort(key=lambda x: x[1].get('p_value', 1.0))  # Sort by p-value
            
            # Show only top 3 to avoid clutter
            y_offset = max(cat_means) + max(cat_stds) + 0.03
            
            for i, (comparison, result) in enumerate(sig_pairs[:3]):
                c1, c2 = comparison.split('_vs_')
                if c1 in categories and c2 in categories:
                    x1, x2 = categories.index(c1), categories.index(c2)
                    
                    # Draw significance line
                    line_y = y_offset + i * 0.05
                    ax.plot([x1, x2], [line_y, line_y], 'k-', linewidth=1)
                    ax.plot([x1, x1], [line_y - 0.01, line_y + 0.01], 'k-', linewidth=1)
                    ax.plot([x2, x2], [line_y - 0.01, line_y + 0.01], 'k-', linewidth=1)
                    
                    # Add significance level
                    p_val = result.get('p_value', 1.0)
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    ax.text((x1 + x2) / 2, line_y + 0.01, sig_text, 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Text Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('P1 Attention Strength', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.title() for c in categories], rotation=0)
        ax.set_ylim(0, max(cat_means) + max(cat_stds) + 0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Figure 3 saved: {save_path}")
    
    def generate_figure4_interaction_heatmap(self, save_path: str = "./figure4_family_category_interaction.pdf"):
        """
        Figure 4: Interaction of Model Family and Text Category on P1 Attention (Heatmap)
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        # Create interaction data
        heatmap_data = self.df.groupby(["Family", "Category"])["Position_1"].mean().unstack()
        
        # Create heatmap
        im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels([col.title() for col in heatmap_data.columns], rotation=0)
        ax.set_yticklabels([idx.upper() for idx in heatmap_data.index])
        
        # Add value annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < heatmap_data.values.mean() else 'black'
                    ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                           color=text_color, fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('P1 Attention Strength', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Formatting
        ax.set_xlabel('Text Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Family', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Figure 4 saved: {save_path}")
    
    def generate_supplementary_dominance_ratio(self, save_path: str = "./figureS1_dominance_ratio.pdf"):
        """
        Supplementary Figure: P1 Dominance Ratio by Family (for appendix/text reference)
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Calculate dominance ratios
        self.df['P1_Dominance_Ratio'] = self.df['Position_1'] / (
            self.df['Position_2'] + self.df['Position_3'] + self.df['Position_4'] + 1e-6
        )
        
        # Create boxplot
        families = sorted(self.df['Family'].unique())
        family_data = [self.df[self.df['Family'] == family]['P1_Dominance_Ratio'].values 
                      for family in families]
        
        bp = ax.boxplot(family_data, labels=[f.upper() for f in families], 
                       patch_artist=True, notch=True, showmeans=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], self.colors['family']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # Style other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1)
        plt.setp(bp['means'], marker='D', markerfacecolor='red', markeredgecolor='black', markersize=4)
        
        # Formatting
        ax.set_xlabel('Model Family', fontsize=12, fontweight='bold')
        ax.set_ylabel('P1 Dominance Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Supplementary Figure saved: {save_path}")
    
    def generate_all_figures(self, output_dir: str = "./aaai_figures/"):
        """Generate all publication-ready figures"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🎨 Generating AAAI publication-ready figures...")
        print("=" * 50)
        
        # Generate main figures
        self.generate_figure1_positional_dominance(
            save_path=str(output_path / "figure1_positional_dominance.pdf")
        )
        
        self.generate_figure2_family_effects(
            save_path=str(output_path / "figure2_family_effects.pdf")
        )
        
        self.generate_figure3_category_effects(
            save_path=str(output_path / "figure3_category_effects.pdf")
        )
        
        self.generate_figure4_interaction_heatmap(
            save_path=str(output_path / "figure4_family_category_interaction.pdf")
        )
        
        # Generate supplementary figure
        self.generate_supplementary_dominance_ratio(
            save_path=str(output_path / "figureS1_dominance_ratio.pdf")
        )
        
        print("=" * 50)
        print(f"✅ All figures generated and saved to: {output_path}")
        print("\n📋 Figure Summary:")
        print("• Figure 1: Overall P1 dominance across all positions")
        print("• Figure 2: Family effects with statistical significance")
        print("• Figure 3: Category effects with key comparisons") 
        print("• Figure 4: Family-category interaction heatmap")
        print("• Figure S1: P1 dominance ratio distributions (supplementary)")
        
        return output_path

def main():
    """Generate all AAAI figures"""
    
    # Paths to your data files
    main_results_path = "./expanded_sink_analysis/comprehensive_analysis_results.json"
    posthoc_results_path = "./expanded_sink_analysis/posthoc_analysis_results.json"
    
    try:
        # Initialize figure generator
        generator = AAAIFigureGenerator(
            json_file_path=main_results_path,
            posthoc_file_path=posthoc_results_path
        )
        
        # Generate all figures
        output_dir = generator.generate_all_figures()
        
        print(f"\n🎉 Publication-ready figures generated successfully!")
        print(f"📁 Location: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found")
        print(f"   Please ensure these files exist:")
        print(f"   - {main_results_path}")
        print(f"   - {posthoc_results_path}")
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")

if __name__ == "__main__":
    main()
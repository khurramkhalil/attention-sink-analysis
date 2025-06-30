"""
Publication-Ready Figures Generator for P1 Causal Analysis
=========================================================

This script generates high-quality, publication-ready figures from the 
comprehensive P1 analysis results.

Usage:
    python publication_figures_generator.py --input results.json --output figures/

Author: Research Team
Date: June 2025
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
from scipy import stats

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Set color palette
colors = {
    'gpt2': '#2E86AB',      # Blue
    'llama': '#A23B72',     # Purple
    'mistral': '#F18F01',   # Orange
    'deepseek': '#C73E1D',  # Red
    'other': '#808080'      # Gray
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublicationFigureGenerator:
    """Generates publication-ready figures from P1 analysis results"""
    
    def __init__(self, results_file: str, output_dir: str = "figures"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded results from {self.results_file}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_all_figures(self):
        """Generate all publication figures"""
        logger.info("🎨 Generating publication-ready figures...")
        
        # Figure 1: Intervention Impact Overview
        self.create_figure1_intervention_impact()
        
        # Figure 2: Layer-wise Ablation Effects  
        self.create_figure2_layer_ablation()
        
        # Figure 3: Downstream Task Performance
        self.create_figure3_downstream_tasks()
        
        # Figure 4: P1 Information Content (Probing)
        self.create_figure4_probing_analysis()
        
        # Figure 5: Intervention Type Comparison
        self.create_figure5_intervention_comparison()
        
        # Figure 6: Baseline Performance Analysis
        self.create_figure6_baseline_analysis()
        
        logger.info(f"✅ All figures saved to {self.output_dir}")
    
    def create_figure1_intervention_impact(self):
        """Figure 1: Comprehensive intervention impact overview"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract intervention data
        intervention_data = self.results.get("intervention_results", {})
        if not intervention_data:
            logger.warning("No intervention data found")
            return
        
        # Panel A: All interventions ranked by impact
        interventions = []
        impacts = []
        types = []
        
        for name, data in intervention_data.items():
            impact = data.get("mean_degradation", 0)
            interventions.append(name.replace('_', ' ').title())
            impacts.append(impact)
            
            # Categorize intervention type
            if 'ablation' in name and 'mean' not in name:
                types.append('Ablation')
            elif 'mean_ablation' in name:
                types.append('Mean Ablation')
            elif 'noise' in name:
                types.append('Noise Injection')
            elif 'random' in name:
                types.append('Random Replacement')
            else:
                types.append('Other')
        
        # Sort by impact
        sorted_data = sorted(zip(interventions, impacts, types), key=lambda x: x[1], reverse=True)
        interventions, impacts, types = zip(*sorted_data)
        
        # Create color map
        type_colors = {'Ablation': '#e74c3c', 'Mean Ablation': '#f39c12', 
                      'Noise Injection': '#3498db', 'Random Replacement': '#9b59b6', 'Other': '#95a5a6'}
        bar_colors = [type_colors.get(t, '#95a5a6') for t in types]
        
        bars = ax1.barh(range(len(interventions)), impacts, color=bar_colors, alpha=0.8)
        ax1.set_yticks(range(len(interventions)))
        ax1.set_yticklabels(interventions, fontsize=10)
        ax1.set_xlabel('Performance Degradation (%)', fontweight='bold')
        ax1.set_title('(A) Intervention Impact Ranking', fontweight='bold', fontsize=14)
        ax1.set_xscale('symlog', linthresh=1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, impact) in enumerate(zip(bars, impacts)):
            if impact > 0:
                ax1.text(impact + max(impacts) * 0.01, i, f'{impact:.1f}%', 
                        va='center', fontsize=9, fontweight='bold')
        
        # Panel B: Critical layers identification (ablation only)
        ablation_layers = []
        ablation_impacts = []
        
        for name, data in intervention_data.items():
            if 'ablation_layer_' in name and 'mean' not in name:
                try:
                    layer = int(name.split('layer_')[1].split('_')[0])
                    impact = data.get("mean_degradation", 0)
                    ablation_layers.append(layer)
                    ablation_impacts.append(impact)
                except:
                    continue
        
        if ablation_layers:
            # Sort by layer
            sorted_ablation = sorted(zip(ablation_layers, ablation_impacts))
            ablation_layers, ablation_impacts = zip(*sorted_ablation)
            
            ax2.plot(ablation_layers, ablation_impacts, 'o-', linewidth=3, markersize=8, 
                    color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
            ax2.set_xlabel('Layer Index', fontweight='bold')
            ax2.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax2.set_title('(B) Layer-wise Ablation Impact', fontweight='bold', fontsize=14)
            ax2.set_yscale('symlog', linthresh=1)
            ax2.grid(True, alpha=0.3)
            
            # Highlight critical layers (top 2)
            critical_indices = np.argsort(ablation_impacts)[-2:]
            for idx in critical_indices:
                ax2.scatter(ablation_layers[idx], ablation_impacts[idx], 
                          s=200, c='red', marker='*', zorder=5)
                ax2.annotate(f'L{ablation_layers[idx]}', 
                           (ablation_layers[idx], ablation_impacts[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold', fontsize=11, color='red')
        
        # Panel C: Intervention type comparison
        type_impacts = {}
        for t in set(types):
            impacts_for_type = [impacts[i] for i, type_name in enumerate(types) if type_name == t]
            if impacts_for_type:
                type_impacts[t] = impacts_for_type
        
        if type_impacts:
            box_data = []
            box_labels = []
            for t, impacts_list in type_impacts.items():
                box_data.append(impacts_list)
                box_labels.append(t)
            
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True, 
                           notch=True, showfliers=True)
            
            for patch, label in zip(bp['boxes'], box_labels):
                patch.set_facecolor(type_colors.get(label, '#95a5a6'))
                patch.set_alpha(0.7)
            
            ax3.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax3.set_title('(C) Intervention Type Distribution', fontweight='bold', fontsize=14)
            ax3.set_yscale('symlog', linthresh=1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel D: Statistical significance overview
        significant_interventions = []
        effect_sizes = []
        
        for name, data in intervention_data.items():
            impact = data.get("mean_degradation", 0)
            std = data.get("std_degradation", 0)
            samples = data.get("samples", [])
            
            if len(samples) > 1:
                # Perform one-sample t-test
                t_stat, p_value = stats.ttest_1samp(samples, 0)
                is_significant = p_value < 0.05
                
                if is_significant and abs(impact) > 1:  # Only show meaningful effects
                    significant_interventions.append(name.replace('_', ' ').title())
                    effect_sizes.append(abs(impact))
        
        if significant_interventions:
            bars = ax4.barh(range(len(significant_interventions)), effect_sizes, 
                          color='#27ae60', alpha=0.8)
            ax4.set_yticks(range(len(significant_interventions)))
            ax4.set_yticklabels(significant_interventions, fontsize=10)
            ax4.set_xlabel('|Effect Size| (%)', fontweight='bold')
            ax4.set_title('(D) Statistically Significant Effects', fontweight='bold', fontsize=14)
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add significance indicators
            for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
                ax4.text(effect * 1.1, i, '***' if effect > 100 else '**' if effect > 10 else '*', 
                        va='center', fontsize=12, fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure1_intervention_impact_overview.pdf")
        plt.close()
        logger.info("Created Figure 1: Intervention Impact Overview")
    
    def create_figure2_layer_ablation(self):
        """Figure 2: Detailed layer-wise ablation analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        intervention_data = self.results.get("intervention_results", {})
        
        # Extract layer data for different ablation types
        ablation_layers = []
        ablation_impacts = []
        mean_ablation_layers = []
        mean_ablation_impacts = []
        
        for name, data in intervention_data.items():
            impact = data.get("mean_degradation", 0)
            
            if 'ablation_layer_' in name and 'mean' not in name:
                try:
                    layer = int(name.split('layer_')[1].split('_')[0])
                    ablation_layers.append(layer)
                    ablation_impacts.append(impact)
                except:
                    continue
            elif 'mean_ablation_layer_' in name:
                try:
                    layer = int(name.split('layer_')[1].split('_')[0])
                    mean_ablation_layers.append(layer)
                    mean_ablation_impacts.append(impact)
                except:
                    continue
        
        # Panel A: Comparison of ablation types
        if ablation_layers and mean_ablation_layers:
            # Sort by layer
            ablation_sorted = sorted(zip(ablation_layers, ablation_impacts))
            mean_ablation_sorted = sorted(zip(mean_ablation_layers, mean_ablation_impacts))
            
            ablation_layers, ablation_impacts = zip(*ablation_sorted)
            mean_ablation_layers, mean_ablation_impacts = zip(*mean_ablation_sorted)
            
            ax1.plot(ablation_layers, ablation_impacts, 'o-', linewidth=3, markersize=8,
                    color='#e74c3c', label='Zero Ablation', markerfacecolor='white', 
                    markeredgewidth=2)
            ax1.plot(mean_ablation_layers, mean_ablation_impacts, 's-', linewidth=3, markersize=8,
                    color='#f39c12', label='Mean Ablation', markerfacecolor='white', 
                    markeredgewidth=2)
            
            ax1.set_xlabel('Layer Index', fontweight='bold')
            ax1.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax1.set_title('Layer-wise Ablation Comparison', fontweight='bold', fontsize=16)
            ax1.set_yscale('symlog', linthresh=1)
            ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Add critical layer annotations
            max_impact_idx = np.argmax(ablation_impacts)
            ax1.annotate(f'Critical Layer {ablation_layers[max_impact_idx]}\n{ablation_impacts[max_impact_idx]:.0f}%',
                        xy=(ablation_layers[max_impact_idx], ablation_impacts[max_impact_idx]),
                        xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=11, fontweight='bold')
        
        # Panel B: Category-specific impacts for critical layer
        critical_layer_data = None
        max_impact = 0
        
        for name, data in intervention_data.items():
            if 'ablation_layer_' in name and 'mean' not in name:
                impact = data.get("mean_degradation", 0)
                if impact > max_impact:
                    max_impact = impact
                    critical_layer_data = data
        
        if critical_layer_data and "degradation_by_category" in critical_layer_data:
            categories = list(critical_layer_data["degradation_by_category"].keys())
            degradations = list(critical_layer_data["degradation_by_category"].values())
            
            # Create category colors
            category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            bars = ax2.bar(categories, degradations, color=category_colors, alpha=0.8, 
                          edgecolor='black', linewidth=1)
            ax2.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax2.set_title('Category-Specific Impact\n(Critical Layer)', fontweight='bold', fontsize=16)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, deg in zip(bars, degradations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(degradations) * 0.02,
                        f'{deg:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure2_layer_ablation_analysis.pdf")
        plt.close()
        logger.info("Created Figure 2: Layer-wise Ablation Analysis")
    
    def create_figure3_downstream_tasks(self):
        """Figure 3: Downstream task performance analysis"""
        downstream_data = self.results.get("downstream_results", {})
        
        if not downstream_data:
            logger.warning("No downstream task data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        tasks = list(downstream_data.keys())
        baseline_accs = []
        intervention_accs = []
        degradations = []
        
        for task in tasks:
            task_data = downstream_data[task]
            baseline_accs.append(task_data.get("baseline_accuracy", 0))
            intervention_accs.append(task_data.get("intervention_accuracy", 0))
            degradations.append(task_data.get("degradation", 0))
        
        # Panel A: Baseline vs Intervention Performance
        x = np.arange(len(tasks))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', 
                       color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, intervention_accs, width, label='P1 Ablation', 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Downstream Tasks', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Downstream Task Performance\nBaseline vs P1 Ablation', fontweight='bold', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.replace('_', ' ').title() for t in tasks])
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel B: Performance Degradation
        bars = ax2.bar(range(len(tasks)), degradations, color='#ff6b6b', alpha=0.8, 
                      edgecolor='black', linewidth=2)
        ax2.set_xlabel('Downstream Tasks', fontweight='bold')
        ax2.set_ylabel('Performance Degradation', fontweight='bold')
        ax2.set_title('P1 Ablation Impact\non Downstream Performance', fontweight='bold', fontsize=16)
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels([t.replace('_', ' ').title() for t in tasks])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and significance indicators
        for i, (bar, deg) in enumerate(zip(bars, degradations)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(degradations) * 0.02,
                    f'{deg:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Add severity indicator
            if deg > 0.5:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2, '⚠\nSEVERE', 
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            elif deg > 0.2:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2, '⚠', 
                        ha='center', va='center', fontweight='bold', color='white', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure3_downstream_tasks.pdf")
        plt.close()
        logger.info("Created Figure 3: Downstream Task Analysis")
    
    def create_figure4_probing_analysis(self):
        """Figure 4: P1 Information Content Analysis"""
        probing_data = self.results.get("probing_results", {}).get("category_classification", {})
        
        if not probing_data:
            logger.warning("No probing data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract layer-wise accuracies
        layers = []
        accuracies = []
        stds = []
        
        for layer_key, layer_data in probing_data.items():
            if layer_key.startswith("layer_") and isinstance(layer_data, dict):
                try:
                    layer_num = int(layer_key.split('_')[1])
                    acc = layer_data.get("accuracy_mean", 0)
                    std = layer_data.get("accuracy_std", 0)
                    
                    layers.append(layer_num)
                    accuracies.append(acc)
                    stds.append(std)
                except:
                    continue
        
        if layers:
            # Sort by layer
            sorted_data = sorted(zip(layers, accuracies, stds))
            layers, accuracies, stds = zip(*sorted_data)
            
            # Panel A: Layer-wise classification accuracy
            ax1.errorbar(layers, accuracies, yerr=stds, marker='o', linewidth=3, 
                        markersize=8, capsize=5, capthick=2, color='#3498db',
                        markerfacecolor='white', markeredgewidth=2)
            
            # Add chance level
            num_categories = 5  # From the data
            chance_level = 1.0 / num_categories
            ax1.axhline(y=chance_level, color='red', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Chance Level ({chance_level:.2f})')
            
            ax1.set_xlabel('Layer Index', fontweight='bold')
            ax1.set_ylabel('Classification Accuracy', fontweight='bold')
            ax1.set_title('P1 Information Content Across Layers', fontweight='bold', fontsize=16)
            ax1.set_ylim(0, 1)
            ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Highlight best performing layer
            best_idx = np.argmax(accuracies)
            best_layer = layers[best_idx]
            best_acc = accuracies[best_idx]
            
            ax1.scatter(best_layer, best_acc, s=200, c='gold', marker='*', 
                       edgecolors='red', linewidth=2, zorder=5)
            ax1.annotate(f'Best: Layer {best_layer}\nAcc: {best_acc:.3f}',
                        xy=(best_layer, best_acc), xytext=(20, 20),
                        textcoords='offset points', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Panel B: Information content evolution
            # Calculate improvement over chance
            improvements = np.array(accuracies) - chance_level
            
            ax2.fill_between(layers, chance_level, accuracies, alpha=0.3, color='#3498db',
                           label='Above Chance Performance')
            ax2.plot(layers, accuracies, 'o-', linewidth=3, markersize=8, color='#2c3e50',
                    markerfacecolor='white', markeredgewidth=2)
            ax2.axhline(y=chance_level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            ax2.set_xlabel('Layer Index', fontweight='bold')
            ax2.set_ylabel('Classification Accuracy', fontweight='bold')
            ax2.set_title('P1 Information Evolution\n(Above-Chance Performance)', fontweight='bold', fontsize=16)
            ax2.set_ylim(chance_level - 0.05, max(accuracies) + 0.05)
            ax2.grid(True, alpha=0.3)
            
            # Add performance zones
            ax2.axhspan(chance_level, chance_level + 0.2, alpha=0.1, color='yellow', label='Weak Signal')
            ax2.axhspan(chance_level + 0.2, chance_level + 0.4, alpha=0.1, color='orange', label='Moderate Signal')
            ax2.axhspan(chance_level + 0.4, 1.0, alpha=0.1, color='green', label='Strong Signal')
            
            ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure4_probing_analysis.pdf")
        plt.close()
        logger.info("Created Figure 4: P1 Information Content Analysis")
    
    def create_figure5_intervention_comparison(self):
        """Figure 5: Comprehensive intervention type comparison"""
        intervention_data = self.results.get("intervention_results", {})
        
        if not intervention_data:
            logger.warning("No intervention data found")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Organize data by intervention type
        ablation_data = {}
        mean_ablation_data = {}
        noise_data = {}
        random_data = {}
        
        for name, data in intervention_data.items():
            impact = data.get("mean_degradation", 0)
            
            if 'ablation_layer_' in name and 'mean' not in name:
                layer = int(name.split('layer_')[1].split('_')[0])
                ablation_data[layer] = impact
            elif 'mean_ablation_layer_' in name:
                layer = int(name.split('layer_')[1].split('_')[0])
                mean_ablation_data[layer] = impact
            elif 'noise_injection' in name:
                parts = name.split('_')
                layer = int(parts[2])
                std = float(parts[1].replace('std', ''))
                if std not in noise_data:
                    noise_data[std] = {}
                noise_data[std][layer] = impact
            elif 'random_replacement' in name:
                layer = int(name.split('layer_')[1])
                random_data[layer] = impact
        
        # Panel A: Ablation comparison
        if ablation_data and mean_ablation_data:
            layers1 = sorted(ablation_data.keys())
            impacts1 = [ablation_data[l] for l in layers1]
            layers2 = sorted(mean_ablation_data.keys())
            impacts2 = [mean_ablation_data[l] for l in layers2]
            
            ax1.semilogy(layers1, impacts1, 'o-', linewidth=3, markersize=8,
                        color='#e74c3c', label='Zero Ablation', markerfacecolor='white')
            ax1.semilogy(layers2, impacts2, 's-', linewidth=3, markersize=8,
                        color='#f39c12', label='Mean Ablation', markerfacecolor='white')
            
            ax1.set_xlabel('Layer Index', fontweight='bold')
            ax1.set_ylabel('Performance Degradation (%) [Log Scale]', fontweight='bold')
            ax1.set_title('(A) Ablation Type Comparison', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Panel B: Noise injection analysis
        if noise_data:
            for std_level, layer_impacts in noise_data.items():
                layers = sorted(layer_impacts.keys())
                impacts = [layer_impacts[l] for l in layers]
                ax2.plot(layers, impacts, 'o-', linewidth=2, markersize=6, 
                        label=f'σ = {std_level}', alpha=0.8)
            
            ax2.set_xlabel('Layer Index', fontweight='bold')
            ax2.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax2.set_title('(B) Noise Injection Effects', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel C: Random replacement
        if random_data:
            layers = sorted(random_data.keys())
            impacts = [random_data[l] for l in layers]
            
            bars = ax3.bar(layers, impacts, color='#9b59b6', alpha=0.8, 
                          edgecolor='black', linewidth=1)
            ax3.set_xlabel('Layer Index', fontweight='bold')
            ax3.set_ylabel('Performance Degradation (%)', fontweight='bold')
            ax3.set_title('(C) Random Replacement Impact', fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, impact in zip(bars, impacts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{impact:.0f}%', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
        
        # Panel D: Intervention effectiveness summary
        all_interventions = []
        all_impacts = []
        all_types = []
        
        for name, data in intervention_data.items():
            impact = abs(data.get("mean_degradation", 0))
            all_interventions.append(name)
            all_impacts.append(impact)
            
            if 'ablation' in name and 'mean' not in name:
                all_types.append('Ablation')
            elif 'mean_ablation' in name:
                all_types.append('Mean Ablation')
            elif 'noise' in name:
                all_types.append('Noise Injection')
            elif 'random' in name:
                all_types.append('Random Replacement')
            else:
                all_types.append('Other')
        
        # Create violin plot
        df = pd.DataFrame({
            'Type': all_types,
            'Impact': all_impacts
        })
        
        if not df.empty:
            sns.violinplot(data=df, x='Type', y='Impact', ax=ax4, inner='box')
            ax4.set_xlabel('Intervention Type', fontweight='bold')
            ax4.set_ylabel('|Performance Impact| (%)', fontweight='bold')
            ax4.set_title('(D) Intervention Type Effectiveness', fontweight='bold')
            ax4.set_yscale('log')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure5_intervention_comparison.pdf")
        plt.close()
        logger.info("Created Figure 5: Intervention Type Comparison")
    
    def create_figure6_baseline_analysis(self):
        """Figure 6: Baseline performance and P1 attention analysis"""
        baseline_data = self.results.get("baseline_results", {})
        
        if not baseline_data:
            logger.warning("No baseline data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel A: Baseline perplexity by category
        perplexity_data = baseline_data.get("perplexity_by_category", {})
        
        if perplexity_data:
            categories = list(perplexity_data.keys())
            means = [perplexity_data[cat]["mean"] for cat in categories]
            stds = [perplexity_data[cat]["std"] for cat in categories]
            
            # Create category colors
            category_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = ax1.bar(categories, means, yerr=stds, capsize=5, 
                          color=category_colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax1.set_ylabel('Perplexity', fontweight='bold')
            ax1.set_title('Baseline Model Performance\nby Text Category', fontweight='bold', fontsize=16)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(means) * 0.02,
                        f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            # Add difficulty ranking
            difficulty_order = sorted(zip(categories, means), key=lambda x: x[1])
            difficulty_text = "Difficulty Ranking:\n" + " < ".join([cat.title() for cat, _ in difficulty_order])
            ax1.text(0.02, 0.98, difficulty_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
        
        # Panel B: Model statistics summary
        model_info = baseline_data.get("model_info", {})
        statistical_summary = self.results.get("statistical_analysis", {}).get("intervention_summary", {})
        
        if model_info or statistical_summary:
            # Create summary statistics
            summary_data = []
            
            # Model information
            if model_info:
                summary_data.extend([
                    f"Model Layers: {model_info.get('num_layers', 'N/A')}",
                    f"Vocabulary Size: {model_info.get('vocab_size', 'N/A'):,}",
                    f"Model Size: {model_info.get('model_size', 'N/A')}"
                ])
            
            # Add separator
            summary_data.append("")
            summary_data.append("Key Intervention Results:")
            
            # Statistical summary
            if statistical_summary:
                # Find most impactful interventions
                sorted_interventions = sorted(statistical_summary.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)[:5]
                
                for name, impact in sorted_interventions:
                    clean_name = name.replace('_', ' ').title()
                    summary_data.append(f"• {clean_name}: {impact:.1f}%")
            
            # Display as text
            ax2.text(0.05, 0.95, '\n'.join(summary_data), transform=ax2.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Model & Analysis Summary', fontweight='bold', fontsize=16)
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure6_baseline_analysis.pdf")
        plt.close()
        logger.info("Created Figure 6: Baseline Analysis")
    
    def create_summary_table(self):
        """Create a summary table of key results"""
        logger.info("Creating summary table...")
        
        # Collect key statistics
        intervention_data = self.results.get("intervention_results", {})
        downstream_data = self.results.get("downstream_results", {})
        baseline_data = self.results.get("baseline_results", {})
        
        summary_stats = {
            "Metric": [],
            "Value": [],
            "Description": []
        }
        
        # Model info
        model_config = self.results.get("model_config", {})
        if model_config:
            summary_stats["Metric"].append("Model")
            summary_stats["Value"].append(f"{model_config.get('name', 'Unknown')} ({model_config.get('size', 'Unknown')})")
            summary_stats["Description"].append("Model architecture and size")
        
        # Most critical intervention
        if intervention_data:
            max_impact = 0
            critical_intervention = ""
            for name, data in intervention_data.items():
                impact = abs(data.get("mean_degradation", 0))
                if impact > max_impact:
                    max_impact = impact
                    critical_intervention = name
            
            summary_stats["Metric"].append("Most Critical Intervention")
            summary_stats["Value"].append(f"{critical_intervention.replace('_', ' ').title()}")
            summary_stats["Description"].append(f"Maximum degradation: {max_impact:.1f}%")
        
        # Downstream task impacts
        if downstream_data:
            for task, task_data in downstream_data.items():
                degradation = task_data.get("degradation", 0)
                summary_stats["Metric"].append(f"{task.replace('_', ' ').title()} Impact")
                summary_stats["Value"].append(f"{degradation:.3f}")
                summary_stats["Description"].append("Performance degradation from P1 ablation")
        
        # Baseline performance
        if baseline_data and "perplexity_by_category" in baseline_data:
            avg_perplexity = np.mean([data["mean"] for data in baseline_data["perplexity_by_category"].values()])
            summary_stats["Metric"].append("Average Perplexity")
            summary_stats["Value"].append(f"{avg_perplexity:.2f}")
            summary_stats["Description"].append("Baseline model performance")
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_stats)
        df.to_csv(self.output_dir / "summary_table.csv", index=False)
        
        # Create a formatted table figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='left', loc='center', colWidths=[0.3, 0.2, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('P1 Causal Analysis - Summary Results', fontweight='bold', fontsize=16, pad=20)
        plt.savefig(self.output_dir / "summary_table.pdf")
        plt.close()
        
        logger.info("Created summary table")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures from P1 analysis results")
    parser.add_argument("--input", "-i", required=True, help="Input JSON results file")
    parser.add_argument("--output", "-o", default="figures", help="Output directory for figures")
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="pdf", help="Output format")
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    try:
        # Create figure generator
        generator = PublicationFigureGenerator(args.input, args.output)
        
        # Generate all figures
        generator.generate_all_figures()
        
        # Create summary table
        generator.create_summary_table()
        
        logger.info("🎉 All publication figures generated successfully!")
        logger.info(f"📁 Output directory: {generator.output_dir}")
        
        # List generated files
        pdf_files = list(generator.output_dir.glob("*.pdf"))
        logger.info(f"📊 Generated {len(pdf_files)} figures:")
        for pdf_file in sorted(pdf_files):
            logger.info(f"   - {pdf_file.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating figures: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
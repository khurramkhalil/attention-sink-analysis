"""
P1 Causal Analysis - Runner & Visualization (Script 2 of 2)
=========================================================

This script orchestrates the comprehensive analysis across multiple models,
performs cross-model comparisons, and generates publication-ready materials.

Companion script: p1_analysis_core.py

Usage:
    python p1_analysis_runner.py --config [quick|standard|comprehensive]

Author: Research Team
Date: June 2025
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
from scipy import stats as scipy_stats

# Import core framework from the companion script
try:
    from p1_analysis_core import (
        P1AnalysisConfig, ModelManager, DatasetManager, SingleModelAnalyzer, save_model_results
    )
except ImportError as e:
    print(f"❌ Error importing core framework: {e}")
    print("📁 Make sure p1_analysis_core.py is in the same directory or accessible in your Python path.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Configure logging for the runner script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossModelAnalyzer:
    """Performs cross-model comparative analysis on aggregated results."""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config

    def analyze(self, all_model_results: Dict[str, Dict]) -> Dict:
        """Run all cross-model analysis steps."""
        logger.info("🔍 Performing cross-model analysis...")
        cross_analysis_results = {
            "family_effects": self._analyze_family_effects(all_model_results),
            "category_effects": self._analyze_category_effects(all_model_results),
            "universal_patterns": self._find_universal_patterns(all_model_results)
        }
        return cross_analysis_results
    
    def _get_all_ppl_degradation(self, all_model_results: Dict) -> pd.DataFrame:
        """Helper to extract all perplexity degradation data into a DataFrame."""
        records = []
        for model_name, model_data in all_model_results.items():
            model_config = model_data['model_config']
            family = model_config.get('family', 'unknown')
            
            for intervention_key, intervention_data in model_data.get('intervention_results', {}).items():
                for layer, layer_data in intervention_data.get('results_by_layer', {}).items():
                    records.append({
                        "model": model_name,
                        "family": family,
                        "intervention": intervention_key,
                        "layer": layer,
                        "degradation": layer_data.get('performance_degradation', 0)
                    })
        return pd.DataFrame(records)

    def _analyze_family_effects(self, all_model_results: Dict) -> Dict:
        """Analyze differences between model families."""
        df = self._get_all_ppl_degradation(all_model_results)
        if df.empty: return {}

        # Focus on a consistent, strong intervention for comparison, e.g., early layer ablation
        ablation_df = df[df['intervention'].str.contains('ablation')]
        if ablation_df.empty: return {}
        
        # ANOVA on degradation by family
        family_groups = [group["degradation"].values for name, group in ablation_df.groupby("family")]
        
        f_stat, p_val = -1, 1
        if len(family_groups) > 1 and all(len(g) > 0 for g in family_groups):
            f_stat, p_val = scipy_stats.f_oneway(*family_groups)
            
        return {
            "anova_on_ablation_degradation": {
                "f_statistic": f_stat,
                "p_value": p_val,
                "significant": p_val < self.config.statistical_significance_level
            },
            "mean_degradation_by_family": df.groupby("family")["degradation"].mean().to_dict()
        }

    def _analyze_category_effects(self, all_model_results: Dict) -> Dict:
        # Note: Category effects were primarily analyzed within the single-model context.
        # This cross-model analysis would require re-running interventions per-category.
        # For now, we summarize based on what we have.
        return {"summary": "Category effects were analyzed per-model. See individual model results."}

    def _find_universal_patterns(self, all_model_results: Dict) -> Dict:
        """Identify intervention effects that are consistent across models."""
        # (This is a simplified version of the logic from the PoC)
        if len(all_model_results) < 2: return {}
        
        # Check if early layer ablation is consistently the most damaging
        most_damaging_layers = {} # Use a dict to keep track of model_name
        for model_name, model_data in all_model_results.items():
            interventions = model_data.get('intervention_results', {})
            degradation_by_layer = {}
            for key, data in interventions.items():
                if 'ablation' in key:
                    for layer, layer_data in data.get('results_by_layer', {}).items():
                        degradation_by_layer[layer] = layer_data.get('performance_degradation', 0)
            if degradation_by_layer:
                most_damaging_layers[model_name] = max(degradation_by_layer, key=degradation_by_layer.get)
        
        if not most_damaging_layers:
            return {"early_layer_ablation_is_critical": False, "mean_peak_damage_layer": -1}

        # Now safely check the criticality for each model
        criticality_checks = []
        for model_name, peak_damage_layer in most_damaging_layers.items():
            family = all_model_results[model_name]['model_config']['family']
            target_layers = self.config.target_layers.get(family, [])
            # Check if layer is in the first 25% of the model depth
            if len(target_layers) > 1:
                # Assuming target_layers[1] corresponds to the 25% mark
                criticality_checks.append(peak_damage_layer <= target_layers[1])
            else:
                criticality_checks.append(False) # Cannot determine

        early_layer_criticality = all(criticality_checks) if criticality_checks else False
                                        
        return {
            "early_layer_ablation_is_critical": early_layer_criticality,
            "mean_peak_damage_layer": np.mean(list(most_damaging_layers.values()))
        }


        # early_layer_criticality = all(layer <= self.config.target_layers.get(model_data['model_config']['family'], [0])[1] # check if peak damage is within first 25% of layers
        #                                 for layer, model_data in zip(most_damaging_layers, all_model_results.values()))
                                        
        # return {
        #     "early_layer_ablation_is_critical": early_layer_criticality,
        #     "mean_peak_damage_layer": np.mean(most_damaging_layers) if most_damaging_layers else -1
        # }


class PublicationVisualizer:
    """Creates publication-ready figures and reports"""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")

    def create_all_figures(self, all_model_results: Dict, cross_analysis: Dict):
        """Create all publication figures"""
        logger.info("🎨 Creating publication-ready figures...")
        try:
            self._create_intervention_impact_figure(all_model_results)
            self._create_layer_sensitivity_figure(all_model_results)
            if self.config.include_downstream_tasks:
                self._create_downstream_impact_figure(all_model_results)
            if self.config.include_extensive_probing:
                self._create_probing_analysis_figure(all_model_results)
            logger.info(f"✅ All figures saved to {self.figures_dir}")
        except Exception as e:
            logger.error(f"❌ Error creating figures: {e}", exc_info=True)
            
    def _create_intervention_impact_figure(self, all_model_results: Dict):
        """Figure 1: Performance degradation for different interventions across models."""
        df_records = []
        for model_name, data in all_model_results.items():
            model_short_name = model_name.split('/')[-1]
            family = data['model_config']['family']
            for intervention_key, intervention_data in data.get('intervention_results', {}).items():
                # Average degradation over all layers for this intervention
                degradations = [layer_data['performance_degradation'] for layer_data in intervention_data['results_by_layer'].values()]
                if degradations:
                    mean_degradation = np.mean(degradations)
                    df_records.append({
                        "model": model_short_name,
                        "family": family,
                        "intervention": intervention_key,
                        "degradation": mean_degradation
                    })
        
        df = pd.DataFrame(df_records)
        if df.empty: return
        
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x="intervention", y="degradation", hue="family", errorbar="sd")
        plt.title('Mean Performance Degradation by Intervention Type and Model Family', fontsize=16, fontweight='bold')
        plt.ylabel('Perplexity Degradation (%)', fontsize=12)
        plt.xlabel('Intervention Type', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yscale('symlog') # Use symlog for large value ranges
        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig1_intervention_impact_by_family.pdf", dpi=300)
        plt.close()
        logger.info("Created Figure 1: Intervention Impact by Family")

    def _create_layer_sensitivity_figure(self, all_model_results: Dict):
        """Figure 2: Layer-wise sensitivity to P1 ablation across models."""
        df_records = []
        for model_name, data in all_model_results.items():
            model_short_name = model_name.split('/')[-1]
            family = data['model_config']['family']
            # Get true num_layers from the results, not from config
            num_layers = data.get('baseline_results', {}).get('model_info', {}).get('num_layers', 32) # Get from results, with a fallback
            
            for key, intervention_data in data.get('intervention_results', {}).items():
                if 'ablation' in key: # More robust check for ablation experiments
                    layer_idx = intervention_data.get('layer_idx')
                    if layer_idx is not None:
                        df_records.append({
                            "model": model_short_name,
                            "family": family,
                            "relative_layer_depth": layer_idx / (num_layers - 1) if num_layers > 1 else 0, # Use num_layers-1 for correct 0-1 scale
                            "degradation": intervention_data.get('mean_performance_degradation', 0)
                        })

        # for model_name, data in all_model_results.items():
        #     model_short_name = model_name.split('/')[-1]
        #     family = data['model_config']['family']
        #     num_layers = self.config.target_layers.get(family, [])[-1] + 1
        #     ablation_data = data.get('intervention_results', {}).get('ablation', {})
        #     for layer, layer_data in ablation_data.get('results_by_layer', {}).items():
        #         df_records.append({
        #             "model": model_short_name,
        #             "family": family,
        #             "relative_layer_depth": layer / num_layers if num_layers > 0 else 0,
        #             "degradation": layer_data.get('performance_degradation', 0)
        #         })
        
        df = pd.DataFrame(df_records)
        if df.empty: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="relative_layer_depth", y="degradation", hue="family", marker='o', errorbar='sd')
        plt.title('Layer-wise Sensitivity to P1 Ablation', fontsize=16, fontweight='bold')
        plt.ylabel('Perplexity Degradation (%)', fontsize=12)
        plt.xlabel('Relative Layer Depth', fontsize=12)
        plt.yscale('symlog')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig2_layer_sensitivity.pdf", dpi=300)
        plt.close()
        logger.info("Created Figure 2: Layer-wise Sensitivity")

    def _create_downstream_impact_figure(self, all_model_results: Dict):
        """Figure 3: Downstream task performance degradation."""
        df_records = []
        for model_name, data in all_model_results.items():
            model_short_name = model_name.split('/')[-1]
            for task_name, task_data in data.get('downstream_results', {}).items():
                df_records.append({
                    "model": model_short_name,
                    "task": task_name.replace('_', ' ').title(),
                    "degradation": task_data.get('layer_0_ablation', {}).get('degradation', 0)
                })
        
        df = pd.DataFrame(df_records)
        if df.empty: return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="model", y="degradation", hue="task")
        plt.title('Downstream Task Accuracy Degradation from P1 Ablation (Layer 0)', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy Drop (Absolute)', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig3_downstream_impact.pdf", dpi=300)
        plt.close()
        logger.info("Created Figure 3: Downstream Task Impact")

    def _create_probing_analysis_figure(self, all_model_results: Dict):
        """Figure 4: Probing accuracy across layers for different models."""
        plt.figure(figsize=(12, 7))
        # for model_name, data in all_model_results.items():
        #     probing_data = data.get('probing_results', {}).get('category_classification', {})
        #     if probing_data:
        #         layers = sorted([int(k.split('_')[1]) for k in probing_data.keys()])
        #         accuracies = [probing_data[f"layer_{l}"]["accuracy_mean"] for l in layers]
        #         plt.plot(layers, accuracies, marker='o', label=model_name.split('/')[-1])
        
        # num_cats = self.dataset_manager.probing_dataset['category_classification'][0].get('num_categories', 5) if self.config.include_extensive_probing else 5
        # chance_level = 1.0 / num_cats
        # plt.axhline(y=chance_level, color='r', linestyle='--', label=f'Chance ({chance_level:.2f})')

        num_cats = 5 # Default
        for data in all_model_results.values():
            probing_data = data.get('probing_results', {}).get('category_classification', {})
            if probing_data:
                first_layer_key = next(iter(probing_data))
                if 'num_categories' in probing_data[first_layer_key]:
                    num_cats = probing_data[first_layer_key]['num_categories']
                    break

        chance_level = 1.0 / num_cats
        plt.axhline(y=chance_level, color='r', linestyle='--', label=f'Chance ({chance_level:.2f})')

        plt.title('P1 Category Classification Probe Accuracy Across Layers', fontsize=16, fontweight='bold')
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Cross-Validated Accuracy', fontsize=12)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig4_probing_accuracy.pdf", dpi=300)
        plt.close()
        logger.info("Created Figure 4: Probing Accuracy")

class ComprehensiveAnalysisRunner:
    """Main runner for comprehensive P1 analysis"""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        self.model_manager = ModelManager(config)
        self.dataset_manager = DatasetManager(config)
        self.single_model_analyzer = SingleModelAnalyzer(config, self.model_manager, self.dataset_manager)
        self.cross_model_analyzer = CrossModelAnalyzer(config)
        self.visualizer = PublicationVisualizer(config)
        
        self.all_model_results = {}
        self.cross_analysis = {}
    
    def run(self):
        """Run the complete comprehensive analysis"""
        logger.info("🚀 Starting Comprehensive P1 Causal Analysis")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            models_to_run = self.config.models_to_test
            logger.info(f"📋 Analysis Plan: {len(models_to_run)} models")

            # Phase 1: Single Model Analysis
            for model_config in models_to_run:
                model_name = model_config["name"]
                logger.info(f"\n📊 Analyzing Model: {model_name}")
                try:
                    model, tokenizer = self.model_manager.load_model(model_config)
                    model_results = self.single_model_analyzer.analyze_model(model, tokenizer, model_config)
                    self.all_model_results[model_name] = model_results
                    if self.config.save_intermediate_results:
                        save_model_results(model_results, self.output_dir, model_name)
                except Exception as e:
                    logger.error(f"❌ Failed to analyze {model_name}: {e}", exc_info=True)
                finally:
                    self.model_manager.unload_current_model()
            
            if not self.all_model_results: raise RuntimeError("No models were successfully analyzed")
            
            # Phase 2: Cross-Model Analysis
            self.cross_analysis = self.cross_model_analyzer.analyze(self.all_model_results)
            
            # Phase 3: Visualization & Reports
            self.visualizer.create_all_figures(self.all_model_results, self.cross_analysis)
            # (Report generation can be added here)
            
            # Phase 4: Final Results Save
            self._save_final_results()

            end_time = time.time()
            logger.info(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY! Total duration: {(end_time - start_time)/3600:.2f} hours")
            
            return self.all_model_results

        except Exception as e:
            logger.error(f"💥 Comprehensive analysis failed: {e}", exc_info=True)
            raise

    def _save_final_results(self):
        """Save the final aggregated results."""
        final_results = {
            "metadata": {"analysis_timestamp": datetime.now().isoformat(), "config": asdict(self.config)},
            "all_model_results": self.all_model_results,
            "cross_model_analysis": self.cross_analysis,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "data" / f"final_comprehensive_results_{timestamp}.json"
        
        # Helper to convert non-serializable types
        def json_converter(o):
            if isinstance(o, (np.floating, np.integer)): return o.item()
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (datetime, Path)): return str(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=json_converter)
        logger.info(f"💾 Final comprehensive results saved to {results_file}")



def create_predefined_configs() -> Dict[str, P1AnalysisConfig]:
    """Create predefined analysis configurations"""
    configs = {}
    
    # Quick test configuration
    configs["quick"] = P1AnalysisConfig(
        models_to_test=[
            {"name": "gpt2", "family": "gpt2", "size": "124M", "priority": "high"},
            {"name": "gpt2-medium", "family": "gpt2", "size": "355M", "priority": "high"},
        ],
        include_downstream_tasks=False,
        include_extensive_probing=False,
        perplexity_samples_per_category=3,
        probing_dataset_size=50,
        output_dir="./quick_p1_analysis"
    )
    
    # Standard analysis configuration
    configs["standard"] = P1AnalysisConfig(
        models_to_test=[
            {"name": "gpt2-medium", "family": "gpt2", "size": "355M", "priority": "high"},
            {"name": "meta-llama/Llama-2-7b-hf", "family": "llama", "size": "7B", "priority": "high"},
            {"name": "mistralai/Mistral-7B-v0.1", "family": "mistral", "size": "7B", "priority": "high"},
        ],
        include_downstream_tasks=True,
        include_extensive_probing=True,
        perplexity_samples_per_category=5,
        probing_dataset_size=300,
        output_dir="./standard_p1_analysis"
    )
    
    # Comprehensive analysis configuration
    configs["comprehensive"] = P1AnalysisConfig(
        models_to_test=[
            {"name": "gpt2", "family": "gpt2", "size": "124M", "priority": "high"},
            {"name": "gpt2-medium", "family": "gpt2", "size": "355M", "priority": "high"},
            {"name": "meta-llama/Llama-2-7b-hf", "family": "llama", "size": "7B", "priority": "high"},
            {"name": "mistralai/Mistral-7B-v0.1", "family": "mistral", "size": "7B", "priority": "high"},
            {"name": "microsoft/DialoGPT-medium", "family": "gpt2", "size": "355M", "priority": "medium"},
        ],
        include_downstream_tasks=True,
        include_extensive_probing=True,
        perplexity_samples_per_category=7,
        probing_dataset_size=500,
        output_dir="./comprehensive_p1_analysis"
    )
    
    return configs

# def main():
#     """Main function"""
#     parser = argparse.ArgumentParser(description="P1 Causal Analysis Runner")
#     parser.add_argument("--config", choices=["quick", "standard", "comprehensive"], 
#                        default="standard", help="Analysis configuration")
#     args = parser.parse_args()
    
#     configs = create_predefined_configs()
#     if args.config not in configs:
#         logger.error(f"Config '{args.config}' not found. Available: {list(configs.keys())}")
#         return 1
#     config = configs[args.config]
    
#     try:
#         runner = ComprehensiveAnalysisRunner(config)
#         runner.run()
#         print("\n🚀 Publication materials generated successfully!")
#     except Exception as e:
#         print(f"\n❌ A critical error occurred during the analysis run: {e}")
#         return 1
#     return 0

def main():
    parser = argparse.ArgumentParser(description="P1 Causal Analysis Runner")
    parser.add_argument("--config", choices=["quick", "standard", "comprehensive"], default="standard", help="Analysis configuration")
    args = parser.parse_args()
    
    config = create_predefined_configs().get(args.config)
    if not config: 
        logger.error(f"Config '{args.config}' not found.")
        return 1
    
    try:
        runner = ComprehensiveAnalysisRunner(config)
        runner.run()
        print("\n🚀 Publication materials generated successfully!")
    except Exception as e: print(f"\n❌ A critical error occurred: {e}"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


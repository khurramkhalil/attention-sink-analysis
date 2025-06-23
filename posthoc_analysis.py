import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import tukey_hsd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

class PostHocAnalyzer:
    """
    Complete post-hoc analysis for significant ANOVA results in attention sink study.
    
    This addresses the critical gap in statistical interpretation by providing
    pairwise comparisons for significant main effects.
    """
    
    def __init__(self, json_file_path: str):
        """Load existing analysis results"""
        self.json_file = Path(json_file_path)
        
        if not self.json_file.exists():
            raise FileNotFoundError(f"Analysis file not found: {json_file_path}")
        
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Extract organized data for analysis
        self.plot_data = self._extract_plot_data()
        self.df = pd.DataFrame(self.plot_data)
        
        print(f"📊 Loaded data: {len(self.df)} observations")
        print(f"   Families: {self.df['Family'].unique()}")
        print(f"   Categories: {self.df['Category'].unique()}")
        print(f"   Models: {self.df['Model'].nunique()}")
    
    def _extract_plot_data(self) -> List[Dict]:
        """Extract plot data from JSON structure"""
        plot_data = []
        
        for family, models in self.data['organized_results'].items():
            for model_name, categories in models.items():
                for category, samples in categories.items():
                    for sample_idx, sample in enumerate(samples):
                        if 'attention_analysis' in sample and 'position_analysis' in sample['attention_analysis']:
                            pos_data = np.array(sample['attention_analysis']['position_analysis'])
                            avg_by_position = pos_data.mean(axis=0)  # Average across layers
                            
                            plot_data.append({
                                "Family": family,
                                "Model": model_name,
                                "Category": category,
                                "Sample_ID": sample_idx,
                                "Position_1": avg_by_position[0] if len(avg_by_position) > 0 else np.nan,
                                "Position_2": avg_by_position[1] if len(avg_by_position) > 1 else np.nan,
                                "Position_3": avg_by_position[2] if len(avg_by_position) > 2 else np.nan,
                                "Position_4": avg_by_position[3] if len(avg_by_position) > 3 else np.nan,
                            })
        
        return plot_data
    
    def perform_family_posthoc_analysis(self, dependent_var: str = "Position_1", 
                                      method: str = "tukey") -> Dict:
        """
        Perform post-hoc analysis for Family differences
        
        Args:
            dependent_var: Which variable to analyze (default: Position_1)
            method: 'tukey' for Tukey HSD or 'bonferroni' for Bonferroni correction
        """
        print(f"\n🔬 FAMILY POST-HOC ANALYSIS ({method.upper()})")
        print("=" * 50)
        
        # Prepare data for analysis
        families = sorted(self.df['Family'].unique())
        family_data = {}
        
        for family in families:
            family_values = self.df[self.df['Family'] == family][dependent_var].dropna().values
            family_data[family] = family_values
            print(f"{family}: n={len(family_values)}, μ={np.mean(family_values):.4f}, σ={np.std(family_values):.4f}")
        
        results = {
            "method": method,
            "dependent_variable": dependent_var,
            "families_tested": families,
            "descriptive_stats": {},
            "pairwise_comparisons": {},
            "effect_sizes": {},
            "summary": {}
        }
        
        # Store descriptive statistics
        for family, values in family_data.items():
            results["descriptive_stats"][family] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        if method == "tukey":
            # Tukey's HSD test
            try:
                # Prepare data for Tukey HSD
                all_values = []
                all_labels = []
                
                for family, values in family_data.items():
                    all_values.extend(values)
                    all_labels.extend([family] * len(values))
                
                # Perform Tukey HSD
                tukey_result = tukey_hsd(*family_data.values())
                
                # Extract pairwise comparisons
                for i, family1 in enumerate(families):
                    for j, family2 in enumerate(families[i+1:], i+1):
                        comparison_key = f"{family1}_vs_{family2}"
                        
                        # Get Tukey HSD results
                        p_value = tukey_result.pvalue[i][j]
                        confidence_interval = tukey_result.confidence_interval(confidence_level=0.95)
                        ci_low = confidence_interval.low[i][j]
                        ci_high = confidence_interval.high[i][j]
                        
                        # Calculate effect size (Cohen's d)
                        cohens_d = self._calculate_cohens_d(family_data[family1], family_data[family2])
                        
                        results["pairwise_comparisons"][comparison_key] = {
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "confidence_interval_95": [float(ci_low), float(ci_high)],
                            "mean_difference": float(np.mean(family_data[family1]) - np.mean(family_data[family2])),
                            "cohens_d": cohens_d,
                            "interpretation": self._interpret_effect_size(cohens_d)
                        }
                        
                        # Print results
                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        print(f"{comparison_key}: p = {p_value:.4f} {sig_marker}, d = {cohens_d:.3f}, CI = [{ci_low:.4f}, {ci_high:.4f}]")
                
            except Exception as e:
                print(f"❌ Tukey HSD failed: {e}")
                print("Falling back to Bonferroni correction...")
                return self.perform_family_posthoc_analysis(dependent_var, method="bonferroni")
        
        elif method == "bonferroni":
            # Bonferroni-corrected pairwise t-tests
            n_comparisons = len(list(combinations(families, 2)))
            alpha_corrected = 0.05 / n_comparisons
            
            print(f"Bonferroni correction: α = 0.05/{n_comparisons} = {alpha_corrected:.4f}")
            
            for family1, family2 in combinations(families, 2):
                comparison_key = f"{family1}_vs_{family2}"
                
                # Perform independent t-test
                t_stat, p_value = stats.ttest_ind(family_data[family1], family_data[family2])
                p_value_corrected = min(p_value * n_comparisons, 1.0)  # Bonferroni correction
                
                # Calculate effect size
                cohens_d = self._calculate_cohens_d(family_data[family1], family_data[family2])
                
                # Calculate confidence interval for mean difference
                mean_diff = np.mean(family_data[family1]) - np.mean(family_data[family2])
                pooled_std = np.sqrt((np.var(family_data[family1], ddof=1) + np.var(family_data[family2], ddof=1)) / 2)
                n1, n2 = len(family_data[family1]), len(family_data[family2])
                se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
                
                # 95% CI
                dof = n1 + n2 - 2
                t_critical = stats.t.ppf(0.975, dof)
                ci_low = mean_diff - t_critical * se_diff
                ci_high = mean_diff + t_critical * se_diff
                
                results["pairwise_comparisons"][comparison_key] = {
                    "t_statistic": float(t_stat),
                    "p_value_uncorrected": float(p_value),
                    "p_value_bonferroni": float(p_value_corrected),
                    "significant_uncorrected": p_value < 0.05,
                    "significant_bonferroni": p_value_corrected < 0.05,
                    "confidence_interval_95": [float(ci_low), float(ci_high)],
                    "mean_difference": float(mean_diff),
                    "cohens_d": cohens_d,
                    "interpretation": self._interpret_effect_size(cohens_d)
                }
                
                # Print results
                sig_uncorr = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                sig_corr = "***" if p_value_corrected < 0.001 else "**" if p_value_corrected < 0.01 else "*" if p_value_corrected < 0.05 else "ns"
                
                print(f"{comparison_key}:")
                print(f"   t = {t_stat:.3f}, p = {p_value:.4f} {sig_uncorr} (uncorrected)")
                print(f"   p = {p_value_corrected:.4f} {sig_corr} (Bonferroni), d = {cohens_d:.3f}")
        
        # Create summary
        significant_comparisons = [k for k, v in results["pairwise_comparisons"].items() 
                                 if v.get("significant_bonferroni" if method == "bonferroni" else "significant", False)]
        
        results["summary"] = {
            "total_comparisons": len(results["pairwise_comparisons"]),
            "significant_comparisons": len(significant_comparisons),
            "significant_pairs": significant_comparisons,
            "correction_method": method,
            "alpha_level": 0.05 / len(results["pairwise_comparisons"]) if method == "bonferroni" else 0.05
        }
        
        print(f"\n📊 SUMMARY: {len(significant_comparisons)}/{len(results['pairwise_comparisons'])} comparisons significant")
        
        return results
    
    def perform_category_posthoc_analysis(self, dependent_var: str = "Position_1", 
                                        method: str = "tukey") -> Dict:
        """
        Perform post-hoc analysis for Category differences
        """
        print(f"\n📝 CATEGORY POST-HOC ANALYSIS ({method.upper()})")
        print("=" * 50)
        
        # Prepare data
        categories = sorted(self.df['Category'].unique())
        category_data = {}
        
        for category in categories:
            category_values = self.df[self.df['Category'] == category][dependent_var].dropna().values
            category_data[category] = category_values
            print(f"{category}: n={len(category_values)}, μ={np.mean(category_values):.4f}, σ={np.std(category_values):.4f}")
        
        results = {
            "method": method,
            "dependent_variable": dependent_var,
            "categories_tested": categories,
            "descriptive_stats": {},
            "pairwise_comparisons": {},
            "effect_sizes": {},
            "summary": {}
        }
        
        # Store descriptive statistics
        for category, values in category_data.items():
            results["descriptive_stats"][category] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        if method == "tukey":
            try:
                # Tukey HSD
                tukey_result = tukey_hsd(*category_data.values())
                
                for i, cat1 in enumerate(categories):
                    for j, cat2 in enumerate(categories[i+1:], i+1):
                        comparison_key = f"{cat1}_vs_{cat2}"
                        
                        p_value = tukey_result.pvalue[i][j]
                        confidence_interval = tukey_result.confidence_interval(confidence_level=0.95)
                        ci_low = confidence_interval.low[i][j]
                        ci_high = confidence_interval.high[i][j]
                        
                        cohens_d = self._calculate_cohens_d(category_data[cat1], category_data[cat2])
                        
                        results["pairwise_comparisons"][comparison_key] = {
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "confidence_interval_95": [float(ci_low), float(ci_high)],
                            "mean_difference": float(np.mean(category_data[cat1]) - np.mean(category_data[cat2])),
                            "cohens_d": cohens_d,
                            "interpretation": self._interpret_effect_size(cohens_d)
                        }
                        
                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        print(f"{comparison_key}: p = {p_value:.4f} {sig_marker}, d = {cohens_d:.3f}")
                        
            except Exception as e:
                print(f"❌ Tukey HSD failed: {e}")
                return self.perform_category_posthoc_analysis(dependent_var, method="bonferroni")
        
        elif method == "bonferroni":
            # Bonferroni correction
            n_comparisons = len(list(combinations(categories, 2)))
            alpha_corrected = 0.05 / n_comparisons
            
            print(f"Bonferroni correction: α = 0.05/{n_comparisons} = {alpha_corrected:.4f}")
            
            for cat1, cat2 in combinations(categories, 2):
                comparison_key = f"{cat1}_vs_{cat2}"
                
                t_stat, p_value = stats.ttest_ind(category_data[cat1], category_data[cat2])
                p_value_corrected = min(p_value * n_comparisons, 1.0)
                
                cohens_d = self._calculate_cohens_d(category_data[cat1], category_data[cat2])
                
                # Confidence interval
                mean_diff = np.mean(category_data[cat1]) - np.mean(category_data[cat2])
                pooled_std = np.sqrt((np.var(category_data[cat1], ddof=1) + np.var(category_data[cat2], ddof=1)) / 2)
                n1, n2 = len(category_data[cat1]), len(category_data[cat2])
                se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
                dof = n1 + n2 - 2
                t_critical = stats.t.ppf(0.975, dof)
                ci_low = mean_diff - t_critical * se_diff
                ci_high = mean_diff + t_critical * se_diff
                
                results["pairwise_comparisons"][comparison_key] = {
                    "t_statistic": float(t_stat),
                    "p_value_uncorrected": float(p_value),
                    "p_value_bonferroni": float(p_value_corrected),
                    "significant_uncorrected": p_value < 0.05,
                    "significant_bonferroni": p_value_corrected < 0.05,
                    "confidence_interval_95": [float(ci_low), float(ci_high)],
                    "mean_difference": float(mean_diff),
                    "cohens_d": cohens_d,
                    "interpretation": self._interpret_effect_size(cohens_d)
                }
                
                sig_uncorr = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                sig_corr = "***" if p_value_corrected < 0.001 else "**" if p_value_corrected < 0.01 else "*" if p_value_corrected < 0.05 else "ns"
                
                print(f"{comparison_key}: p = {p_value_corrected:.4f} {sig_corr}, d = {cohens_d:.3f}")
        
        # Summary
        significant_comparisons = [k for k, v in results["pairwise_comparisons"].items() 
                                 if v.get("significant_bonferroni" if method == "bonferroni" else "significant", False)]
        
        results["summary"] = {
            "total_comparisons": len(results["pairwise_comparisons"]),
            "significant_comparisons": len(significant_comparisons),
            "significant_pairs": significant_comparisons,
            "correction_method": method,
            "alpha_level": 0.05 / len(results["pairwise_comparisons"]) if method == "bonferroni" else 0.05
        }
        
        print(f"\n📊 SUMMARY: {len(significant_comparisons)}/{len(results['pairwise_comparisons'])} comparisons significant")
        
        return results
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        dof = n1 + n2 - 2
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / dof)
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return float(d)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_posthoc_visualization(self, family_results: Dict, category_results: Dict, 
                                   save_path: str = "./posthoc_analysis_results.png"):
        """Create visualization of post-hoc results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Post-Hoc Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Family means with significance annotations
        families = family_results["families_tested"]
        family_means = [family_results["descriptive_stats"][f]["mean"] for f in families]
        family_stds = [family_results["descriptive_stats"][f]["std"] for f in families]
        
        bars1 = axes[0,0].bar(families, family_means, yerr=family_stds, capsize=5, alpha=0.7)
        axes[0,0].set_title('Family Means (± SD) with Post-Hoc Results')
        axes[0,0].set_ylabel('Position 1 Attention')
        
        # Add significance annotations
        y_pos = max(family_means) + max(family_stds) + 0.02
        for comparison, result in family_results["pairwise_comparisons"].items():
            if result.get("significant_bonferroni" if family_results["method"] == "bonferroni" else "significant", False):
                f1, f2 = comparison.split("_vs_")
                x1, x2 = families.index(f1), families.index(f2)
                axes[0,0].plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                sig_level = "***" if result.get("p_value_bonferroni", result["p_value"]) < 0.001 else "**" if result.get("p_value_bonferroni", result["p_value"]) < 0.01 else "*"
                axes[0,0].text((x1 + x2) / 2, y_pos + 0.005, sig_level, ha='center', fontweight='bold')
                y_pos += 0.03
        
        # Plot 2: Category means with significance annotations
        categories = category_results["categories_tested"]
        cat_means = [category_results["descriptive_stats"][c]["mean"] for c in categories]
        cat_stds = [category_results["descriptive_stats"][c]["std"] for c in categories]
        
        bars2 = axes[0,1].bar(categories, cat_means, yerr=cat_stds, capsize=5, alpha=0.7, color='orange')
        axes[0,1].set_title('Category Means (± SD) with Post-Hoc Results')
        axes[0,1].set_ylabel('Position 1 Attention')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Effect sizes heatmap (families)
        family_pairs = list(family_results["pairwise_comparisons"].keys())
        family_effect_sizes = [family_results["pairwise_comparisons"][pair]["cohens_d"] for pair in family_pairs]
        
        # Create matrix for heatmap
        n_families = len(families)
        effect_matrix = np.zeros((n_families, n_families))
        
        for pair, result in family_results["pairwise_comparisons"].items():
            f1, f2 = pair.split("_vs_")
            i, j = families.index(f1), families.index(f2)
            effect_matrix[i, j] = abs(result["cohens_d"])
            effect_matrix[j, i] = abs(result["cohens_d"])
        
        im1 = axes[1,0].imshow(effect_matrix, cmap='Reds', aspect='equal')
        axes[1,0].set_title('Family Pairwise Effect Sizes (|Cohen\'s d|)')
        axes[1,0].set_xticks(range(len(families)))
        axes[1,0].set_yticks(range(len(families)))
        axes[1,0].set_xticklabels(families)
        axes[1,0].set_yticklabels(families)
        
        # Add text annotations
        for i in range(len(families)):
            for j in range(len(families)):
                if i != j:
                    text = axes[1,0].text(j, i, f'{effect_matrix[i, j]:.2f}', 
                                        ha="center", va="center", color="white" if effect_matrix[i, j] > 0.5 else "black")
        
        plt.colorbar(im1, ax=axes[1,0], fraction=0.046, pad=0.04)
        
        # Plot 4: Summary statistics table
        axes[1,1].axis('off')
        
        # Create summary text
        summary_text = "POST-HOC ANALYSIS SUMMARY\n\n"
        summary_text += f"Family Analysis ({family_results['method'].upper()}):\n"
        summary_text += f"  Significant pairs: {len(family_results['summary']['significant_pairs'])}/{family_results['summary']['total_comparisons']}\n"
        for pair in family_results['summary']['significant_pairs']:
            result = family_results['pairwise_comparisons'][pair]
            p_val = result.get('p_value_bonferroni', result['p_value'])
            summary_text += f"    {pair}: p={p_val:.3f}, d={result['cohens_d']:.3f}\n"
        
        summary_text += f"\nCategory Analysis ({category_results['method'].upper()}):\n"
        summary_text += f"  Significant pairs: {len(category_results['summary']['significant_pairs'])}/{category_results['summary']['total_comparisons']}\n"
        for pair in category_results['summary']['significant_pairs'][:5]:  # Show first 5
            result = category_results['pairwise_comparisons'][pair]
            p_val = result.get('p_value_bonferroni', result['p_value'])
            summary_text += f"    {pair}: p={p_val:.3f}, d={result['cohens_d']:.3f}\n"
        
        if len(category_results['summary']['significant_pairs']) > 5:
            summary_text += f"    ... and {len(category_results['summary']['significant_pairs']) - 5} more\n"
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Post-hoc visualization saved: {save_path}")
        plt.show()
    
    def run_complete_posthoc_analysis(self, method: str = "tukey", save_results: bool = True) -> Dict:
        """Run complete post-hoc analysis and save results"""
        
        print("🔬 COMPREHENSIVE POST-HOC ANALYSIS")
        print("=" * 60)
        print("Completing statistical interpretation of significant ANOVA results")
        print("=" * 60)
        
        # Run family analysis
        family_results = self.perform_family_posthoc_analysis(method=method)
        
        # Run category analysis  
        category_results = self.perform_category_posthoc_analysis(method=method)
        
        # Create visualization
        self.create_posthoc_visualization(family_results, category_results)
        
        # Combine results
        complete_results = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "method": method,
            "family_analysis": family_results,
            "category_analysis": category_results,
            "interpretation": self._create_interpretation(family_results, category_results)
        }
        
        if save_results:
            # Save to JSON
            output_file = self.json_file.parent / "posthoc_analysis_results.json"
            with open(output_file, 'w') as f:
                json.dump(complete_results, f, indent=2)
            print(f"✅ Complete post-hoc results saved: {output_file}")
        
        # Print final interpretation
        self._print_final_interpretation(complete_results)
        
        return complete_results
    
    def _create_interpretation(self, family_results: Dict, category_results: Dict) -> Dict:
        """Create interpretation of post-hoc results"""
        
        interpretation = {
            "family_findings": [],
            "category_findings": [],
            "effect_size_summary": {},
            "practical_implications": []
        }
        
        # Family interpretation
        for pair, result in family_results["pairwise_comparisons"].items():
            if result.get("significant_bonferroni" if family_results["method"] == "bonferroni" else "significant", False):
                f1, f2 = pair.split("_vs_")
                mean_diff = result["mean_difference"]
                direction = "higher" if mean_diff > 0 else "lower"
                interpretation["family_findings"].append({
                    "comparison": pair,
                    "finding": f"{f1} shows significantly {direction} Position 1 attention than {f2}",
                    "magnitude": f"Mean difference: {abs(mean_diff):.3f}",
                    "effect_size": f"Cohen's d = {result['cohens_d']:.3f} ({result['interpretation']})"
                })
        
        # Category interpretation
        for pair, result in category_results["pairwise_comparisons"].items():
            if result.get("significant_bonferroni" if category_results["method"] == "bonferroni" else "significant", False):
                c1, c2 = pair.split("_vs_")
                mean_diff = result["mean_difference"]
                direction = "higher" if mean_diff > 0 else "lower"
                interpretation["category_findings"].append({
                    "comparison": pair,
                    "finding": f"'{c1}' text shows significantly {direction} Position 1 attention than '{c2}' text",
                    "magnitude": f"Mean difference: {abs(mean_diff):.3f}",
                    "effect_size": f"Cohen's d = {result['cohens_d']:.3f} ({result['interpretation']})"
                })
        
        return interpretation
    
    def _print_final_interpretation(self, complete_results: Dict):
        """Print comprehensive interpretation of results"""
        
        print(f"\n{'='*80}")
        print("COMPLETE POST-HOC ANALYSIS INTERPRETATION")
        print(f"{'='*80}")
        
        interp = complete_results["interpretation"]
        
        print("\n🏗️ FAMILY DIFFERENCES:")
        print("-" * 30)
        if interp["family_findings"]:
            for finding in interp["family_findings"]:
                print(f"• {finding['finding']}")
                print(f"  {finding['magnitude']}, {finding['effect_size']}")
        else:
            print("• No significant family differences found after correction")
        
        print("\n📝 CATEGORY DIFFERENCES:")
        print("-" * 30)
        if interp["category_findings"]:
            for finding in interp["category_findings"]:
                print(f"• {finding['finding']}")
                print(f"  {finding['magnitude']}, {finding['effect_size']}")
        else:
            print("• No significant category differences found after correction")
        
        print(f"\n🎯 KEY INSIGHTS FOR PUBLICATION:")
        print("-" * 40)
        
        # Family insights
        family_significant = len(interp["family_findings"])
        if family_significant > 0:
            print(f"✅ ARCHITECTURE EFFECTS: {family_significant} significant family differences")
            # Find strongest effect
            strongest_family = max(complete_results["family_analysis"]["pairwise_comparisons"].items(), 
                                 key=lambda x: abs(x[1]["cohens_d"]))
            print(f"   Strongest effect: {strongest_family[0]} (d = {strongest_family[1]['cohens_d']:.3f})")
        else:
            print("❌ ARCHITECTURE EFFECTS: No significant differences after correction")
        
        # Category insights
        category_significant = len(interp["category_findings"])
        if category_significant > 0:
            print(f"✅ CONTENT EFFECTS: {category_significant} significant category differences")
            # Find strongest effect
            strongest_category = max(complete_results["category_analysis"]["pairwise_comparisons"].items(), 
                                   key=lambda x: abs(x[1]["cohens_d"]))
            print(f"   Strongest effect: {strongest_category[0]} (d = {strongest_category[1]['cohens_d']:.3f})")
        else:
            print("❌ CONTENT EFFECTS: No significant differences after correction")
        
        print(f"\n💡 IMPLICATIONS FOR PAPER:")
        print("-" * 30)
        
        if family_significant > 0:
            print("• Can claim statistically validated architectural differences")
            print("• Mistral's unique behavior is likely statistically significant")
            print("• Architecture-specific optimization strategies are justified")
        
        if category_significant > 0:
            print("• Can claim statistically validated content-dependent effects")
            print("• Content-aware sink selection is empirically supported")
            print("• Different text types require different attention strategies")
        
        if family_significant == 0 and category_significant == 0:
            print("• Main effects not significant after multiple comparison correction")
            print("• Focus on overall Position 1 dominance (which is highly significant)")
            print("• May need larger sample sizes for subtle between-group effects")
        
        print(f"\n📊 STATISTICAL REPORTING FOR PAPER:")
        print("-" * 40)
        
        method_name = complete_results["method"].capitalize()
        
        if family_significant > 0:
            print(f"Family Analysis: One-way ANOVA followed by {method_name} post-hoc tests")
            print(f"revealed {family_significant} significant pairwise differences:")
            for finding in interp["family_findings"][:3]:  # Show top 3
                pair = finding["comparison"].replace("_vs_", " vs ")
                d_val = finding["effect_size"].split("=")[1].split("(")[0].strip()
                print(f"  • {pair}: d = {d_val}")
        
        if category_significant > 0:
            print(f"\nCategory Analysis: One-way ANOVA followed by {method_name} post-hoc tests")
            print(f"revealed {category_significant} significant pairwise differences:")
            for finding in interp["category_findings"][:3]:  # Show top 3
                pair = finding["comparison"].replace("_vs_", " vs ")
                d_val = finding["effect_size"].split("=")[1].split("(")[0].strip()
                print(f"  • {pair}: d = {d_val}")
        
        print(f"\n{'='*80}")

def main():
    """
    Main function to run post-hoc analysis on existing results
    """
    
    print("🔬 POST-HOC STATISTICAL ANALYSIS")
    print("=" * 50)
    print("Completing ANOVA interpretation with pairwise comparisons")
    print("=" * 50)
    
    # Path to your existing results
    json_file_path = "./expanded_sink_analysis/comprehensive_analysis_results.json"
    
    try:
        # Initialize analyzer
        analyzer = PostHocAnalyzer(json_file_path)
        
        # Run complete analysis
        # Try Tukey HSD first (more conservative), fall back to Bonferroni if needed
        results = analyzer.run_complete_posthoc_analysis(method="tukey", save_results=True)
        
        print(f"\n✅ POST-HOC ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {analyzer.json_file.parent}")
        print(f"🔍 Check 'posthoc_analysis_results.json' for complete statistical details")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Please ensure you have run the main analysis first")
        print(f"   Expected file: {json_file_path}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"💡 Please check your data format and try again")

if __name__ == "__main__":
    main()
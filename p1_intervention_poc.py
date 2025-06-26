"""
Proof of Concept: P1 Causal Intervention Framework
==================================================

This script implements a proof-of-concept for Phase 1 (Causal Interventions) 
using GPT-2 medium as the test model. It builds on the existing codebase to 
add intervention capabilities with comprehensive logging and visualization.

Author: Research Team
Date: June 2025
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime
import warnings
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import pickle

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class P1InterventionAnalyzer:
    """
    Proof-of-concept analyzer for P1 causal interventions.
    Focuses on GPT-2 medium for initial validation.
    """
    
    def __init__(self, output_dir: str = "./p1_intervention_poc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model_name = "gpt2-medium"
        self._load_model()
        
        # Prepare test texts (subset from original paper)
        self.test_texts = self._prepare_test_texts()
        
        # Results storage
        self.results = {
            "metadata": {
                "model": self.model_name,
                "device": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "num_test_texts": len(self.test_texts)
            },
            "baseline_results": {},
            "intervention_results": {},
            "statistical_analysis": {},
            "probing_results": {}
        }
    
    def _load_model(self):
        """Load GPT-2 medium model and tokenizer"""
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()
        
        # Pre-compute mean embedding for consistent mean_ablation intervention
        with torch.no_grad():
            all_embeddings = self.model.get_input_embeddings().weight.data
            self._mean_embedding = all_embeddings.mean(dim=0).to(self.device)
        logger.info("Pre-computed mean embedding vector for interventions")
        
        logger.info(f"Model loaded. Layers: {self.model.config.n_layer}")
    
    def _prepare_test_texts(self) -> List[Dict]:
        """Prepare diverse test texts for intervention analysis"""
        texts = [
            {
                "category": "narrative",
                "text": "The ancient castle stood majestically on the hilltop, its weathered stones telling tales of centuries past. Knights once roamed these halls, their armor clanking as they prepared for battle."
            },
            {
                "category": "narrative", 
                "text": "Sarah walked through the misty forest, her footsteps echoing softly on the damp leaves. The moonlight filtered through the canopy, creating dancing shadows that seemed alive."
            },
            {
                "category": "technical", 
                "text": "Machine learning algorithms utilize mathematical optimization techniques to minimize loss functions. Gradient descent iteratively adjusts model parameters to improve prediction accuracy."
            },
            {
                "category": "technical",
                "text": "Neural networks consist of interconnected layers of artificial neurons that process information through weighted connections. Backpropagation enables efficient training by computing gradients."
            },
            {
                "category": "dialogue",
                "text": "\"Hello, how are you today?\" she asked with a warm smile. \"I'm doing well, thank you for asking,\" he replied, adjusting his glasses nervously."
            },
            {
                "category": "dialogue",
                "text": "\"Can you help me with this problem?\" the student inquired. \"Of course, let's work through it step by step,\" the teacher responded encouragingly."
            },
            {
                "category": "code",
                "text": "def calculate_attention(query, key, value):\n    scores = torch.matmul(query, key.transpose(-2, -1))\n    attention_weights = F.softmax(scores, dim=-1)\n    return torch.matmul(attention_weights, value)"
            },
            {
                "category": "code",
                "text": "for i in range(len(data)):\n    if data[i] is not None:\n        result = process_item(data[i])\n        output.append(result)\n    else:\n        output.append(default_value)"
            },
            {
                "category": "short",
                "text": "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet exactly once."
            },
            {
                "category": "short",
                "text": "Time flies when you're having fun. Every moment counts in life's precious journey."
            }
        ]
        
        logger.info(f"Prepared {len(texts)} test texts across {len(set(t['category'] for t in texts))} categories")
        logger.info(f"Samples per category: {dict(pd.Series([t['category'] for t in texts]).value_counts())}")
        return texts
    
    def measure_baseline_performance(self):
        """Measure baseline performance without interventions"""
        logger.info("Measuring baseline performance...")
        
        baseline_results = {
            "perplexity": {},
            "attention_patterns": {},
            "hidden_states": {}
        }
        
        for i, text_info in enumerate(self.test_texts):
            text = text_info["text"]
            category = text_info["category"]
            
            logger.info(f"Processing baseline for {category} text {i+1}/{len(self.test_texts)}")
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Calculate perplexity with proper padding handling
                logits = outputs.logits
                shifted_logits = logits[..., :-1, :].contiguous()
                labels = inputs["input_ids"][..., 1:].contiguous()
                
                # CRITICAL FIX: Ignore padding tokens in loss calculation
                loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
                
                # Extract attention patterns (focus on P1)
                attentions = outputs.attentions
                p1_attention_by_layer = []
                
                for layer_idx, layer_attn in enumerate(attentions):
                    # Average across heads and get attention to P1
                    avg_attn = layer_attn[0].mean(dim=0)  # [seq_len, seq_len]
                    if avg_attn.size(0) > 4:  # Ensure we have enough tokens
                        p1_attn = avg_attn[4:, 0].mean().item()  # Attention from tokens 4+ to P1
                        p1_attention_by_layer.append(p1_attn)
                    else:
                        p1_attention_by_layer.append(0.0)
                
                # Extract P1 hidden states
                hidden_states = outputs.hidden_states
                p1_hidden_states = [h[0, 0].cpu().numpy() for h in hidden_states]  # P1 across layers
                
            baseline_results["perplexity"][category] = perplexity
            baseline_results["attention_patterns"][category] = p1_attention_by_layer
            baseline_results["hidden_states"][category] = p1_hidden_states
            
            logger.info(f"Baseline PPL for {category}: {perplexity:.2f}")
        
        self.results["baseline_results"] = baseline_results
        self._save_results()
        
        return baseline_results
    
    def create_intervention_hook(self, intervention_type: str, layer_idx: int, **kwargs) -> Callable:
        """Create intervention hooks for different types of P1 modifications"""
        
        def intervention_hook(module, input, output):
            """Hook function that modifies P1 representations"""
            # output[0] is the hidden states: [batch_size, seq_len, hidden_dim]
            if output[0].size(1) > 0:  # Ensure we have tokens
                
                if intervention_type == "ablation":
                    # Zero out P1
                    output[0][:, 0, :] = 0.0
                
                elif intervention_type == "mean_ablation":
                    # Replace P1 with pre-computed mean embedding (consistent across all layers)
                    output[0][:, 0, :] = self._mean_embedding
                
                elif intervention_type == "noise_injection":
                    noise_std = kwargs.get("noise_std", 0.1)
                    noise = torch.randn_like(output[0][:, 0, :]) * noise_std
                    output[0][:, 0, :] += noise
                
                elif intervention_type == "random_replacement":
                    # Replace with random vector from normal distribution
                    output[0][:, 0, :] = torch.randn_like(output[0][:, 0, :])
            
            return output
        
        return intervention_hook
    
    def run_intervention_experiment(self, intervention_type: str, layer_indices: List[int], **kwargs):
        """Run intervention experiments at specified layers"""
        logger.info(f"Running {intervention_type} intervention at layers {layer_indices}")
        
        intervention_results = {
            "intervention_type": intervention_type,
            "layer_indices": layer_indices,
            "kwargs": kwargs,
            "results_by_layer": {},
            "results_by_category": {}
        }
        
        for layer_idx in layer_indices:
            logger.info(f"Testing intervention at layer {layer_idx}")
            
            layer_results = {
                "perplexity": {},
                "attention_changes": {},
                "performance_degradation": {}
            }
            
            for i, text_info in enumerate(self.test_texts):
                text = text_info["text"]
                category = text_info["category"]
                
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Register intervention hook
                hook = self.create_intervention_hook(intervention_type, layer_idx, **kwargs)
                target_layer = self.model.transformer.h[layer_idx]
                handle = target_layer.register_forward_hook(hook)
                
                try:
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                        # Calculate perplexity with proper padding handling
                        logits = outputs.logits
                        shifted_logits = logits[..., :-1, :].contiguous()
                        labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        # CRITICAL FIX: Ignore padding tokens in loss calculation
                        loss = F.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=self.tokenizer.pad_token_id,
                            reduction='mean'
                        )
                        perplexity = torch.exp(loss).item()
                        
                        # Calculate attention changes
                        attentions = outputs.attentions
                        p1_attention_by_layer = []
                        
                        for att_layer_idx, layer_attn in enumerate(attentions):
                            avg_attn = layer_attn[0].mean(dim=0)
                            if avg_attn.size(0) > 4:
                                p1_attn = avg_attn[4:, 0].mean().item()
                                p1_attention_by_layer.append(p1_attn)
                            else:
                                p1_attention_by_layer.append(0.0)
                        
                        # Calculate performance degradation
                        baseline_ppl = self.results["baseline_results"]["perplexity"][category]
                        degradation = (perplexity - baseline_ppl) / baseline_ppl
                        
                finally:
                    handle.remove()  # Clean up hook
                
                layer_results["perplexity"][category] = perplexity
                layer_results["attention_changes"][category] = p1_attention_by_layer
                layer_results["performance_degradation"][category] = degradation
                
                logger.info(f"Layer {layer_idx}, {category}: PPL {perplexity:.2f} (Δ: {degradation:.1%})")
            
            intervention_results["results_by_layer"][layer_idx] = layer_results
        
        # Organize results by category for easier analysis
        for category in [t["category"] for t in self.test_texts]:
            intervention_results["results_by_category"][category] = {
                "perplexity_by_layer": {},
                "degradation_by_layer": {}
            }
            
            for layer_idx in layer_indices:
                intervention_results["results_by_category"][category]["perplexity_by_layer"][layer_idx] = \
                    intervention_results["results_by_layer"][layer_idx]["perplexity"][category]
                intervention_results["results_by_category"][category]["degradation_by_layer"][layer_idx] = \
                    intervention_results["results_by_layer"][layer_idx]["performance_degradation"][category]
        
        return intervention_results
    
    def run_comprehensive_intervention_analysis(self):
        """Run comprehensive intervention analysis across multiple types and layers"""
        logger.info("Starting comprehensive intervention analysis...")
        
        # Define intervention configurations
        interventions = [
            {"type": "ablation", "layers": [0, 6, 12, 18, 23], "kwargs": {}},
            {"type": "noise_injection", "layers": [6, 12, 18], "kwargs": {"noise_std": 0.1}},
            {"type": "noise_injection", "layers": [6, 12, 18], "kwargs": {"noise_std": 0.5}},
            {"type": "mean_ablation", "layers": [6, 12, 18], "kwargs": {}},
            {"type": "random_replacement", "layers": [12, 18], "kwargs": {}}
        ]
        
        all_intervention_results = {}
        
        for config in interventions:
            intervention_key = f"{config['type']}_" + "_".join(map(str, config['layers']))
            if config['kwargs']:
                intervention_key += f"_{'_'.join(f'{k}{v}' for k, v in config['kwargs'].items())}"
            
            logger.info(f"Running intervention: {intervention_key}")
            
            results = self.run_intervention_experiment(
                config["type"], 
                config["layers"], 
                **config["kwargs"]
            )
            
            all_intervention_results[intervention_key] = results
        
        self.results["intervention_results"] = all_intervention_results
        self._save_results()
        
        return all_intervention_results
    
    def simple_probing_analysis(self):
        """Simple probing analysis to understand P1 information content"""
        logger.info("Starting simple probing analysis...")
        
        # Collect P1 hidden states and labels
        p1_states_by_layer = {i: [] for i in range(self.model.config.n_layer + 1)}  # +1 for embedding
        category_labels = []
        
        for text_info in self.test_texts:
            text = text_info["text"]
            category = text_info["category"]
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
                
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    p1_state = layer_hidden[0, 0].cpu().numpy()  # P1 state
                    p1_states_by_layer[layer_idx].append(p1_state)
            
            category_labels.append(category)
        
        # Train simple probes for category classification
        probing_results = {}
        categories = list(set(category_labels))
        
        # Check if we have enough samples for cross-validation
        min_samples_per_class = min(category_labels.count(cat) for cat in categories)
        max_cv_folds = min(3, min_samples_per_class)  # FIX: Ensure CV folds don't exceed class size
        
        logger.info(f"Categories: {categories}")
        logger.info(f"Min samples per class: {min_samples_per_class}")
        logger.info(f"Using {max_cv_folds} CV folds")
        
        if max_cv_folds < 2:
            logger.warning("Insufficient samples for reliable cross-validation. Using simple train/test split.")
        
        for layer_idx in range(0, self.model.config.n_layer + 1, 6):  # Sample every 6 layers
            X = np.array(p1_states_by_layer[layer_idx])
            y = [categories.index(label) for label in category_labels]
            
            if len(set(y)) > 1:  # Ensure we have multiple classes
                # Simple logistic regression probe
                probe = LogisticRegression(random_state=42, max_iter=1000)
                
                if max_cv_folds >= 2:
                    # Cross-validation
                    cv_scores = cross_val_score(probe, X, y, cv=max_cv_folds, scoring='accuracy')
                    
                    probing_results[f"layer_{layer_idx}"] = {
                        "accuracy_mean": float(np.mean(cv_scores)),
                        "accuracy_std": float(np.std(cv_scores)),
                        "num_features": X.shape[1],
                        "num_samples": X.shape[0],
                        "cv_folds": max_cv_folds,
                        "method": "cross_validation"
                    }
                    
                    logger.info(f"Layer {layer_idx} probe accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                else:
                    # Simple fit on all data (not ideal but works for PoC)
                    probe.fit(X, y)
                    accuracy = probe.score(X, y)
                    
                    probing_results[f"layer_{layer_idx}"] = {
                        "accuracy_mean": float(accuracy),
                        "accuracy_std": 0.0,
                        "num_features": X.shape[1],
                        "num_samples": X.shape[0],
                        "cv_folds": 1,
                        "method": "full_fit",
                        "warning": "Insufficient samples for cross-validation"
                    }
                    
                    logger.info(f"Layer {layer_idx} probe accuracy (full fit): {accuracy:.3f}")
        
        # Add metadata about limitations
        probing_results["metadata"] = {
            "total_samples": len(category_labels),
            "num_categories": len(categories),
            "samples_per_category": {cat: category_labels.count(cat) for cat in categories},
            "limitations": "Small sample size limits statistical reliability. Suitable for PoC only."
        }
        
        self.results["probing_results"] = probing_results
        self._save_results()
        
        return probing_results
    
    def perform_statistical_analysis(self):
        """Perform statistical analysis of intervention effects"""
        logger.info("Performing statistical analysis...")
        
        if "intervention_results" not in self.results:
            logger.warning("No intervention results found for statistical analysis")
            return
        
        statistical_results = {}
        
        # Analyze performance degradation across interventions
        for intervention_name, intervention_data in self.results["intervention_results"].items():
            intervention_stats = {
                "degradation_by_category": {},
                "degradation_summary": {},
                "layer_effects": {},
                "statistical_tests": {}
            }
            
            # Collect degradation data
            all_degradations = []
            category_degradations = {}
            
            for category in [t["category"] for t in self.test_texts]:
                if category in intervention_data["results_by_category"]:
                    degradations = list(intervention_data["results_by_category"][category]["degradation_by_layer"].values())
                    all_degradations.extend(degradations)
                    category_degradations[category] = degradations
                    
                    intervention_stats["degradation_by_category"][category] = {
                        "mean": float(np.mean(degradations)),
                        "std": float(np.std(degradations)),
                        "max": float(np.max(degradations)),
                        "min": float(np.min(degradations))
                    }
            
            # Overall degradation statistics
            if all_degradations:
                intervention_stats["degradation_summary"] = {
                    "mean": float(np.mean(all_degradations)),
                    "std": float(np.std(all_degradations)),
                    "median": float(np.median(all_degradations)),
                    "max": float(np.max(all_degradations)),
                    "min": float(np.min(all_degradations)),
                    "significant_degradation_5pct": np.mean(all_degradations) > 0.05,  # >5% degradation
                    "significant_degradation_10pct": np.mean(all_degradations) > 0.10,  # >10% degradation
                    "sample_size": len(all_degradations)
                }
                
                # Simple statistical tests (one-sample t-test against 0)
                if len(all_degradations) > 1:
                    t_stat, p_value = scipy_stats.ttest_1samp(all_degradations, 0)
                    intervention_stats["statistical_tests"]["one_sample_ttest"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant_at_05": p_value < 0.05,
                        "interpretation": "Degradation significantly different from zero" if p_value < 0.05 else "No significant degradation"
                    }
            
            # Layer-wise analysis
            for layer_idx in intervention_data["layer_indices"]:
                layer_degradations = []
                for category in category_degradations:
                    # Fix the nested access issue
                    if (layer_idx in intervention_data["results_by_layer"] and 
                        category in intervention_data["results_by_layer"][layer_idx]["performance_degradation"]):
                        layer_degradations.append(
                            intervention_data["results_by_layer"][layer_idx]["performance_degradation"][category]
                        )
                
                if layer_degradations:
                    intervention_stats["layer_effects"][f"layer_{layer_idx}"] = {
                        "mean_degradation": float(np.mean(layer_degradations)),
                        "std_degradation": float(np.std(layer_degradations)),
                        "max_degradation": float(np.max(layer_degradations)),
                        "critical_layer": np.mean(layer_degradations) > 0.1  # >10% degradation
                    }
            
            statistical_results[intervention_name] = intervention_stats
        
        # Cross-intervention comparison (if multiple interventions)
        if len(self.results["intervention_results"]) > 1:
            logger.info("Performing cross-intervention comparison...")
            all_intervention_effects = []
            intervention_names = []
            
            for intervention_name, stats in statistical_results.items():
                if "degradation_summary" in stats and "mean" in stats["degradation_summary"]:
                    all_intervention_effects.append(stats["degradation_summary"]["mean"])
                    intervention_names.append(intervention_name)
            
            if len(all_intervention_effects) > 1:
                # Find most/least impactful interventions
                max_effect_idx = np.argmax(all_intervention_effects)
                min_effect_idx = np.argmin(all_intervention_effects)
                
                statistical_results["cross_intervention_summary"] = {
                    "most_impactful": {
                        "intervention": intervention_names[max_effect_idx],
                        "mean_degradation": float(all_intervention_effects[max_effect_idx])
                    },
                    "least_impactful": {
                        "intervention": intervention_names[min_effect_idx],
                        "mean_degradation": float(all_intervention_effects[min_effect_idx])
                    },
                    "effect_range": float(np.max(all_intervention_effects) - np.min(all_intervention_effects))
                }
        
        self.results["statistical_analysis"] = statistical_results
        self._save_results()
        
        return statistical_results
    
    def create_publication_figures(self):
        """Create publication-ready figures"""
        logger.info("Creating publication-ready figures...")
        
        # Set publication style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Baseline P1 attention patterns across layers
        self._create_baseline_attention_figure()
        
        # Figure 2: Intervention effects on perplexity (only if intervention data exists)
        if self.results.get("intervention_results"):
            self._create_intervention_effects_figure()
            
            # Figure 3: Layer-wise intervention impact (only if multi-layer data exists)
            self._create_layer_wise_impact_figure()
        else:
            logger.warning("No intervention results found - skipping intervention figures")
        
        # Figure 4: Probing results (only if probing data exists)
        if self.results.get("probing_results") and len(self.results["probing_results"]) > 1:
            self._create_probing_results_figure()
        else:
            logger.warning("No sufficient probing results found - skipping probing figure")
        
        logger.info(f"All available figures saved to {self.output_dir / 'figures'}")
    
    def _create_baseline_attention_figure(self):
        """Create baseline P1 attention pattern figure"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Left panel: P1 attention by layer across categories
            categories = [t["category"] for t in self.test_texts]
            unique_categories = sorted(set(categories))
            layers = list(range(self.model.config.n_layer))
            
            for category in unique_categories:
                if category in self.results["baseline_results"]["attention_patterns"]:
                    attention_values = self.results["baseline_results"]["attention_patterns"][category]
                    if len(attention_values) == len(layers):  # Ensure data length matches
                        axes[0].plot(layers, attention_values, marker='o', label=category, linewidth=2)
                    else:
                        logger.warning(f"Attention data length mismatch for {category}: {len(attention_values)} vs {len(layers)}")
            
            axes[0].set_xlabel('Layer', fontsize=12)
            axes[0].set_ylabel('P1 Attention Strength', fontsize=12)
            axes[0].set_title('P1 Attention Across Layers by Category', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Right panel: Baseline perplexity by category
            baseline_ppls = [self.results["baseline_results"]["perplexity"][cat] for cat in unique_categories]
            bars = axes[1].bar(unique_categories, baseline_ppls, alpha=0.7)
            axes[1].set_ylabel('Perplexity', fontsize=12)
            axes[1].set_title('Baseline Perplexity by Category', fontsize=14, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, ppl in zip(bars, baseline_ppls):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{ppl:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "baseline_analysis.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created baseline analysis figure")
            
        except Exception as e:
            logger.error(f"Error creating baseline figure: {e}")
            plt.close('all')  # Clean up any open figures
    
    def _create_intervention_effects_figure(self):
        """Create intervention effects visualization"""
        if not self.results.get("intervention_results"):
            logger.warning("No intervention results to plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            categories = [t["category"] for t in self.test_texts]
            unique_categories = sorted(set(categories))
            
            # Get representative interventions for visualization
            intervention_keys = list(self.results["intervention_results"].keys())
            key_interventions = intervention_keys[:4]  # Take first 4 available
            
            for idx, intervention_name in enumerate(key_interventions):
                if idx >= 4:
                    break
                    
                intervention_data = self.results["intervention_results"][intervention_name]
                
                # Plot performance degradation by category
                degradations = []
                for category in unique_categories:
                    if category in intervention_data.get("results_by_category", {}):
                        deg_values = list(intervention_data["results_by_category"][category].get("degradation_by_layer", {}).values())
                        if deg_values:
                            degradations.append(np.mean(deg_values))
                        else:
                            degradations.append(0)
                    else:
                        degradations.append(0)
                
                if degradations:  # Only plot if we have data
                    bars = axes[idx].bar(unique_categories, degradations, alpha=0.7)
                    axes[idx].set_title(intervention_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    axes[idx].set_ylabel('Performance Degradation (%)', fontsize=10)
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Color bars by degradation level
                    for bar, deg in zip(bars, degradations):
                        if deg > 0.1:  # >10% degradation
                            bar.set_color('red')
                        elif deg > 0.05:  # >5% degradation
                            bar.set_color('orange')
                        else:
                            bar.set_color('green')
                        
                        # Add value labels
                        axes[idx].text(bar.get_x() + bar.get_width()/2, 
                                      bar.get_height() + (0.01 if deg >= 0 else -0.02), 
                                      f'{deg:.1%}', ha='center', 
                                      va='bottom' if deg >= 0 else 'top', fontsize=9)
                else:
                    axes[idx].text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                                  transform=axes[idx].transAxes, fontsize=12)
                    axes[idx].set_title(intervention_name.replace('_', ' ').title(), fontsize=12)
            
            # Hide unused subplots
            for idx in range(len(key_interventions), 4):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "intervention_effects.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created intervention effects figure")
            
        except Exception as e:
            logger.error(f"Error creating intervention effects figure: {e}")
            plt.close('all')
    
    def _create_layer_wise_impact_figure(self):
        """Create layer-wise intervention impact visualization"""
        if not self.results.get("intervention_results"):
            logger.warning("No intervention results for layer-wise analysis")
            return
        
        try:
            # Find ablation experiment with multiple layers
            ablation_experiment = None
            for key, data in self.results["intervention_results"].items():
                if "ablation" in key and len(data.get("layer_indices", [])) > 3:
                    ablation_experiment = data
                    break
            
            if not ablation_experiment:
                logger.warning("No suitable ablation experiment found for layer-wise analysis")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = [t["category"] for t in self.test_texts]
            unique_categories = sorted(set(categories))
            layers = ablation_experiment["layer_indices"]
            
            # Create heatmap of performance degradation
            degradation_matrix = []
            
            for category in unique_categories:
                if category in ablation_experiment.get("results_by_category", {}):
                    row = []
                    for layer in layers:
                        degradation = ablation_experiment["results_by_category"][category].get("degradation_by_layer", {}).get(layer, 0)
                        row.append(degradation)
                    degradation_matrix.append(row)
                else:
                    degradation_matrix.append([0] * len(layers))
            
            if not degradation_matrix or not any(any(row) for row in degradation_matrix):
                logger.warning("No degradation data found for heatmap")
                return
            
            degradation_matrix = np.array(degradation_matrix)
            
            # Create heatmap
            im = ax.imshow(degradation_matrix, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f'Layer {l}' for l in layers])
            ax.set_yticks(range(len(unique_categories)))
            ax.set_yticklabels(unique_categories)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Degradation', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(unique_categories)):
                for j in range(len(layers)):
                    text = ax.text(j, i, f'{degradation_matrix[i, j]:.1%}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title('Layer-wise P1 Ablation Impact by Text Category', fontsize=14, fontweight='bold')
            ax.set_xlabel('Intervention Layer', fontsize=12)
            ax.set_ylabel('Text Category', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "layer_wise_impact.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created layer-wise impact figure")
            
        except Exception as e:
            logger.error(f"Error creating layer-wise impact figure: {e}")
            plt.close('all')
    
    def _create_probing_results_figure(self):
        """Create probing results visualization"""
        if not self.results.get("probing_results"):
            logger.warning("No probing results to plot")
            return
        
        try:
            # Filter out metadata and get actual layer results
            layer_results = {k: v for k, v in self.results["probing_results"].items() 
                           if k.startswith("layer_") and "_" in k}
            
            if not layer_results:
                logger.warning("No layer results found in probing data")
                return
            
            # Extract layer numbers safely
            layers = []
            for k in layer_results.keys():
                try:
                    parts = k.split('_')
                    if len(parts) >= 2:
                        layer_num = int(parts[1])
                        layers.append(layer_num)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse layer number from key '{k}': {e}")
                    continue
            
            if not layers:
                logger.warning("No valid layer numbers found")
                return
            
            layers = sorted(layers)
            
            # Extract accuracies and errors for valid layers
            accuracies = []
            std_errors = []
            
            for layer in layers:
                layer_key = f"layer_{layer}"
                if layer_key in layer_results:
                    acc = layer_results[layer_key].get("accuracy_mean", 0)
                    std = layer_results[layer_key].get("accuracy_std", 0)
                    accuracies.append(acc)
                    std_errors.append(std)
                else:
                    accuracies.append(0)
                    std_errors.append(0)
            
            if not accuracies or all(acc == 0 for acc in accuracies):
                logger.warning("No valid accuracy data found")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with error bars
            ax.errorbar(layers, accuracies, yerr=std_errors, marker='o', linewidth=2, 
                       markersize=8, capsize=5, capthick=2)
            
            # Add chance level line
            num_categories = len(set(t["category"] for t in self.test_texts))
            chance_level = 1.0 / num_categories
            ax.axhline(y=chance_level, color='red', linestyle='--', alpha=0.7, 
                      label=f'Chance Level ({chance_level:.2f})')
            
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Classification Accuracy', fontsize=12)
            ax.set_title('P1 Category Classification Probe Accuracy Across Layers', 
                         fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for layer, acc, std in zip(layers, accuracies, std_errors):
                if acc > 0:  # Only label non-zero values
                    ax.text(layer, acc + std + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "probing_results.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created probing results figure")
            
        except Exception as e:
            logger.error(f"Error creating probing results figure: {e}")
            plt.close('all')
    
    def _save_results(self):
        """Save results to JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "data" / f"intervention_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Also save as latest
        latest_file = self.output_dir / "data" / "latest_results.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def run_full_proof_of_concept(self):
        """Run the complete proof-of-concept analysis"""
        logger.info("🚀 Starting P1 Intervention Proof of Concept")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Baseline measurement
            logger.info("📊 Phase 1: Measuring baseline performance")
            baseline_results = self.measure_baseline_performance()
            
            # Phase 2: Intervention experiments
            logger.info("🔬 Phase 2: Running intervention experiments")
            intervention_results = self.run_comprehensive_intervention_analysis()
            
            # Phase 3: Probing analysis
            logger.info("🧠 Phase 3: Probing P1 information content")
            probing_results = self.simple_probing_analysis()
            
            # Phase 4: Statistical analysis
            logger.info("📈 Phase 4: Statistical analysis")
            statistical_results = self.perform_statistical_analysis()
            
            # Phase 5: Create visualizations
            logger.info("🎨 Phase 5: Creating publication figures")
            self.create_publication_figures()
            
            # Generate summary report
            self.generate_summary_report()
            
            logger.info("✅ Proof of concept completed successfully!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Proof of concept failed: {e}")
            raise
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report = []
        report.append("# P1 Causal Intervention Analysis - Proof of Concept Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_name}")
        report.append("")
        
        # Baseline results summary
        report.append("## Baseline Performance Summary")
        report.append("")
        if "baseline_results" in self.results:
            baseline = self.results["baseline_results"]
            report.append("### Perplexity by Category:")
            for category, ppl in baseline["perplexity"].items():
                report.append(f"- {category.title()}: {ppl:.2f}")
            report.append("")
            
            # P1 attention patterns
            report.append("### P1 Attention Patterns:")
            for category in baseline["attention_patterns"]:
                attention_values = baseline["attention_patterns"][category]
                peak_layer = np.argmax(attention_values)
                peak_value = np.max(attention_values)
                report.append(f"- {category.title()}: Peak at layer {peak_layer} ({peak_value:.3f})")
            report.append("")
        
        # Intervention results summary
        report.append("## Intervention Results Summary")
        report.append("")
        if "statistical_analysis" in self.results:
            stats = self.results["statistical_analysis"]
            
            report.append("### Most Impactful Interventions:")
            # Sort interventions by mean degradation
            interventions_by_impact = []
            for intervention, data in stats.items():
                if "degradation_summary" in data:
                    mean_deg = data["degradation_summary"]["mean"]
                    interventions_by_impact.append((intervention, mean_deg))
            
            interventions_by_impact.sort(key=lambda x: x[1], reverse=True)
            
            for intervention, impact in interventions_by_impact[:5]:
                report.append(f"- {intervention}: {impact:.1%} average degradation")
                # Safe check for significant degradation
                degradation_summary = stats[intervention].get("degradation_summary", {})
                if (degradation_summary.get("significant_degradation_5pct", False) or 
                    degradation_summary.get("significant_degradation_10pct", False)):
                    report.append("  ⚠️ Significant impact detected")
            report.append("")
        
        # Probing results summary
        report.append("## Probing Analysis Summary")
        report.append("")
        if "probing_results" in self.results and self.results["probing_results"]:
            probing = self.results["probing_results"]
            
            # Filter out metadata and get actual layer results
            layer_results = {k: v for k, v in probing.items() 
                           if k.startswith("layer_") and isinstance(v, dict)}
            
            if layer_results:
                # Find best performing layer
                best_accuracy = 0
                best_layer = 0
                for layer_key, data in layer_results.items():
                    if isinstance(data, dict) and "accuracy_mean" in data:
                        if data["accuracy_mean"] > best_accuracy:
                            best_accuracy = data["accuracy_mean"]
                            try:
                                best_layer = int(layer_key.split('_')[1])
                            except (ValueError, IndexError):
                                best_layer = 0
                
                num_categories = len(set(t["category"] for t in self.test_texts))
                chance_level = 1.0 / num_categories
                
                report.append(f"### Category Classification Results:")
                if best_accuracy > 0:
                    report.append(f"- Best performance: Layer {best_layer} ({best_accuracy:.3f} accuracy)")
                    report.append(f"- Chance level: {chance_level:.3f}")
                    report.append(f"- Above chance: {'Yes' if best_accuracy > chance_level + 0.1 else 'No'}")
                else:
                    report.append("- No valid probing results found")
                    report.append(f"- Chance level: {chance_level:.3f}")
            else:
                report.append("### Category Classification Results:")
                report.append("- No layer results available")
            report.append("")
        else:
            report.append("### Category Classification Results:")
            report.append("- No probing analysis performed")
            report.append("")
        
        # Key findings
        report.append("## Key Findings")
        report.append("")
        
        # Analyze intervention impacts
        if "statistical_analysis" in self.results:
            significant_interventions = []
            for intervention, data in self.results["statistical_analysis"].items():
                degradation_summary = data.get("degradation_summary", {})
                if (degradation_summary.get("significant_degradation_5pct", False) or 
                    degradation_summary.get("significant_degradation_10pct", False)):
                    significant_interventions.append(intervention)
            
            if significant_interventions:
                report.append("### 🔍 Causal Evidence Found:")
                report.append(f"- {len(significant_interventions)} interventions showed significant performance degradation")
                report.append("- This suggests P1 plays a functionally important role")
                report.append("")
            else:
                report.append("### 🤔 Limited Causal Evidence:")
                report.append("- Most interventions showed minimal performance impact")
                report.append("- P1's role may be more redundant than critical")
                report.append("")
        
        # Technical notes
        report.append("## Technical Notes")
        report.append("")
        report.append("### Experimental Setup:")
        report.append(f"- Model: {self.model_name} ({self.model.config.n_layer} layers)")
        report.append(f"- Test texts: {len(self.test_texts)} samples across {len(set(t['category'] for t in self.test_texts))} categories")
        report.append(f"- Device: {self.device}")
        report.append("")
        
        report.append("### Files Generated:")
        report.append("- `baseline_analysis.pdf`: Baseline P1 attention and perplexity patterns")
        report.append("- `intervention_effects.pdf`: Performance degradation across interventions")
        report.append("- `layer_wise_impact.pdf`: Layer-specific intervention effects")
        report.append("- `probing_results.pdf`: P1 information content analysis")
        report.append("- `latest_results.json`: Complete numerical results")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / "SUMMARY_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved to {report_file}")
        
        # Also print key findings to console
        print("\n" + "="*60)
        print("🎯 PROOF OF CONCEPT - KEY FINDINGS")
        print("="*60)
        
        if "statistical_analysis" in self.results:
            significant_count = 0
            total_interventions = len(self.results["statistical_analysis"])
            
            for data in self.results["statistical_analysis"].values():
                degradation_summary = data.get("degradation_summary", {})
                if (degradation_summary.get("significant_degradation_5pct", False) or 
                    degradation_summary.get("significant_degradation_10pct", False)):
                    significant_count += 1
            
            print(f"📊 Interventions tested: {total_interventions}")
            print(f"⚡ Significant impacts: {significant_count}")
            print(f"🎲 Causal evidence: {'Strong' if significant_count > total_interventions/2 else 'Moderate' if significant_count > 0 else 'Weak'}")
        
        if "probing_results" in self.results and self.results["probing_results"]:
            # Filter out metadata and get actual layer results  
            layer_results = {k: v for k, v in self.results["probing_results"].items() 
                           if k.startswith("layer_") and isinstance(v, dict)}
            
            if layer_results:
                # Calculate best probe performance
                best_acc = 0
                for data in layer_results.values():
                    if isinstance(data, dict) and "accuracy_mean" in data:
                        best_acc = max(best_acc, data["accuracy_mean"])
                
                chance = 1.0 / len(set(t["category"] for t in self.test_texts))
                print(f"🧠 P1 information content: {best_acc:.3f} accuracy (chance: {chance:.3f})")
            else:
                print("🧠 P1 information content: No valid results")
        else:
            print("🧠 P1 information content: No probing performed")
        
        print(f"📁 All results saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function to run the proof of concept"""
    print("🚀 P1 Causal Intervention Framework - Proof of Concept")
    print("=" * 60)
    print("This script will:")
    print("1. Load GPT-2 medium model")
    print("2. Measure baseline P1 attention patterns")
    print("3. Run causal intervention experiments")
    print("4. Perform simple probing analysis")
    print("5. Generate publication-ready figures")
    print("6. Save all results in JSON format")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Exiting...")
        return
    
    try:
        # Initialize analyzer
        analyzer = P1InterventionAnalyzer()
        
        # Run full analysis
        results = analyzer.run_full_proof_of_concept()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Check the results in: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    main()
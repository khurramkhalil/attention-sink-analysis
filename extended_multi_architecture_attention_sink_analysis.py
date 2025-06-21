import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    MistralForCausalLM, 
    # Add other architectures as needed
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
import logging
from dataclasses import dataclass
import warnings
from scipy import stats
from itertools import combinations
import random

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpandedAnalysisConfig:
    """Enhanced configuration for multi-architecture analysis"""
    # Model configurations
    models_to_test: List[Dict] = None
    
    # Text sampling configuration
    texts_per_category: int = 5  # Multiple samples per category for statistical rigor
    max_length: int = 512
    
    # Analysis parameters
    num_sink_tokens: int = 4
    device: str = "auto"
    
    # Output configuration
    output_dir: str = "./expanded_sink_analysis"
    save_plots: bool = True
    save_data: bool = True
    
    # Statistical analysis
    perform_statistical_tests: bool = True
    alpha_level: float = 0.05
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = [
                # GPT-2 Family (baseline)
                {"name": "gpt2", "family": "gpt2", "size": "base"},
                {"name": "gpt2-medium", "family": "gpt2", "size": "medium"},
                
                # Llama Family (different architecture/training)
                {"name": "meta-llama/Llama-2-7b-hf", "family": "llama", "size": "7b"},
                
                # Mistral (sliding window attention)
                {"name": "mistralai/Mistral-7B-v0.1", "family": "mistral", "size": "7b"},
                
                # Microsoft Models (different training data/objectives)
                {"name": "microsoft/DialoGPT-medium", "family": "gpt2", "size": "medium"},
                
                # Add more as resources allow:
                # {"name": "microsoft/CodeGPT-small-py", "family": "gpt2", "size": "small"},
                # {"name": "EleutherAI/pythia-1.4b", "family": "pythia", "size": "1.4b"},
            ]

class MultiArchitectureAttentionAnalyzer:
    """
    Enhanced analyzer for attention sink patterns across different model architectures
    with statistical rigor and multiple samples per category.
    """
    
    def __init__(self, config: ExpandedAnalysisConfig):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create diverse text samples for statistical analysis
        self.text_samples = self._create_diverse_text_samples()
        
        logger.info(f"Initialized analyzer with {len(self.config.models_to_test)} models")
        logger.info(f"Text categories: {len(self.text_samples)} with {self.config.texts_per_category} samples each")
        
    def _setup_device(self) -> torch.device:
        """Setup computation device with better memory management"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
        return device
    
    def _create_diverse_text_samples(self) -> Dict[str, List[str]]:
        """
        Create multiple diverse samples for each text category for statistical rigor
        """
        text_samples = {
            "narrative": [
                """In the heart of the bustling city, amidst towering skyscrapers and ceaseless traffic, 
                lay a small forgotten park. This verdant oasis, named Willow Creek, was a relic from a bygone era, 
                a testament to nature's resilience. Ancient willow trees drooped gracefully over a serene pond, 
                their leaves whispering secrets to the gentle breeze. Despite being surrounded by concrete and steel, 
                Willow Creek maintained an air of tranquility, a sanctuary for those seeking momentary escape from urban chaos.""",
                
                """The old lighthouse stood sentinel on the rocky cliff, its weathered stone walls bearing witness 
                to countless storms. For over a century, it had guided ships safely to harbor, its beacon cutting 
                through fog and darkness. The lighthouse keeper's quarters, though modest, held stories of generations 
                who had tended this vital light. Now automated, the structure served as a monument to maritime history, 
                attracting visitors who came to admire both its architecture and the spectacular ocean views.""",
                
                """Deep in the Amazon rainforest, Dr. Elena Martinez carefully documented a new species of orchid. 
                The delicate flower, no larger than a thumbnail, displayed intricate patterns that had evolved over millennia. 
                Her research team had been tracking this particular species for months, following indigenous knowledge 
                passed down through generations. The discovery represented not just scientific achievement, but a bridge 
                between traditional wisdom and modern conservation efforts.""",
                
                """The antique bookshop occupied a narrow building between a café and a tailor's shop. Inside, 
                countless volumes lined floor-to-ceiling shelves, their spines creating a mosaic of colors and eras. 
                Mrs. Chen, the elderly proprietor, knew every book's location and story. Customers often spent hours 
                browsing, discovering forgotten classics and rare first editions. The shop was more than a business; 
                it was a repository of human knowledge and imagination.""",
                
                """On the distant planet of Kepler-442b, the research colony faced its greatest challenge yet. 
                The red dwarf star's unpredictable solar flares threatened their delicate ecosystem experiments. 
                Commander Sarah Liu monitored the atmospheric processors while her team prepared for the next phase 
                of terraforming. The colony represented humanity's first serious attempt at interplanetary agriculture, 
                with implications that would determine the species' future among the stars."""
            ],
            
            "technical": [
                """The Transformer architecture revolutionized natural language processing through its self-attention mechanism. 
                Unlike recurrent neural networks, Transformers process sequences in parallel, computing attention weights 
                between all token pairs simultaneously. Multi-head attention allows models to focus on different 
                representation aspects. Layer normalization and residual connections stabilize training, while positional 
                encodings provide sequence order information. Recent developments like BERT and GPT demonstrate 
                the power of pre-training large Transformer models on massive text corpora.""",
                
                """Quantum computing leverages quantum mechanical phenomena to process information in fundamentally 
                different ways than classical computers. Quantum bits (qubits) can exist in superposition states, 
                allowing simultaneous representation of multiple values. Quantum entanglement creates correlations 
                between qubits that enable certain algorithms to achieve exponential speedups. However, quantum 
                decoherence and gate errors present significant challenges for maintaining quantum states during computation.""",
                
                """Blockchain technology implements distributed ledger systems through cryptographic hash functions 
                and consensus mechanisms. Each block contains a cryptographic hash of the previous block, transaction data, 
                and a timestamp, creating an immutable chain. Proof-of-work consensus requires computational effort 
                to validate transactions, while proof-of-stake systems rely on economic incentives. Smart contracts 
                enable programmable transaction logic, expanding blockchain applications beyond simple value transfer.""",
                
                """Machine learning optimization algorithms iteratively minimize loss functions through gradient descent 
                and its variants. Stochastic gradient descent introduces randomness to escape local minima, while 
                adaptive learning rate methods like Adam adjust step sizes based on gradient history. Regularization 
                techniques such as dropout and weight decay prevent overfitting. Batch normalization normalizes layer 
                inputs to stabilize training and accelerate convergence in deep neural networks.""",
                
                """Distributed systems architecture addresses challenges of scalability, fault tolerance, and consistency 
                across multiple nodes. The CAP theorem states that systems can guarantee at most two of consistency, 
                availability, and partition tolerance. Microservices decompose applications into loosely coupled services 
                communicating through APIs. Load balancing distributes requests across servers, while circuit breakers 
                prevent cascade failures during service outages."""
            ],
            
            "dialogue": [
                """"Good morning, Professor Chen!" Sarah called out as she entered the laboratory.
                "Ah, Sarah! Perfect timing. I wanted to discuss your latest research findings."
                "About the protein folding simulations? I think we're seeing some interesting patterns."
                "Exactly. The results suggest our hypothesis about alpha-helix stability might need revision."
                "I've been thinking the same thing. Should we design additional experiments?"
                "Yes, let's schedule a team meeting for next week to plan the next phase."
                "I'll prepare a detailed analysis of the anomalous results we've been seeing."
                "Excellent. Your systematic approach is exactly what this project needs."
                "Thank you, Professor. I'm excited to see where this research leads us." """,
                
                """"Welcome to Giovanni's! Table for two tonight?" the hostess asked warmly.
                "Yes, please. We have a reservation under Johnson."
                "Perfect! Right this way. Your server will be Maria, and she'll be right with you."
                "Thank you. This place has such a lovely atmosphere."
                "We're proud of our authentic Italian ambiance. The owner's family recipes go back generations."
                "That's wonderful. We're celebrating our anniversary tonight."
                "How special! I'll make sure Maria knows – we have a complimentary dessert for celebrations."
                "That's very kind of you. We really appreciate the thoughtful service."
                "Enjoy your evening, and congratulations on your anniversary!" """,
                
                """"Dr. Williams, the patient in room 302 is asking about her test results," the nurse reported.
                "Has the lab work come back yet?"
                "Yes, everything looks normal except for slightly elevated white blood cell count."
                "That's consistent with what we expected. I'll go explain the results to her."
                "She seemed quite anxious when I checked her vitals this morning."
                "I understand. These procedures can be stressful. I'll make sure to address all her concerns."
                "Should I prepare the discharge paperwork?"
                "Yes, assuming she has no other questions. We'll schedule a follow-up in two weeks."
                "I'll get everything ready and coordinate with her family."
                "Thank you. Your attention to patient care makes all the difference." """,
                
                """"The quarterly board meeting will now come to order," Chairman Roberts announced.
                "Thank you, Chairman. First item is the financial report," CFO Martinez began.
                "Revenue increased 12% year-over-year, exceeding our projections."
                "Excellent news. What factors contributed to this growth?"
                "Primarily our expansion into the Asian markets and improved operational efficiency."
                "Any concerns about sustainability of this growth rate?"
                "We're monitoring supply chain costs closely, but outlook remains positive."
                "Good. Let's move to the strategic planning discussion."
                "I've prepared a presentation on our five-year expansion roadmap."
                "Perfect. The board is eager to review the long-term vision." """,
                
                """"Emergency dispatch, what's your location?" the operator asked urgently.
                "We're at 1247 Oak Street, apartment 4B. There's been an accident."
                "What type of emergency are you reporting?"
                "Someone fell down the stairs. They're conscious but can't move their leg."
                "I'm dispatching paramedics immediately. Can you stay on the line?"
                "Yes, I'm with the injured person now. She's alert and talking."
                "Good. Don't try to move her. Paramedics should arrive in approximately 4 minutes."
                "Okay, I can hear the sirens now. Thank you for your help."
                "You did the right thing calling immediately. The paramedics will take good care of her."
                "The ambulance just pulled up. I feel much better knowing help is here." """
            ],
            
            "code": [
                """def fibonacci(n):
                    if n <= 1:
                        return n
                    else:
                        return fibonacci(n-1) + fibonacci(n-2)

                def factorial(n):
                    if n == 0 or n == 1:
                        return 1
                    else:
                        return n * factorial(n-1)

                # Example usage with error handling
                try:
                    result = fibonacci(10)
                    print(f"Fibonacci(10): {result}")
                    
                    fact_result = factorial(5)
                    print(f"Factorial(5): {fact_result}")
                except RecursionError:
                    print("Input too large for recursive approach")""",
                
                """import numpy as np
                from sklearn.model_selection import train_test_split
                from sklearn.linear_regression import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score

                class DataAnalyzer:
                    def __init__(self, data):
                        self.data = data
                        self.model = LinearRegression()
                    
                    def preprocess(self):
                        # Remove outliers using IQR method
                        Q1 = self.data.quantile(0.25)
                        Q3 = self.data.quantile(0.75)
                        IQR = Q3 - Q1
                        return self.data[~((self.data < (Q1 - 1.5 * IQR)) | 
                                         (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
                    
                    def train_model(self, X, y):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        self.model.fit(X_train, y_train)
                        return X_test, y_test""",
                
                """class BinarySearchTree:
                    def __init__(self, value):
                        self.value = value
                        self.left = None
                        self.right = None
                    
                    def insert(self, value):
                        if value < self.value:
                            if self.left is None:
                                self.left = BinarySearchTree(value)
                            else:
                                self.left.insert(value)
                        else:
                            if self.right is None:
                                self.right = BinarySearchTree(value)
                            else:
                                self.right.insert(value)
                    
                    def search(self, target):
                        if target == self.value:
                            return True
                        elif target < self.value and self.left:
                            return self.left.search(target)
                        elif target > self.value and self.right:
                            return self.right.search(target)
                        return False
                    
                    def inorder_traversal(self):
                        result = []
                        if self.left:
                            result.extend(self.left.inorder_traversal())
                        result.append(self.value)
                        if self.right:
                            result.extend(self.right.inorder_traversal())
                        return result""",
                
                """async function fetchUserData(userId) {
                    try {
                        const response = await fetch(`/api/users/${userId}`);
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        const userData = await response.json();
                        return userData;
                    } catch (error) {
                        console.error('Error fetching user data:', error);
                        return null;
                    }
                }

                class UserManager {
                    constructor() {
                        this.users = new Map();
                        this.cache = new Map();
                    }
                    
                    async getUser(userId) {
                        if (this.cache.has(userId)) {
                            return this.cache.get(userId);
                        }
                        
                        const userData = await fetchUserData(userId);
                        if (userData) {
                            this.cache.set(userId, userData);
                            this.users.set(userId, userData);
                        }
                        return userData;
                    }
                    
                    invalidateCache(userId) {
                        this.cache.delete(userId);
                    }
                }""",
                
                """#!/bin/bash

                # Database backup script with error handling
                set -euo pipefail

                DB_NAME="production_db"
                BACKUP_DIR="/var/backups/database"
                DATE=$(date +%Y%m%d_%H%M%S)
                BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${DATE}.sql"

                # Create backup directory if it doesn't exist
                mkdir -p "$BACKUP_DIR"

                # Function to log messages
                log_message() {
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${BACKUP_DIR}/backup.log"
                }

                # Function to cleanup old backups
                cleanup_old_backups() {
                    find "$BACKUP_DIR" -name "${DB_NAME}_*.sql" -mtime +7 -delete
                    log_message "Cleaned up backups older than 7 days"
                }

                # Perform database backup
                log_message "Starting database backup for $DB_NAME"
                if mysqldump "$DB_NAME" > "$BACKUP_FILE"; then
                    log_message "Backup completed successfully: $BACKUP_FILE"
                    gzip "$BACKUP_FILE"
                    cleanup_old_backups
                else
                    log_message "Backup failed for $DB_NAME"
                    exit 1
                fi"""
            ],
            
            "short": [
                "The quick brown fox jumps over the lazy dog. This pangram contains every letter.",
                "Machine learning algorithms learn patterns from data to make predictions on new examples.",
                "Climate change affects global weather patterns and requires immediate coordinated action.",
                "Quantum computers use quantum mechanics to solve certain problems exponentially faster.",
                "Artificial intelligence systems can now generate human-like text and creative content."
            ]
        }
        
        # Validate we have the right number of samples
        for category, samples in text_samples.items():
            if len(samples) < self.config.texts_per_category:
                logger.warning(f"Category '{category}' has {len(samples)} samples, need {self.config.texts_per_category}")
        
        return text_samples
    
    def load_model_safely(self, model_config: Dict) -> Tuple[Optional[torch.nn.Module], Optional[any]]:
        """
        Safely load models with proper error handling and memory management
        """
        model_name = model_config["name"]
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Handle pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model with appropriate settings based on family
            model_kwargs = {
                "output_attentions": True,
                "output_hidden_states": True,
                "trust_remote_code": True,
            }
            
            # Adjust settings based on model size and available memory
            if "7b" in model_config.get("size", "").lower():
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto" if torch.cuda.is_available() else None,
                    "low_cpu_mem_usage": True,
                })
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Move to device if not using device_map
            if not (torch.cuda.is_available() and "device_map" in model_kwargs):
                model = model.to(self.device)
            
            model.eval()
            
            # Resize embeddings if we added tokens
            if tokenizer.pad_token == '[PAD]':
                model.resize_token_embeddings(len(tokenizer))
            
            logger.info(f"✅ Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return None, None
    
    def analyze_single_combination(self, model: torch.nn.Module, tokenizer: any, 
                                 text: str, model_config: Dict, text_category: str, 
                                 text_index: int) -> Dict:
        """
        Analyze a single model-text combination
        """
        try:
            # Process text
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_length
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            actual_length = attention_mask.sum().item()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:actual_length])
            
            if actual_length <= self.config.num_sink_tokens:
                logger.warning(f"Text too short ({actual_length} tokens) for sink analysis")
                return {"error": "text_too_short", "length": actual_length}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True
                )
            
            # Extract outputs
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            
            # Move to CPU for analysis
            attentions = tuple(attn.cpu().float() for attn in attentions)
            hidden_states = tuple(hs.cpu().float() for hs in hidden_states)
            
            # Analyze attention patterns
            attention_results = self._analyze_attention_patterns(attentions, actual_length, tokens)
            
            # Analyze representations
            repr_results = self._analyze_representations(hidden_states, actual_length)
            
            return {
                "model_config": model_config,
                "text_category": text_category,
                "text_index": text_index,
                "sequence_length": actual_length,
                "attention_analysis": attention_results,
                "representation_analysis": repr_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {model_config['name']} on {text_category}[{text_index}]: {e}")
            return {
                "error": str(e),
                "model_config": model_config,
                "text_category": text_category,
                "text_index": text_index,
                "success": False
            }
    
    def _analyze_attention_patterns(self, attentions: Tuple, actual_length: int, tokens: List[str]) -> Dict:
        """Analyze attention sink patterns"""
        effective_sinks = min(self.config.num_sink_tokens, actual_length)
        
        if actual_length <= effective_sinks:
            return {"error": "sequence_too_short"}
        
        results = {
            "layer_averages": [],
            "position_analysis": [],
            "effective_sinks": effective_sinks,
            "sequence_length": actual_length
        }
        
        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn = layer_attn[0]  # Remove batch dimension
            layer_attn = layer_attn[:, :actual_length, :actual_length]
            
            # Average across heads
            avg_attn = layer_attn.mean(dim=0)
            
            # Attention from non-sink to sink tokens
            attn_to_sinks = avg_attn[effective_sinks:, :effective_sinks]
            
            # Layer average
            layer_avg = attn_to_sinks.sum(dim=1).mean().item()
            results["layer_averages"].append(layer_avg)
            
            # Position analysis
            position_scores = attn_to_sinks.mean(dim=0).tolist()
            results["position_analysis"].append(position_scores)
        
        return results
    
    def _analyze_representations(self, hidden_states: Tuple, actual_length: int) -> Dict:
        """Analyze sink token representations"""
        effective_sinks = min(self.config.num_sink_tokens, actual_length)
        
        if effective_sinks == 0:
            return {"error": "no_sinks_to_analyze"}
        
        layer_outputs = hidden_states[1:]  # Skip embedding layer
        results = {
            "similarity_analysis": [],
            "effective_sinks": effective_sinks
        }
        
        for layer_hs in layer_outputs:
            sink_states = layer_hs[0, :effective_sinks, :]
            
            if effective_sinks > 1:
                similarity_matrix = F.cosine_similarity(
                    sink_states.unsqueeze(1), 
                    sink_states.unsqueeze(0), 
                    dim=2
                )
                mask = ~torch.eye(effective_sinks, dtype=bool)
                avg_similarity = similarity_matrix[mask].mean().item()
            else:
                avg_similarity = 1.0
            
            results["similarity_analysis"].append(avg_similarity)
        
        return results
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis across all models and text samples
        """
        logger.info("Starting comprehensive multi-architecture analysis")
        
        all_results = []
        successful_runs = 0
        total_runs = len(self.config.models_to_test) * len(self.text_samples) * self.config.texts_per_category
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MULTI-ARCHITECTURE ATTENTION SINK ANALYSIS")
        print(f"{'='*80}")
        print(f"Models: {len(self.config.models_to_test)}")
        print(f"Text categories: {len(self.text_samples)}")
        print(f"Samples per category: {self.config.texts_per_category}")
        print(f"Total analysis runs: {total_runs}")
        print(f"{'='*80}\n")
        
        for model_idx, model_config in enumerate(self.config.models_to_test):
            print(f"\n🤖 MODEL {model_idx + 1}/{len(self.config.models_to_test)}: {model_config['name']}")
            print(f"Family: {model_config['family']}, Size: {model_config['size']}")
            print("=" * 60)
            
            # Load model
            model, tokenizer = self.load_model_safely(model_config)
            
            if model is None:
                logger.error(f"Skipping {model_config['name']} due to loading failure")
                continue
            
            for category, texts in self.text_samples.items():
                print(f"\n📝 Processing category: {category}")
                
                # Process multiple samples from this category
                for text_idx in range(min(self.config.texts_per_category, len(texts))):
                    text = texts[text_idx].strip()
                    print(f"   Sample {text_idx + 1}/{self.config.texts_per_category}: {text[:50]}...")
                    
                    result = self.analyze_single_combination(
                        model, tokenizer, text, model_config, category, text_idx
                    )
                    
                    if result.get("success", False):
                        print(f"   ✅ Success - Seq len: {result['sequence_length']}")
                        successful_runs += 1
                    else:
                        print(f"   ❌ Failed - {result.get('error', 'Unknown error')}")
                    
                    all_results.append(result)
            
            # Clear memory after each model
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n✅ Completed {model_config['name']}")
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE: {successful_runs}/{total_runs} successful runs")
        print(f"{'='*80}")
        
        # Process results
        processed_results = self._process_and_save_results(all_results)
        
        # Perform statistical analysis
        if self.config.perform_statistical_tests:
            statistical_results = self._perform_statistical_analysis(processed_results)
            processed_results["statistical_analysis"] = statistical_results
        
        return processed_results
    
    def _process_and_save_results(self, all_results: List[Dict]) -> Dict:
        """
        Process and save results with enhanced data organization
        """
        logger.info("Processing and organizing results...")
        
        # Filter successful results
        successful_results = [r for r in all_results if r.get("success", False)]
        
        if not successful_results:
            logger.warning("No successful results to process!")
            return {"error": "no_successful_results"}
        
        # Organize results by model family and text category
        organized_results = {}
        
        for result in successful_results:
            model_name = result["model_config"]["name"]
            family = result["model_config"]["family"]
            category = result["text_category"]
            
            # Create nested structure: family -> model -> category -> samples
            if family not in organized_results:
                organized_results[family] = {}
            if model_name not in organized_results[family]:
                organized_results[family][model_name] = {}
            if category not in organized_results[family][model_name]:
                organized_results[family][model_name][category] = []
            
            organized_results[family][model_name][category].append(result)
        
        # Create summary statistics
        summary_stats = self._create_summary_statistics(organized_results)
        
        # Save detailed results
        detailed_results = {
            "config": {
                "models_tested": self.config.models_to_test,
                "texts_per_category": self.config.texts_per_category,
                "num_sink_tokens": self.config.num_sink_tokens,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            },
            "organized_results": organized_results,
            "summary_statistics": summary_stats,
            "raw_results": all_results
        }
        
        # Save to JSON
        results_file = self.output_dir / "comprehensive_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._convert_for_json(detailed_results), f, indent=2)
        
        logger.info(f"✅ Detailed results saved to: {results_file}")
        
        return detailed_results
    
    def _create_summary_statistics(self, organized_results: Dict) -> Dict:
        """
        Create comprehensive summary statistics across architectures and text types
        """
        summary = {
            "by_family": {},
            "by_text_category": {},
            "position_analysis": {},
            "layer_analysis": {},
            "representation_analysis": {}
        }
        
        # Collect all data points for analysis
        all_position_data = []
        all_layer_data = []
        all_repr_data = []
        
        for family, models in organized_results.items():
            family_position_data = []
            family_layer_data = []
            family_repr_data = []
            
            for model_name, categories in models.items():
                for category, samples in categories.items():
                    for sample in samples:
                        # Position analysis
                        if "position_analysis" in sample["attention_analysis"]:
                            pos_data = np.array(sample["attention_analysis"]["position_analysis"])
                            avg_by_position = pos_data.mean(axis=0)  # Average across layers
                            
                            position_entry = {
                                "family": family,
                                "model": model_name,
                                "category": category,
                                "position_1": avg_by_position[0] if len(avg_by_position) > 0 else np.nan,
                                "position_2": avg_by_position[1] if len(avg_by_position) > 1 else np.nan,
                                "position_3": avg_by_position[2] if len(avg_by_position) > 2 else np.nan,
                                "position_4": avg_by_position[3] if len(avg_by_position) > 3 else np.nan,
                            }
                            all_position_data.append(position_entry)
                            family_position_data.append(position_entry)
                        
                        # Layer analysis
                        if "layer_averages" in sample["attention_analysis"]:
                            layer_avgs = sample["attention_analysis"]["layer_averages"]
                            layer_entry = {
                                "family": family,
                                "model": model_name,
                                "category": category,
                                "avg_attention": np.mean(layer_avgs),
                                "peak_attention": np.max(layer_avgs),
                                "peak_layer": np.argmax(layer_avgs) + 1,
                                "num_layers": len(layer_avgs)
                            }
                            all_layer_data.append(layer_entry)
                            family_layer_data.append(layer_entry)
                        
                        # Representation analysis
                        if "similarity_analysis" in sample["representation_analysis"]:
                            sim_data = sample["representation_analysis"]["similarity_analysis"]
                            repr_entry = {
                                "family": family,
                                "model": model_name,
                                "category": category,
                                "avg_similarity": np.mean(sim_data),
                                "final_layer_similarity": sim_data[-1] if sim_data else np.nan
                            }
                            all_repr_data.append(repr_entry)
                            family_repr_data.append(repr_entry)
            
            # Family-level statistics
            if family_position_data:
                summary["by_family"][family] = {
                    "position_1_mean": np.mean([d["position_1"] for d in family_position_data if not np.isnan(d["position_1"])]),
                    "position_1_std": np.std([d["position_1"] for d in family_position_data if not np.isnan(d["position_1"])]),
                    "avg_peak_layer": np.mean([d["peak_layer"] for d in family_layer_data]),
                    "avg_similarity": np.mean([d["avg_similarity"] for d in family_repr_data if not np.isnan(d["avg_similarity"])]),
                    "sample_count": len(family_position_data)
                }
        
        # Text category statistics
        category_groups = {}
        for entry in all_position_data:
            cat = entry["category"]
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(entry)
        
        for category, entries in category_groups.items():
            summary["by_text_category"][category] = {
                "position_1_mean": np.mean([e["position_1"] for e in entries if not np.isnan(e["position_1"])]),
                "position_1_std": np.std([e["position_1"] for e in entries if not np.isnan(e["position_1"])]),
                "sample_count": len(entries),
                "families_tested": len(set(e["family"] for e in entries))
            }
        
        # Overall position analysis
        pos_1_values = [d["position_1"] for d in all_position_data if not np.isnan(d["position_1"])]
        pos_2_values = [d["position_2"] for d in all_position_data if not np.isnan(d["position_2"])]
        pos_3_values = [d["position_3"] for d in all_position_data if not np.isnan(d["position_3"])]
        pos_4_values = [d["position_4"] for d in all_position_data if not np.isnan(d["position_4"])]
        
        summary["position_analysis"] = {
            "position_1": {"mean": np.mean(pos_1_values), "std": np.std(pos_1_values), "count": len(pos_1_values)},
            "position_2": {"mean": np.mean(pos_2_values), "std": np.std(pos_2_values), "count": len(pos_2_values)},
            "position_3": {"mean": np.mean(pos_3_values), "std": np.std(pos_3_values), "count": len(pos_3_values)},
            "position_4": {"mean": np.mean(pos_4_values), "std": np.std(pos_4_values), "count": len(pos_4_values)},
        }
        
        return summary
    
    def _perform_statistical_analysis(self, processed_results: Dict) -> Dict:
        """
        Perform rigorous statistical tests on the collected data
        """
        logger.info("Performing statistical analysis...")
        
        if "organized_results" not in processed_results:
            return {"error": "no_data_for_statistical_analysis"}
        
        stats_results = {
            "family_comparisons": {},
            "text_category_comparisons": {},
            "position_dominance_tests": {},
            "effect_sizes": {}
        }
        
        # Collect data for statistical tests
        position_data_by_family = {}
        position_data_by_category = {}
        all_position_1_data = []
        all_position_2_data = []
        all_position_3_data = []
        all_position_4_data = []
        
        # Extract position data organized by different groupings
        for family, models in processed_results["organized_results"].items():
            position_data_by_family[family] = []
            
            for model_name, categories in models.items():
                for category, samples in categories.items():
                    if category not in position_data_by_category:
                        position_data_by_category[category] = []
                    
                    for sample in samples:
                        if "position_analysis" in sample["attention_analysis"]:
                            pos_data = np.array(sample["attention_analysis"]["position_analysis"])
                            avg_by_position = pos_data.mean(axis=0)
                            
                            if len(avg_by_position) >= 4:
                                pos_values = {
                                    "family": family,
                                    "model": model_name,
                                    "category": category,
                                    "position_1": avg_by_position[0],
                                    "position_2": avg_by_position[1],
                                    "position_3": avg_by_position[2],
                                    "position_4": avg_by_position[3],
                                }
                                
                                position_data_by_family[family].append(pos_values)
                                position_data_by_category[category].append(pos_values)
                                
                                all_position_1_data.append(avg_by_position[0])
                                all_position_2_data.append(avg_by_position[1])
                                all_position_3_data.append(avg_by_position[2])
                                all_position_4_data.append(avg_by_position[3])
        
        # Test 1: Position 1 dominance (one-sample t-tests and effect sizes)
        if all_position_1_data and all_position_2_data:
            # Test if position 1 is significantly greater than positions 2-4
            pos_1_vs_2_ttest = stats.ttest_rel(all_position_1_data, all_position_2_data)
            pos_1_vs_3_ttest = stats.ttest_rel(all_position_1_data, all_position_3_data)
            pos_1_vs_4_ttest = stats.ttest_rel(all_position_1_data, all_position_4_data)
            
            # Effect sizes (Cohen's d)
            def cohens_d(x, y):
                nx, ny = len(x), len(y)
                dof = nx + ny - 2
                pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
                return (np.mean(x) - np.mean(y)) / pooled_std
            
            stats_results["position_dominance_tests"] = {
                "position_1_vs_2": {
                    "t_statistic": pos_1_vs_2_ttest.statistic,
                    "p_value": pos_1_vs_2_ttest.pvalue,
                    "significant": pos_1_vs_2_ttest.pvalue < self.config.alpha_level,
                    "cohens_d": cohens_d(all_position_1_data, all_position_2_data)
                },
                "position_1_vs_3": {
                    "t_statistic": pos_1_vs_3_ttest.statistic,
                    "p_value": pos_1_vs_3_ttest.pvalue,
                    "significant": pos_1_vs_3_ttest.pvalue < self.config.alpha_level,
                    "cohens_d": cohens_d(all_position_1_data, all_position_3_data)
                },
                "position_1_vs_4": {
                    "t_statistic": pos_1_vs_4_ttest.statistic,
                    "p_value": pos_1_vs_4_ttest.pvalue,
                    "significant": pos_1_vs_4_ttest.pvalue < self.config.alpha_level,
                    "cohens_d": cohens_d(all_position_1_data, all_position_4_data)
                }
            }
        
        # Test 2: Family comparisons (ANOVA if multiple families)
        families_with_data = [f for f in position_data_by_family.keys() if len(position_data_by_family[f]) > 1]
        
        if len(families_with_data) >= 2:
            family_pos1_groups = []
            family_labels = []
            
            for family in families_with_data:
                family_pos1_values = [d["position_1"] for d in position_data_by_family[family]]
                family_pos1_groups.append(family_pos1_values)
                family_labels.extend([family] * len(family_pos1_values))
            
            if len(family_pos1_groups) >= 2 and all(len(group) > 0 for group in family_pos1_groups):
                try:
                    f_stat, p_val = stats.f_oneway(*family_pos1_groups)
                    stats_results["family_comparisons"]["anova"] = {
                        "f_statistic": f_stat,
                        "p_value": p_val,
                        "significant": p_val < self.config.alpha_level,
                        "families_tested": families_with_data
                    }
                    
                    # Post-hoc pairwise comparisons if ANOVA is significant
                    if p_val < self.config.alpha_level:
                        pairwise_results = {}
                        for i, family1 in enumerate(families_with_data):
                            for j, family2 in enumerate(families_with_data[i+1:], i+1):
                                group1 = [d["position_1"] for d in position_data_by_family[family1]]
                                group2 = [d["position_1"] for d in position_data_by_family[family2]]
                                
                                if len(group1) > 1 and len(group2) > 1:
                                    t_stat, p_val_pair = stats.ttest_ind(group1, group2)
                                    pairwise_results[f"{family1}_vs_{family2}"] = {
                                        "t_statistic": t_stat,
                                        "p_value": p_val_pair,
                                        "significant": p_val_pair < self.config.alpha_level,
                                        "cohens_d": cohens_d(group1, group2)
                                    }
                        
                        stats_results["family_comparisons"]["pairwise"] = pairwise_results
                
                except Exception as e:
                    logger.warning(f"Family comparison failed: {e}")
        
        # Test 3: Text category comparisons
        categories_with_data = [c for c in position_data_by_category.keys() if len(position_data_by_category[c]) > 1]
        
        if len(categories_with_data) >= 2:
            category_pos1_groups = []
            
            for category in categories_with_data:
                cat_pos1_values = [d["position_1"] for d in position_data_by_category[category]]
                category_pos1_groups.append(cat_pos1_values)
            
            if len(category_pos1_groups) >= 2 and all(len(group) > 0 for group in category_pos1_groups):
                try:
                    f_stat, p_val = stats.f_oneway(*category_pos1_groups)
                    stats_results["text_category_comparisons"]["anova"] = {
                        "f_statistic": f_stat,
                        "p_value": p_val,
                        "significant": p_val < self.config.alpha_level,
                        "categories_tested": categories_with_data
                    }
                    
                    # Pairwise comparisons for categories
                    if p_val < self.config.alpha_level:
                        pairwise_results = {}
                        for i, cat1 in enumerate(categories_with_data):
                            for j, cat2 in enumerate(categories_with_data[i+1:], i+1):
                                group1 = [d["position_1"] for d in position_data_by_category[cat1]]
                                group2 = [d["position_1"] for d in position_data_by_category[cat2]]
                                
                                if len(group1) > 1 and len(group2) > 1:
                                    t_stat, p_val_pair = stats.ttest_ind(group1, group2)
                                    pairwise_results[f"{cat1}_vs_{cat2}"] = {
                                        "t_statistic": t_stat,
                                        "p_value": p_val_pair,
                                        "significant": p_val_pair < self.config.alpha_level,
                                        "cohens_d": cohens_d(group1, group2)
                                    }
                        
                        stats_results["text_category_comparisons"]["pairwise"] = pairwise_results
                
                except Exception as e:
                    logger.warning(f"Category comparison failed: {e}")
        
        return stats_results
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other objects for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def create_comprehensive_visualizations(self, processed_results: Dict):
        """
        Create comprehensive visualizations comparing across architectures
        """
        if "organized_results" not in processed_results:
            logger.warning("No organized results for visualization")
            return
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Architecture Attention Sink Analysis', fontsize=16, fontweight='bold')
        
        # Collect data for plotting
        plot_data = []
        for family, models in processed_results["organized_results"].items():
            for model_name, categories in models.items():
                for category, samples in categories.items():
                    for sample in samples:
                        if "position_analysis" in sample["attention_analysis"]:
                            pos_data = np.array(sample["attention_analysis"]["position_analysis"])
                            avg_by_position = pos_data.mean(axis=0)
                            
                            plot_data.append({
                                "Family": family,
                                "Model": model_name.split('/')[-1],  # Short name
                                "Category": category,
                                "Position_1": avg_by_position[0] if len(avg_by_position) > 0 else 0,
                                "Position_2": avg_by_position[1] if len(avg_by_position) > 1 else 0,
                                "Position_3": avg_by_position[2] if len(avg_by_position) > 2 else 0,
                                "Position_4": avg_by_position[3] if len(avg_by_position) > 3 else 0,
                            })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            logger.warning("No data for visualization")
            return
        
        # Plot 1: Position 1 attention by family
        sns.boxplot(data=df, x="Family", y="Position_1", ax=axes[0,0])
        axes[0,0].set_title("Position 1 Attention by Model Family")
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Position 1 attention by text category
        sns.boxplot(data=df, x="Category", y="Position_1", ax=axes[0,1])
        axes[0,1].set_title("Position 1 Attention by Text Category")
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Position comparison
        position_cols = ["Position_1", "Position_2", "Position_3", "Position_4"]
        position_means = df[position_cols].mean()
        axes[0,2].bar(range(1, 5), position_means, alpha=0.7)
        axes[0,2].set_title("Average Attention by Sink Position")
        axes[0,2].set_xlabel("Sink Position")
        axes[0,2].set_ylabel("Average Attention")
        axes[0,2].set_xticks(range(1, 5))
        
        # Plot 4: Family vs Category heatmap
        heatmap_data = df.groupby(["Family", "Category"])["Position_1"].mean().unstack()
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=axes[1,0], cmap='viridis')
            axes[1,0].set_title("Position 1 Attention: Family vs Category")
        
        # Plot 5: Position ratios
        df['P1_to_others_ratio'] = df['Position_1'] / (df['Position_2'] + df['Position_3'] + df['Position_4'] + 1e-6)
        sns.boxplot(data=df, x="Family", y="P1_to_others_ratio", ax=axes[1,1])
        axes[1,1].set_title("Position 1 Dominance Ratio by Family")
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel("P1 / (P2+P3+P4) Ratio")
        
        # Plot 6: Statistical significance indicators
        if "statistical_analysis" in processed_results:
            stats_data = processed_results["statistical_analysis"]
            
            # Create significance summary
            sig_text = "Statistical Significance Summary:\n\n"
            
            if "position_dominance_tests" in stats_data:
                for test_name, result in stats_data["position_dominance_tests"].items():
                    sig_marker = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                    sig_text += f"{test_name}: {sig_marker} (d={result['cohens_d']:.2f})\n"
            
            if "family_comparisons" in stats_data and "anova" in stats_data["family_comparisons"]:
                anova_result = stats_data["family_comparisons"]["anova"]
                sig_marker = "***" if anova_result["p_value"] < 0.001 else "**" if anova_result["p_value"] < 0.01 else "*" if anova_result["p_value"] < 0.05 else "ns"
                sig_text += f"\nFamily differences: {sig_marker}"
            
            if "text_category_comparisons" in stats_data and "anova" in stats_data["text_category_comparisons"]:
                anova_result = stats_data["text_category_comparisons"]["anova"]
                sig_marker = "***" if anova_result["p_value"] < 0.001 else "**" if anova_result["p_value"] < 0.01 else "*" if anova_result["p_value"] < 0.05 else "ns"
                sig_text += f"\nCategory differences: {sig_marker}"
            
            axes[1,2].text(0.1, 0.9, sig_text, transform=axes[1,2].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1,2].set_title("Statistical Test Results")
            axes[1,2].axis('off')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_file = self.output_dir / "comprehensive_multi_architecture_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Comprehensive plot saved: {plot_file}")
        
        plt.show()
    
    def print_statistical_summary(self, processed_results: Dict):
        """
        Print a comprehensive statistical summary
        """
        if "statistical_analysis" not in processed_results:
            print("No statistical analysis available")
            return
        
        stats_data = processed_results["statistical_analysis"]
        
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Position dominance tests
        if "position_dominance_tests" in stats_data:
            print("\n🎯 POSITION 1 DOMINANCE TESTS:")
            print("-" * 40)
            
            for test_name, result in stats_data["position_dominance_tests"].items():
                significance = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                effect_size = "large" if abs(result["cohens_d"]) > 0.8 else "medium" if abs(result["cohens_d"]) > 0.5 else "small"
                
                print(f"{test_name}:")
                print(f"  t = {result['t_statistic']:.3f}, p = {result['p_value']:.3e} {significance}")
                print(f"  Cohen's d = {result['cohens_d']:.3f} ({effect_size} effect)")
                print()
        
        # Family comparisons
        if "family_comparisons" in stats_data:
            print("🏗️ MODEL FAMILY COMPARISONS:")
            print("-" * 40)
            
            if "anova" in stats_data["family_comparisons"]:
                anova = stats_data["family_comparisons"]["anova"]
                significance = "***" if anova["p_value"] < 0.001 else "**" if anova["p_value"] < 0.01 else "*" if anova["p_value"] < 0.05 else "ns"
                
                print(f"One-way ANOVA across families:")
                print(f"  F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.3e} {significance}")
                print(f"  Families tested: {', '.join(anova['families_tested'])}")
                
                if "pairwise" in stats_data["family_comparisons"]:
                    print(f"\n  Pairwise comparisons:")
                    for comparison, result in stats_data["family_comparisons"]["pairwise"].items():
                        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                        print(f"    {comparison}: p = {result['p_value']:.3e} {sig} (d = {result['cohens_d']:.3f})")
                print()
        
        # Text category comparisons
        if "text_category_comparisons" in stats_data:
            print("📝 TEXT CATEGORY COMPARISONS:")
            print("-" * 40)
            
            if "anova" in stats_data["text_category_comparisons"]:
                anova = stats_data["text_category_comparisons"]["anova"]
                significance = "***" if anova["p_value"] < 0.001 else "**" if anova["p_value"] < 0.01 else "*" if anova["p_value"] < 0.05 else "ns"
                
                print(f"One-way ANOVA across text categories:")
                print(f"  F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.3e} {significance}")
                print(f"  Categories tested: {', '.join(anova['categories_tested'])}")
                
                if "pairwise" in stats_data["text_category_comparisons"]:
                    print(f"\n  Pairwise comparisons:")
                    for comparison, result in stats_data["text_category_comparisons"]["pairwise"].items():
                        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                        print(f"    {comparison}: p = {result['p_value']:.3e} {sig} (d = {result['cohens_d']:.3f})")
        
        print(f"\n{'='*80}")
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("Effect sizes (Cohen's d): small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8")
        print(f"{'='*80}\n")

def main():
    """
    Main function for expanded multi-architecture analysis
    """
    # Configuration
    config = ExpandedAnalysisConfig(
        texts_per_category=5,  # Multiple samples per category
        perform_statistical_tests=True,
        save_plots=True,
        save_data=True
    )
    
    print("🚀 EXPANDED MULTI-ARCHITECTURE ATTENTION SINK ANALYSIS")
    print("=" * 70)
    print(f"Models to test: {len(config.models_to_test)}")
    print(f"Samples per text category: {config.texts_per_category}")
    print(f"Statistical testing: {config.perform_statistical_tests}")
    print("=" * 70)
    
    try:
        # Initialize analyzer
        analyzer = MultiArchitectureAttentionAnalyzer(config)
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        # Create visualizations
        analyzer.create_comprehensive_visualizations(results)
        
        # Print statistical summary
        analyzer.print_statistical_summary(results)
        
        # Print final summary
        if "summary_statistics" in results:
            summary = results["summary_statistics"]
            
            print(f"\n{'='*80}")
            print("FINAL SUMMARY")
            print(f"{'='*80}")
            
            if "position_analysis" in summary:
                pos_analysis = summary["position_analysis"]
                print("🎯 POSITION ANALYSIS SUMMARY:")
                for i in range(1, 5):
                    pos_key = f"position_{i}"
                    if pos_key in pos_analysis:
                        pos_data = pos_analysis[pos_key]
                        print(f"  Position {i}: μ = {pos_data['mean']:.4f} (σ = {pos_data['std']:.4f}, n = {pos_data['count']})")
            
            if "by_family" in summary:
                print(f"\n🏗️ FAMILY COMPARISON:")
                for family, family_data in summary["by_family"].items():
                    print(f"  {family}: P1 = {family_data['position_1_mean']:.4f} (±{family_data['position_1_std']:.4f})")
            
            if "by_text_category" in summary:
                print(f"\n📝 TEXT CATEGORY COMPARISON:")
                for category, cat_data in summary["by_text_category"].items():
                    print(f"  {category}: P1 = {cat_data['position_1_mean']:.4f} (±{cat_data['position_1_std']:.4f})")
            
            print(f"\n📁 Results saved to: {analyzer.output_dir}")
            print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()
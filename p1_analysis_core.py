"""
P1 Causal Analysis - Core Framework (Script 1 of 2)
===================================================

This script contains the core analysis framework including:
- Model management and loading
- Dataset preparation and management
- Intervention engine
- Evaluation metrics
- Individual model analysis

Companion script: p1_analysis_runner.py

Author: Research Team
Date: June 2025
"""


import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
from datetime import datetime
import warnings
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, asdict
import gc
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class P1AnalysisConfig:
    """Configuration for P1 analysis experiments"""
    
    # Model configurations
    models_to_test: List[Dict] = None
    
    # Intervention configurations
    intervention_types: List[str] = None
    target_layers: Dict[str, List[int]] = None
    noise_levels: List[float] = None
    
    # Evaluation configurations
    include_downstream_tasks: bool = True
    include_extensive_probing: bool = True
    probing_dataset_size: int = 500
    perplexity_samples_per_category: int = 5
    
    # Computational settings
    use_quantization: bool = True
    max_sequence_length: int = 512
    device_strategy: str = "auto"
    
    # Output settings
    output_dir: str = "./p1_comprehensive_analysis"
    save_intermediate_results: bool = True
    
    # Statistical settings
    statistical_significance_level: float = 0.05
    effect_size_threshold: float = 0.1
    random_seed: int = 42
    cv_folds: int = 5
    
    def __post_init__(self):
        # Set a random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        if self.models_to_test is None:
            self.models_to_test = [
                {"name": "gpt2", "family": "gpt2", "size": "124M", "priority": "high"},
                {"name": "gpt2-medium", "family": "gpt2", "size": "355M", "priority": "high"},
                {"name": "meta-llama/Llama-2-7b-hf", "family": "llama", "size": "7B", "priority": "high"},
                {"name": "mistralai/Mistral-7B-v0.1", "family": "mistral", "size": "7B", "priority": "high"},
                {"name": "deepseek-ai/deepseek-coder-6.7b-instruct", "family": "deepseek", "size": "6.7B", "priority": "medium"},
            ]
        
        if self.intervention_types is None:
            self.intervention_types = ["ablation", "mean_ablation", "noise_injection", "random_replacement"]
        
        if self.target_layers is None:
            # Defines layers as percentages of model depth to handle different model sizes
            layer_percentages = [0.0, 0.25, 0.5, 0.75, 0.95] # 0%, 25%, 50%, 75%, ~final
            self.target_layers = {
                "gpt2": [int(p * 11) for p in layer_percentages],          # 12 layers (0-11)
                "gpt2-medium": [int(p * 23) for p in layer_percentages],   # 24 layers (0-23)
                "llama": [int(p * 31) for p in layer_percentages],         # 32 layers (0-31)
                "mistral": [int(p * 31) for p in layer_percentages],       # 32 layers (0-31)
                "deepseek": [int(p * 29) for p in layer_percentages],      # 30 layers (0-29)
            }
        
        if self.noise_levels is None:
            self.noise_levels = [0.1, 0.3, 0.5]

def save_model_results(results: Dict, output_dir: Path, model_name: str):
    """Saves the results for a single model to a JSON file."""
    model_name_sanitized = model_name.replace("/", "_")
    results_file = output_dir / "data" / f"results_{model_name_sanitized}.json"
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_for_json(item) for item in obj]
        return obj

    try:
        with open(results_file, 'w') as f:
            json.dump(convert_for_json(results), f, indent=2)
        logger.info(f"✅ Intermediate results for {model_name} saved to {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save intermediate results for {model_name}: {e}")

class ModelManager:
    """Manages loading and unloading of different model architectures"""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_config = None
        self.device = self._setup_device()
        self._mean_embedding = None
        
    def _setup_device(self):
        """Setup device strategy"""
        if self.config.device_strategy == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device_strategy)
    
    def load_model(self, model_config: Dict) -> Tuple[torch.nn.Module, any]:
        """Load model with optimized settings"""
        model_name = model_config["name"]
        
        logger.info(f"Loading {model_name} ({model_config['family']} family)")
        
        self.unload_current_model()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
            if tokenizer.pad_token == '[PAD]':
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            model_kwargs = {"output_attentions": True, "output_hidden_states": True, "trust_remote_code": True}
            
            if (self.config.use_quantization and "7b" in model_config.get("size", "").lower() and torch.cuda.is_available()):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif "7b" in model_config.get("size", "").lower() and torch.cuda.is_available():
                model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto", "low_cpu_mem_usage": True})
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            if not hasattr(model, 'hf_device_map') or not model.hf_device_map:
                model = model.to(self.device)
            
            model.eval()
            
            if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
                model.resize_token_embeddings(len(tokenizer))
            
            self.current_model, self.current_tokenizer, self.current_model_config = model, tokenizer, model_config
            
            with torch.no_grad():
                embeddings = model.get_input_embeddings().weight.data.float()
                self._mean_embedding = embeddings.mean(dim=0)
            
            logger.info(f"Successfully loaded {model_name} with {self.get_model_layers()} layers")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    def unload_current_model(self):
        """Properly unload current model to free memory"""
        if self.current_model is not None:
            del self.current_model, self.current_tokenizer
            self.current_model = self.current_tokenizer = self.current_model_config = self._mean_embedding = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    
    def get_model_layers(self) -> int:
        """Get number of layers in current model"""
        if not self.current_model: return 0
        config = self.current_model.config
        return getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 24)))
    
    def get_target_layer(self, model, layer_idx: int):
        """Get the target layer for intervention based on model architecture"""
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): return model.transformer.h[layer_idx]
        if hasattr(model, 'model') and hasattr(model.model, 'layers'): return model.model.layers[layer_idx]
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'): return model.model.decoder.layers[layer_idx]
        raise ValueError(f"Unknown model architecture for intervention: {type(model)}")


class DatasetManager:
    """Manages datasets for evaluation and probing"""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config
        self.perplexity_dataset = self._create_perplexity_dataset()
        self.probing_dataset = self._create_probing_dataset() if config.include_extensive_probing else None
        self.downstream_datasets = self._create_downstream_datasets() if config.include_downstream_tasks else None
    
    def _create_perplexity_dataset(self) -> List[Dict]:
        """Enhanced perplexity evaluation dataset"""
        texts = []
        samples_per_cat = self.config.perplexity_samples_per_category
        
        # Narrative texts
        narrative_texts = [
            "The ancient castle stood majestically on the hilltop, its weathered stones telling tales of centuries past. Knights once roamed these halls, their armor clanking as they prepared for battle.",
            "Sarah walked through the misty forest, her footsteps echoing softly on the damp leaves. The moonlight filtered through the canopy, creating dancing shadows that seemed alive.",
            "The detective examined the crime scene carefully, noting every detail that might provide a clue to the mysterious disappearance of the wealthy businessman.",
            "In the small village nestled between rolling hills, life moved at a gentle pace where everyone knew their neighbors and stories were passed down through generations.",
            "The spaceship descended through the alien atmosphere, its hull glowing from the intense heat as the crew prepared for humanity's first contact with an extraterrestrial civilization.",
            "Margaret discovered the old diary hidden beneath the floorboards of her grandmother's attic, its yellowed pages containing secrets that would change everything she thought she knew about her family.",
            "The lighthouse keeper watched the storm approach from his tower, knowing that ships would need his beacon to navigate safely through the treacherous rocks below."
        ]
        
        # Technical texts
        technical_texts = [
            "Machine learning algorithms utilize mathematical optimization techniques to minimize loss functions. Gradient descent iteratively adjusts model parameters to improve prediction accuracy.",
            "Neural networks consist of interconnected layers of artificial neurons that process information through weighted connections. Backpropagation enables efficient training by computing gradients.",
            "Transformer architectures employ self-attention mechanisms to capture long-range dependencies in sequential data, revolutionizing natural language processing and computer vision tasks.",
            "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that are intractable for classical computers.",
            "Cryptographic protocols ensure secure communication through mathematical algorithms that make it computationally infeasible for unauthorized parties to decrypt sensitive information.",
            "Distributed systems architecture requires careful consideration of consistency, availability, and partition tolerance according to the CAP theorem, with different trade-offs for different applications.",
            "Advanced materials science focuses on engineered nanostructures that exhibit novel properties through precise control of atomic arrangements and interfacial phenomena."
        ]
        
        # Dialogue texts
        dialogue_texts = [
            "\"Hello, how are you today?\" she asked with a warm smile. \"I'm doing well, thank you for asking,\" he replied, adjusting his glasses nervously.",
            "\"Can you help me with this problem?\" the student inquired. \"Of course, let's work through it step by step,\" the teacher responded encouragingly.",
            "\"What time does the meeting start?\" Sarah asked her colleague. \"It's scheduled for 3 PM in the conference room,\" he answered while checking his calendar.",
            "\"I think we should reconsider our approach,\" the manager suggested during the team discussion. \"You're right, let's explore alternative solutions,\" the team lead agreed.",
            "\"Have you seen the latest research on attention mechanisms?\" the researcher asked. \"Yes, it's fascinating how they've improved model performance,\" her colleague responded enthusiastically.",
            "\"Could you pass me that wrench?\" the mechanic called out from under the car. \"Which one? The socket wrench or the adjustable?\" his apprentice asked, looking at the tool rack.",
            "\"The weather forecast looks promising for our picnic tomorrow,\" Mom said while packing sandwiches. \"I hope it doesn't rain like last time,\" replied her daughter with concern."
        ]
        
        # Code texts
        code_texts = [
            "def calculate_attention(query, key, value):\n    scores = torch.matmul(query, key.transpose(-2, -1))\n    attention_weights = F.softmax(scores, dim=-1)\n    return torch.matmul(attention_weights, value)",
            "for i in range(len(data)):\n    if data[i] is not None:\n        result = process_item(data[i])\n        output.append(result)\n    else:\n        output.append(default_value)",
            "class TransformerLayer(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.attention = MultiHeadAttention(d_model, n_heads)\n        self.feedforward = FeedForward(d_model)",
            "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
            "SELECT customers.name, orders.total\nFROM customers\nJOIN orders ON customers.id = orders.customer_id\nWHERE orders.date >= '2024-01-01'\nORDER BY orders.total DESC;",
            "import numpy as np\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        ]
        
        # Short texts
        short_texts = [
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet exactly once.",
            "Time flies when you're having fun. Every moment counts in life's precious journey.",
            "Knowledge is power, but wisdom is knowing how to use it effectively and responsibly.",
            "Innovation drives progress. Creative solutions emerge from collaborative thinking and persistent effort.",
            "Success requires dedication, hard work, and the ability to learn from both failures and achievements.",
            "Beautiful sunsets remind us that endings can be magnificent too. Nature's artistry never ceases to amaze.",
            "Laughter is the best medicine for the soul. It connects people across all boundaries and differences."
        ]
        
        # Combine all categories, taking specified number of samples
        for category, category_texts in [
            ("narrative", narrative_texts),
            ("technical", technical_texts), 
            ("dialogue", dialogue_texts),
            ("code", code_texts),
            ("short", short_texts)
        ]:
            for text in category_texts[:samples_per_cat]:
                texts.append({"category": category, "text": text})
        
        logger.info(f"Created perplexity dataset with {len(texts)} samples")
        return texts
    
    def _create_probing_dataset(self) -> Dict[str, List[Dict]]:
        """Create extensive probing dataset"""
        logger.info(f"Creating extensive probing dataset with {self.config.probing_dataset_size} samples")
        
        categories = ["narrative", "technical", "dialogue", "code", "short"]
        samples_per_category = self.config.probing_dataset_size // len(categories)
        
        probing_data = {
            "category_classification": []
        }
        
        # Generate samples for each category
        base_texts = {
            "narrative": [
                "The story begins with a character who faces an unexpected challenge.",
                "Once upon a time, in a land far away, there lived a brave hero.",
                "The novel explores themes of love, loss, and redemption through its characters.",
                "She walked through the ancient forest, feeling the weight of destiny upon her.",
                "The tale unfolds with mystery and adventure at every turn."
            ],
            "technical": [
                "The algorithm processes data using advanced computational methods.",
                "Technical specifications require precise implementation and testing.",
                "System architecture must consider scalability and performance requirements.",
                "The methodology involves statistical analysis and machine learning techniques.",
                "Engineering solutions demand rigorous mathematical modeling and validation."
            ],
            "dialogue": [
                "\"How are you feeling today?\" she asked with genuine concern.",
                "\"I believe we can solve this problem together,\" he said confidently.",
                "\"What do you think about the proposal?\" the manager inquired.",
                "\"Let's schedule a meeting to discuss this further,\" she suggested.",
                "\"Thank you for your help with this project,\" he replied gratefully."
            ],
            "code": [
                "def process_data(input_list): return [item * 2 for item in input_list]",
                "if condition: execute_function() else: handle_error()",
                "class DataProcessor: def __init__(self): self.data = []",
                "import pandas as pd; df = pd.read_csv('data.csv')",
                "try: result = compute_value() except Exception as e: print(e)"
            ],
            "short": [
                "Quick facts about the topic. Brief summary follows.",
                "Simple statement. Clear and concise information.",
                "Short text sample. Easy to understand content.",
                "Brief note. Minimal but informative.",
                "Concise message. Direct communication style."
            ]
        }
        
        # Generate synthetic samples
        for category in categories:
            base_category_texts = base_texts[category]
            for i in range(samples_per_category):
                # Create variations of base texts
                base_text = base_category_texts[i % len(base_category_texts)]
                variation_text = f"{base_text} Sample variation {i+1} for comprehensive analysis."
                
                probing_data["category_classification"].append({
                    "text": variation_text,
                    "label": category,
                    "sample_id": f"{category}_{i:04d}"
                })
        
        logger.info(f"Created probing dataset with {len(probing_data['category_classification'])} samples")
        return probing_data
    
    def _create_downstream_datasets(self) -> Dict[str, List[Dict]]:
        """Create downstream task datasets"""
        logger.info("Creating downstream task datasets")
        
        downstream_data = {
            "passkey_retrieval": self._create_passkey_dataset(),
            "document_qa": self._create_document_qa_dataset()
        }
        
        return downstream_data
    
    def _create_passkey_dataset(self) -> List[Dict]:
        """Create passkey retrieval dataset"""
        passkey_samples = []
        
        # Generate passkey retrieval tasks
        for i in range(20):
            passkey = f"PASS{i:03d}"
            
            # Create distractor text of varying lengths
            distractor_words = [
                "computer", "science", "research", "analysis", "method", "system", "process", "data",
                "information", "technology", "algorithm", "function", "variable", "parameter",
                "optimization", "evaluation", "assessment", "measurement", "calculation", "computation"
            ]
            
            distractor_length = random.choice([30, 50, 80])
            distractor_text = []
            
            for j in range(distractor_length):
                word = random.choice(distractor_words)
                distractor_text.append(f"{word}{j}")
            
            # Insert passkey at random position
            insert_pos = random.randint(5, len(distractor_text) - 5)
            passkey_sentence = f"The passkey is {passkey} and should be remembered."
            distractor_text.insert(insert_pos, passkey_sentence)
            
            text = " ".join(distractor_text)
            question = "What is the passkey mentioned in the text?"
            
            passkey_samples.append({
                "text": text,
                "question": question,
                "answer": passkey,
                "passkey_position": insert_pos,
                "text_length": len(distractor_text)
            })
        
        logger.info(f"Created {len(passkey_samples)} passkey retrieval samples")
        return passkey_samples
    
    def _create_document_qa_dataset(self) -> List[Dict]:
        """Create document QA dataset"""
        qa_samples = []
        
        # Generate simple document QA tasks
        documents = [
            {
                "text": "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It revolutionized natural language processing by using self-attention mechanisms instead of recurrent connections. The model achieved state-of-the-art performance on machine translation tasks and became the foundation for modern language models.",
                "questions": [
                    ("When was the Transformer architecture introduced?", "2017"),
                    ("Who introduced the Transformer architecture?", "Vaswani"),
                    ("What mechanism does the Transformer use?", "self-attention"),
                    ("What tasks did it achieve state-of-the-art performance on?", "machine translation")
                ]
            },
            {
                "text": "GPT-2 is a large-scale language model developed by OpenAI in 2019. It has 1.5 billion parameters and was trained on a diverse dataset of internet text. The model demonstrated impressive text generation capabilities and raised concerns about potential misuse, leading to a staged release approach.",
                "questions": [
                    ("How many parameters does GPT-2 have?", "1.5 billion"),
                    ("Who developed GPT-2?", "OpenAI"),
                    ("What was GPT-2 trained on?", "internet text"),
                    ("When was GPT-2 developed?", "2019")
                ]
            },
            {
                "text": "Attention mechanisms in neural networks allow models to focus on different parts of the input when processing information. The self-attention mechanism computes attention weights for each position in a sequence relative to all other positions. This enables the model to capture long-range dependencies more effectively than traditional recurrent architectures.",
                "questions": [
                    ("What do attention mechanisms allow models to do?", "focus on different parts"),
                    ("What does self-attention compute?", "attention weights"),
                    ("What can models capture more effectively?", "long-range dependencies"),
                    ("What type of architectures are compared?", "recurrent")
                ]
            }
        ]
        
        for doc in documents:
            for question, answer in doc["questions"]:
                qa_samples.append({
                    "text": doc["text"],
                    "question": question,
                    "answer": answer,
                    "doc_length": len(doc["text"].split())
                })
        
        logger.info(f"Created {len(qa_samples)} document QA samples")
        return qa_samples


class InterventionEngine:
    """Handles all intervention operations"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def create_intervention_hook(self, intervention_type: str, magnitude: float = None) -> Callable:
        """Create intervention hooks with enhanced options"""
        
        def intervention_hook(module, input, output):
            if output[0].size(1) > 0:  # Ensure we have tokens
                
                if intervention_type == "ablation":
                    # Zero out P1
                    output[0][:, 0, :] = 0.0
                
                elif intervention_type == "mean_ablation":
                    # Replace P1 with pre-computed mean embedding
                    if self.model_manager._mean_embedding is not None:
                        # Move mean_embedding to the same device as the layer's output on-the-fly
                        mean_emb_device = self.model_manager._mean_embedding.to(output[0].device)
                        output[0][:, 0, :] = mean_emb_device
                
                elif intervention_type == "noise_injection":
                    noise_std = magnitude if magnitude is not None else 0.1
                    noise = torch.randn_like(output[0][:, 0, :]) * noise_std
                    output[0][:, 0, :] += noise
                
                elif intervention_type == "random_replacement":
                    # Replace with random vector
                    output[0][:, 0, :] = torch.randn_like(output[0][:, 0, :])
            
            return output
        
        return intervention_hook
    
    def apply_intervention(self, model, intervention_type: str, 
                          layer_idx: int, magnitude: float = None) -> Callable:
        """Apply intervention and return cleanup function"""
        
        # Create and register hook
        hook_fn = self.create_intervention_hook(intervention_type, magnitude)
        target_layer = self.model_manager.get_target_layer(model, layer_idx)
        handle = target_layer.register_forward_hook(hook_fn)
        
        return handle


class EvaluationEngine:
    """Handles all evaluation metrics"""
    
    def __init__(self, config: P1AnalysisConfig):
        self.config = config
    
    # def calculate_perplexity(self, model, tokenizer, text: str) -> float:
    #     """Calculate perplexity with proper padding handling"""
    #     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_sequence_length)
    #     device = next(model.parameters()).device
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
        
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         logits = outputs.logits
    #         shifted_logits = logits[..., :-1, :].contiguous()
    #         labels = inputs["input_ids"][..., 1:].contiguous()
            
    #         loss = F.cross_entropy(
    #             shifted_logits.view(-1, shifted_logits.size(-1)),
    #             labels.view(-1),
    #             ignore_index=tokenizer.pad_token_id,
    #         )
        
    #     return torch.exp(loss).item() if not torch.isnan(loss) else float('inf')
    
    def calculate_perplexity(self, model, tokenizer, text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_sequence_length, padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            shifted_logits = outputs.logits[..., :-1, :].contiguous()
            labels = inputs["input_ids"][..., 1:].contiguous()
            loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)
            return torch.exp(loss).item() if not torch.isnan(loss) else float('inf')
        
    # def evaluate_downstream_task(self, model, tokenizer, sample: Dict, task_name: str) -> Dict:
    #     """Unified function for evaluating downstream tasks."""
    #     text, question, correct_answer = sample["text"], sample["question"], sample["answer"]
    #     prompt = f"Context: {text}\n\nQuestion: {question}\nAnswer:" if task_name == "document_qa" else f"{text}\n\nQ: {question}\nA:"
        
    #     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_sequence_length)
    #     device = next(model.parameters()).device
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
        
    #     try:
    #         with torch.no_grad():
    #             outputs = model.generate(
    #                 **inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id
    #             )
    #             generated_ids = outputs[0][inputs["input_ids"].size(1):]
    #             generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    #             success = correct_answer.lower() in generated.lower()
    #             return {"success": success, "generated": generated, "expected": correct_answer}
    #     except Exception as e:
    #         logger.warning(f"Error in {task_name} evaluation: {e}")
    #         return {"success": False, "generated": "GENERATION_ERROR", "expected": correct_answer}
        
    def evaluate_downstream_task(self, model, tokenizer, sample, task_name):
        prompt = f"Context: {sample['text']}\nQuestion: {sample['question']}\nAnswer:" if task_name == "document_qa" else f"{sample['text']}\nQ: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_sequence_length)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
                return {"success": sample["answer"].lower() in generated.lower(), "generated": generated}
        except Exception as e: 
            return {"success": False, "generated": f"ERROR: {e}"}

    def extract_attention_patterns(self, model, tokenizer, text: str) -> Dict:
        """Extract detailed attention patterns"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=self.config.max_sequence_length)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            
            # P1 attention analysis
            p1_attention_by_layer = []
            attention_entropy_by_layer = []
            
            for layer_idx, layer_attn in enumerate(attentions):
                # Average across heads
                avg_attn = layer_attn[0].mean(dim=0)  # [seq_len, seq_len]
                
                if avg_attn.size(0) > 4:  # Ensure sufficient tokens
                    # P1 attention strength
                    p1_attn = avg_attn[4:, 0].mean().item()
                    p1_attention_by_layer.append(p1_attn)
                    
                    # Attention entropy (measure of attention distribution)
                    attn_dist = avg_attn[4:, :4].sum(dim=0)  # Attention to first 4 tokens
                    entropy = scipy_stats.entropy(attn_dist.cpu().numpy() + 1e-12)
                    attention_entropy_by_layer.append(float(entropy))
                else:
                    p1_attention_by_layer.append(0.0)
                    attention_entropy_by_layer.append(0.0)
            
            # P1 hidden state norms
            p1_norms = [h[0, 0].norm().item() for h in hidden_states]
            
        return {
            "p1_attention_by_layer": p1_attention_by_layer,
            "attention_entropy_by_layer": attention_entropy_by_layer,
            "p1_hidden_norms": p1_norms,
            "sequence_length": inputs["input_ids"].size(1)
        }
    
    def evaluate_passkey_retrieval(self, model, tokenizer, sample: Dict) -> Dict:
        """Evaluate passkey retrieval task"""
        text = sample["text"]
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Simple evaluation: check if model generates the correct passkey
        prompt = f"{text}\n\nQ: {question}\nA:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=self.config.max_sequence_length)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=1.0
                )
                
                # Decode generated text
                generated = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], 
                                           skip_special_tokens=True)
                
                # Check if correct answer is in generated text
                success = correct_answer.lower() in generated.lower()
                
                return {
                    "success": success,
                    "generated": generated,
                    "expected": correct_answer
                }
        except Exception as e:
            logger.warning(f"Error in passkey evaluation: {e}")
            return {
                "success": False,
                "generated": "",
                 "expected": ""
                }

class SingleModelAnalyzer:
    """Orchestrates the full analysis pipeline for a single model."""
    def __init__(self, config: P1AnalysisConfig, model_manager: ModelManager, dataset_manager: DatasetManager):
        self.config, self.model_manager, self.dataset_manager = config, model_manager, dataset_manager
        self.intervention_engine = InterventionEngine(model_manager)
        self.evaluation_engine = EvaluationEngine(config)

    def analyze_model(self, model, tokenizer, model_config):
        model_results = {"model_config": model_config}
        logger.info("--- Measuring Baseline Performance ---"); model_results["baseline_results"] = self._run_baseline_evaluations(model, tokenizer)
        logger.info("--- Running Intervention Experiments ---"); model_results["intervention_results"] = self._run_all_interventions(model, tokenizer, model_config, model_results["baseline_results"])
        if self.config.include_downstream_tasks: logger.info("--- Evaluating Downstream Tasks ---"); model_results["downstream_results"] = self._run_downstream_evaluations(model, tokenizer)
        if self.config.include_extensive_probing: logger.info("--- Performing Probing Analysis ---"); model_results["probing_results"] = self._run_probing(model, tokenizer)
        logger.info("--- Generating Model-Specific Statistical Summary ---"); model_results["statistical_analysis"] = self._summarize_model_stats(model_results)
        return model_results

    def _run_baseline_evaluations(self, model, tokenizer):
        results = {"perplexity_by_category": {}}
        for sample in tqdm(self.dataset_manager.perplexity_dataset, desc="Baseline PPL"):
            category = sample["category"]
            if category not in results["perplexity_by_category"]: results["perplexity_by_category"][category] = []
            results["perplexity_by_category"][category].append(self.evaluation_engine.calculate_perplexity(model, tokenizer, sample["text"]))
        for cat, ppls in results["perplexity_by_category"].items(): results["perplexity_by_category"][cat] = {"mean": np.mean(ppls), "std": np.std(ppls)}
        return results

    def _run_all_interventions(self, model, tokenizer, model_config, baseline_results):
        all_results = {}
        target_layers = self.config.target_layers.get(model_config["family"], [0, self.model_manager.get_model_layers()//2])
        for int_type in self.config.intervention_types:
            if int_type == "noise_injection":
                for noise_std in self.config.noise_levels:
                    for layer in target_layers: all_results[f"{int_type}_layer_{layer}_std{noise_std}"] = self._run_single_intervention(model, tokenizer, int_type, layer, baseline_results, magnitude=noise_std)
            else:
                for layer in target_layers: all_results[f"{int_type}_layer_{layer}"] = self._run_single_intervention(model, tokenizer, int_type, layer, baseline_results)
        return all_results

    def _run_single_intervention(self, model, tokenizer, int_type, layer, baselines, **kwargs):
        degradations = []
        handle = self.intervention_engine.apply_intervention(model, int_type, layer, **kwargs)
        try:
            for sample in self.dataset_manager.perplexity_dataset:
                baseline_ppl = baselines["perplexity_by_category"][sample["category"]]["mean"]
                int_ppl = self.evaluation_engine.calculate_perplexity(model, tokenizer, sample["text"])
                if baseline_ppl > 0 and baseline_ppl != float('inf'): degradations.append((int_ppl - baseline_ppl) / baseline_ppl)
        finally: handle.remove()
        return {"mean_degradation": np.mean(degradations), "std_degradation": np.std(degradations), "samples": degradations}

    def _run_downstream_evaluations(self, model, tokenizer):
        results = {}
        for task in self.dataset_manager.downstream_datasets: results[task] = self._evaluate_task_with_interventions(model, tokenizer, task)
        return results

    def _evaluate_task_with_interventions(self, model, tokenizer, task_name):
        dataset = self.dataset_manager.downstream_datasets[task_name]
        baseline_scores = [self.evaluation_engine.evaluate_downstream_task(model, tokenizer, s, task_name)['success'] for s in tqdm(dataset, desc=f"Baseline {task_name}")]
        handle = self.intervention_engine.apply_intervention(model, "ablation", 0)
        try:
            int_scores = [self.evaluation_engine.evaluate_downstream_task(model, tokenizer, s, task_name)['success'] for s in tqdm(dataset, desc=f"Ablation {task_name}")]
        finally: handle.remove()
        return {"baseline_accuracy": np.mean(baseline_scores), "intervention_accuracy": np.mean(int_scores), "degradation": np.mean(baseline_scores) - np.mean(int_scores)}

    def _run_probing(self, model, tokenizer):
        results = {"category_classification": {}}; data = self.dataset_manager.probing_dataset["category_classification"]
        if not data: return results
        states = {i:[] for i in range(self.model_manager.get_model_layers() + 1)}; labels = [s['label'] for s in data]
        for sample in tqdm(data, desc="Probing: Extracting States"):
            inputs = tokenizer(sample['text'], return_tensors="pt", max_length=self.config.max_sequence_length, truncation=True)
            inputs = {k: v.to(model.device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                for i, h in enumerate(outputs.hidden_states): states[i].append(h[0,0].cpu().numpy())
        cats = sorted(list(set(labels))); y = [cats.index(l) for l in labels]
        min_class_count = min(np.bincount(y)) if y else 0; cv_folds = min(self.config.cv_folds, min_class_count)
        for layer, X in tqdm(states.items(), desc="Probing: Training Probes"):
            if len(X) > 1 and len(set(y)) > 1 and cv_folds >= 2:
                probe = LogisticRegression(random_state=self.config.random_seed, max_iter=1000)
                scores = cross_val_score(probe, np.array(X), y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=self.config.random_seed))
                results["category_classification"][f"layer_{layer}"] = {"accuracy_mean": np.mean(scores), "accuracy_std": np.std(scores), "num_categories": len(cats)}
        return results
        
    def _summarize_model_stats(self, model_results):
        # (This is a complex function; for this script, we'll keep it simple. The runner script can do more complex cross-model stats)
        summary = {"intervention_summary": {}}
        for key, data in model_results.get("intervention_results", {}).items():
            summary["intervention_summary"][key] = data.get("mean_degradation", 0)
        return summary


# class SingleModelAnalyzer:
#     """Orchestrates the full analysis pipeline for a single model."""
    
#     def __init__(self, config: P1AnalysisConfig, model_manager: ModelManager, dataset_manager: DatasetManager):
#         self.config = config
#         self.model_manager = model_manager
#         self.dataset_manager = dataset_manager
#         self.intervention_engine = InterventionEngine(model_manager)
#         self.evaluation_engine = EvaluationEngine(config)

#     def analyze_model(self, model, tokenizer, model_config: Dict) -> Dict:
#         """Run the complete analysis suite for one model."""
#         model_results = {"model_config": model_config}
        
#         logger.info("--- Measuring Baseline Performance ---")
#         model_results["baseline_results"] = self._run_baseline_evaluations(model, tokenizer)
        
#         logger.info("--- Running Intervention Experiments ---")
#         model_results["intervention_results"] = self._run_all_interventions(model, tokenizer, model_config, model_results["baseline_results"])
        
#         if self.config.include_downstream_tasks:
#             logger.info("--- Evaluating Downstream Tasks ---")
#             model_results["downstream_results"] = self._run_downstream_evaluations(model, tokenizer)
            
#         if self.config.include_extensive_probing:
#             logger.info("--- Performing Probing Analysis ---")
#             model_results["probing_results"] = self._run_probing(model, tokenizer)

#         logger.info("--- Generating Model-Specific Statistical Summary ---")
#         model_results["statistical_analysis"] = self._summarize_model_stats(model_results)

#         return model_results


#     def _run_all_interventions(self, model, tokenizer, model_config, baseline_results) -> Dict:
#         all_results = {}
#         family = model_config["family"]
#         num_layers = self.model_manager.get_model_layers()
        
#         # Get target layers specific to this model's family
#         target_layers_indices = self.config.target_layers.get(family)
#         if target_layers_indices is None:
#             # Fallback if family not in config
#             layer_percentages = [0.0, 0.25, 0.5, 0.75, 0.95]
#             target_layers_indices = list(set([int(p * (num_layers - 1)) for p in layer_percentages]))

#         logger.info(f"Running interventions on layers: {target_layers_indices} for family {family}")

#         for intervention_type in self.config.intervention_types:
#             if intervention_type == "noise_injection":
#                 for noise_std in self.config.noise_levels:
#                     for layer_idx in target_layers_indices:
#                         params = {"magnitude": noise_std}
#                         key = f"{intervention_type}_layer_{layer_idx}_std{noise_std}"
#                         all_results[key] = self._run_single_intervention(model, tokenizer, intervention_type, layer_idx, params, baseline_results)
#             else:
#                 for layer_idx in target_layers_indices:
#                     key = f"{intervention_type}_layer_{layer_idx}"
#                     all_results[key] = self._run_single_intervention(model, tokenizer, intervention_type, layer_idx, {}, baseline_results)
        
#         return all_results

#     # And a corresponding change to _run_single_intervention's signature and logic
#     def _run_single_intervention(self, model, tokenizer, intervention_type: str, 
#                                 layer_idx: int, params: Dict, baseline_results: Dict) -> Dict:
#         # Simplified logic for single intervention run
#         test_samples = self.dataset_manager.perplexity_dataset
#         category_degradations = {}

#         handle = self.intervention_engine.apply_intervention(model, intervention_type, layer_idx, **params)
#         try:
#             for sample in tqdm(test_samples, desc=f"{intervention_type} L{layer_idx} {params}"):
#                 category = sample['category']
#                 baseline_ppl = baseline_results['perplexity_by_category'][category]['mean']
#                 intervention_ppl = self.evaluation_engine.calculate_perplexity(model, tokenizer, sample['text'])
                
#                 if baseline_ppl > 0 and baseline_ppl != float('inf'):
#                     degradation = (intervention_ppl - baseline_ppl) / baseline_ppl
#                     if category not in category_degradations:
#                         category_degradations[category] = []
#                     category_degradations[category].append(degradation)
#         finally:
#             handle.remove()

#         # Aggregate results
#         overall_degradation = np.mean([item for sublist in category_degradations.values() for item in sublist])
#         return {
#             "intervention_type": intervention_type,
#             "layer_idx": layer_idx,
#             "params": params,
#             "mean_performance_degradation": float(overall_degradation),
#             "degradation_by_category": {k: float(np.mean(v)) for k, v in category_degradations.items()}
#         }


#     # def _run_all_interventions(self, model, tokenizer, model_config: Dict, baseline_results: Dict) -> Dict:
#     #     """Run comprehensive intervention experiments"""
#     #     family = model_config["family"]
        
#     #     # Get target layers for this model family
#     #     if family in self.config.target_layers:
#     #         target_layers = self.config.target_layers[family]
#     #     else:
#     #         # Fallback to percentage-based layers
#     #         num_layers = self.model_manager.get_model_layers()
#     #         layer_percentages = [0.0, 0.25, 0.5, 0.75, 0.95]
#     #         target_layers = [int(p * (num_layers - 1)) for p in layer_percentages]
        
#     #     # Ensure target layers are valid
#     #     num_layers = self.model_manager.get_model_layers()
#     #     target_layers = [l for l in target_layers if 0 <= l < num_layers]
        
#     #     if not target_layers:
#     #         target_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
#     #         target_layers = [l for l in target_layers if 0 <= l < num_layers]
        
#     #     intervention_results = {}
        
#     #     logger.info(f"Running interventions on layers: {target_layers}")
        
#     #     # Define core interventions based on PoC insights
#     #     core_interventions = [
#     #         {"type": "ablation", "layers": [0], "params": {}},  # Layer 0 - critical from PoC
#     #         {"type": "ablation", "layers": target_layers[:3], "params": {}},  # Multiple layers
#     #         {"type": "mean_ablation", "layers": [target_layers[len(target_layers)//2]], "params": {}},
#     #         {"type": "noise_injection", "layers": [target_layers[len(target_layers)//2]], "params": {"magnitude": 0.1}},
#     #         {"type": "noise_injection", "layers": [target_layers[len(target_layers)//2]], "params": {"magnitude": 0.5}},
#     #         {"type": "random_replacement", "layers": [target_layers[-1]], "params": {}},
#     #     ]
        
#     #     # Sample subset of perplexity dataset for efficiency
#     #     test_samples = random.sample(self.dataset_manager.perplexity_dataset, 
#     #                                 min(15, len(self.dataset_manager.perplexity_dataset)))
        
#     #     for intervention_config in core_interventions:
#     #         intervention_type = intervention_config["type"]
#     #         layers = intervention_config["layers"]
#     #         params = intervention_config["params"]
            
#     #         for layer_idx in layers:
#     #             if layer_idx >= num_layers:
#     #                 continue
                    
#     #             intervention_key = f"{intervention_type}_layer_{layer_idx}"
#     #             if params:
#     #                 param_str = "_".join(f"{k}{v}" for k, v in params.items())
#     #                 intervention_key += f"_{param_str}"
                
#     #             logger.info(f"Running intervention: {intervention_key}")
                
#     #             try:
#     #                 intervention_result = self._run_single_intervention(
#     #                     model, tokenizer, intervention_type, layer_idx, params, test_samples, baseline_results
#     #                 )
#     #                 intervention_results[intervention_key] = intervention_result
#     #             except Exception as e:
#     #                 logger.error(f"Failed intervention {intervention_key}: {e}")
#     #                 continue
        
#     #     return intervention_results

#     # def _run_single_intervention(self, model, tokenizer, intervention_type: str, 
#     #                         layer_idx: int, params: Dict, test_samples: List[Dict], 
#     #                         baseline_results: Dict) -> Dict:
#     #     """Run a single intervention experiment"""
        
#     #     results = {
#     #         "intervention_type": intervention_type,
#     #         "layer_idx": layer_idx,
#     #         "params": params,
#     #         "performance_changes": {},
#     #         "sample_results": []
#     #     }
        
#     #     for sample in tqdm(test_samples, desc=f"{intervention_type} L{layer_idx}"):
#     #         category = sample["category"]
#     #         text = sample["text"]
            
#     #         try:
#     #             # Get baseline perplexity from baseline results if available
#     #             baseline_ppl = None
#     #             if category in baseline_results["perplexity_by_category"]:
#     #                 baseline_ppl = baseline_results["perplexity_by_category"][category]["mean"]
                
#     #             # If not available, calculate on the fly
#     #             if baseline_ppl is None:
#     #                 baseline_ppl = self.evaluation_engine.calculate_perplexity(model, tokenizer, text)
                
#     #             # Apply intervention
#     #             magnitude = params.get("magnitude", None)
#     #             handle = self.intervention_engine.apply_intervention(
#     #                 model, intervention_type, layer_idx, magnitude
#     #             )
                
#     #             try:
#     #                 # Measure perplexity with intervention
#     #                 intervention_ppl = self.evaluation_engine.calculate_perplexity(model, tokenizer, text)
                    
#     #                 # Calculate performance change
#     #                 if baseline_ppl != 0 and baseline_ppl != float('inf') and intervention_ppl != float('inf'):
#     #                     performance_change = (intervention_ppl - baseline_ppl) / baseline_ppl
#     #                 else:
#     #                     performance_change = 0.0
                    
#     #                 sample_result = {
#     #                     "category": category,
#     #                     "baseline_ppl": baseline_ppl,
#     #                     "intervention_ppl": intervention_ppl,
#     #                     "performance_change": performance_change
#     #                 }
                    
#     #                 results["sample_results"].append(sample_result)
                    
#     #                 # Aggregate by category
#     #                 if category not in results["performance_changes"]:
#     #                     results["performance_changes"][category] = []
#     #                 results["performance_changes"][category].append(performance_change)
                    
#     #             finally:
#     #                 # Always remove hook
#     #                 handle.remove()
                    
#     #         except Exception as e:
#     #             logger.warning(f"Error in intervention {intervention_type} layer {layer_idx}: {e}")
#     #             continue
        
#     #     # Aggregate results by category
#     #     for category in results["performance_changes"]:
#     #         changes = results["performance_changes"][category]
#     #         if changes:
#     #             results["performance_changes"][category] = {
#     #                 "mean": float(np.mean(changes)),
#     #                 "std": float(np.std(changes)),
#     #                 "median": float(np.median(changes)),
#     #                 "max": float(np.max(changes)),
#     #                 "min": float(np.min(changes)),
#     #                 "samples": len(changes)
#     #             }
        
#     #     return results

#     def _run_downstream_evaluations(self, model, tokenizer) -> Dict:
#         results = {}
#         for task_name in self.dataset_manager.downstream_datasets.keys():
#             results[task_name] = self._evaluate_task_with_interventions(model, tokenizer, task_name)
#         return results

#     def _evaluate_task_with_interventions(self, model, tokenizer, task_name):
#         task_results = {"baseline": {}, "layer_0_ablation": {}}
#         dataset = self.dataset_manager.downstream_datasets[task_name]
        
#         baseline_scores = [self.evaluation_engine.evaluate_downstream_task(model, tokenizer, s, task_name)['success'] for s in tqdm(dataset, desc=f"Baseline {task_name}")]
#         task_results["baseline"]["accuracy"] = np.mean(baseline_scores)

#         handle = self.intervention_engine.apply_intervention(model, "ablation", 0)
#         try:
#             intervention_scores = [self.evaluation_engine.evaluate_downstream_task(model, tokenizer, s, task_name)['success'] for s in tqdm(dataset, desc=f"Ablation {task_name}")]
#             task_results["layer_0_ablation"]["accuracy"] = np.mean(intervention_scores)
#         finally:
#             handle.remove()
        
#         task_results["layer_0_ablation"]["degradation"] = task_results["baseline"]["accuracy"] - task_results["layer_0_ablation"]["accuracy"]
#         return task_results
    
#     def _run_probing(self, model, tokenizer) -> Dict:
#         """Run extensive probing analysis"""
#         if not self.dataset_manager.probing_dataset:
#             logger.warning("No probing dataset available")
#             return {}
        
#         logger.info("Running extensive probing analysis")
        
#         probing_results = {
#             "category_classification": {},
#             "metadata": {
#                 "dataset_size": len(self.dataset_manager.probing_dataset["category_classification"]),
#                 "model_layers": self.model_manager.get_model_layers()
#             }
#         }
        
#         # Get probing data
#         probe_data = self.dataset_manager.probing_dataset["category_classification"]
        
#         # Sample data for efficiency if dataset is large
#         if len(probe_data) > 200:
#             probe_data = random.sample(probe_data, 200)
        
#         # Collect hidden states and labels
#         layer_count = self.model_manager.get_model_layers()
#         p1_states_by_layer = {i: [] for i in range(layer_count + 1)}
#         labels = []
        
#         logger.info(f"Extracting P1 states from {len(probe_data)} samples across {layer_count} layers")
        
#         for sample in tqdm(probe_data, desc="Extracting states"):
#             text = sample["text"]
#             label = sample["label"]
            
#             inputs = tokenizer(text, return_tensors="pt", truncation=True,
#                             max_length=self.config.max_sequence_length)
            
#             device = next(model.parameters()).device
#             inputs = {k: v.to(device) for k, v in inputs.items()}
            
#             try:
#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                     hidden_states = outputs.hidden_states
                    
#                     for layer_idx, layer_hidden in enumerate(hidden_states):
#                         if layer_hidden.size(1) > 0:  # Ensure we have tokens
#                             p1_state = layer_hidden[0, 0].cpu().numpy()
#                             p1_states_by_layer[layer_idx].append(p1_state)
                
#                 labels.append(label)
                
#             except Exception as e:
#                 logger.warning(f"Error extracting states: {e}")
#                 continue
        
#         # Train probes across layers
#         categories = list(set(labels))
#         y = [categories.index(label) for label in labels]
        
#         # Determine CV folds
#         min_samples_per_class = min([labels.count(cat) for cat in categories]) if categories else 1
#         cv_folds = min(self.config.cv_folds, min_samples_per_class)
        
#         if cv_folds < 2:
#             logger.warning("Insufficient samples for cross-validation, using simple train-test")
#             cv_folds = None
        
#         logger.info(f"Training probes with {cv_folds if cv_folds else 'simple'} evaluation")
        
#         # Sample layers for probing (every 4th layer to save time)
#         probe_layers = list(range(0, layer_count + 1, max(1, layer_count // 8)))
        
#         for layer_idx in probe_layers:
#             if layer_idx in p1_states_by_layer and p1_states_by_layer[layer_idx]:
#                 X = np.array(p1_states_by_layer[layer_idx])
                
#                 if len(set(y)) > 1 and len(X) >= (cv_folds or 2):
#                     try:
#                         # Logistic regression probe
#                         probe = LogisticRegression(
#                             random_state=self.config.random_seed, 
#                             max_iter=1000,
#                             solver='liblinear'  # Better for small datasets
#                         )
                        
#                         if cv_folds:
#                             # Stratified cross-validation
#                             skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
#                                                 random_state=self.config.random_seed)
#                             cv_scores = cross_val_score(probe, X, y, cv=skf, scoring='accuracy')
                            
#                             probing_results["category_classification"][f"layer_{layer_idx}"] = {
#                                 "accuracy_mean": float(np.mean(cv_scores)),
#                                 "accuracy_std": float(np.std(cv_scores)),
#                                 "cv_scores": [float(s) for s in cv_scores],
#                                 "num_features": X.shape[1],
#                                 "num_samples": X.shape[0],
#                                 "cv_folds": cv_folds,
#                                 "num_categories": len(categories)
#                             }
#                         else:
#                             # Simple train on all data
#                             probe.fit(X, y)
#                             accuracy = probe.score(X, y)
                            
#                             probing_results["category_classification"][f"layer_{layer_idx}"] = {
#                                 "accuracy_mean": float(accuracy),
#                                 "accuracy_std": 0.0,
#                                 "num_features": X.shape[1],
#                                 "num_samples": X.shape[0],
#                                 "cv_folds": 1,
#                                 "num_categories": len(categories),
#                                 "method": "simple_fit"
#                             }
                        
#                     except Exception as e:
#                         logger.warning(f"Probing failed for layer {layer_idx}: {e}")
#                         continue
        
#         return probing_results
    
#     def _summarize_model_stats(self, model_results: Dict) -> Dict:
#         """Perform statistical analysis for single model"""
#         stats_results = {
#             "intervention_significance": {},
#             "critical_layers": [],
#             "downstream_impacts": {},
#             "probing_summary": {},
#             "overall_assessment": {}
#         }
        
#         # Analyze intervention effects
#         if "intervention_results" in model_results:
#             for intervention_name, intervention_data in model_results["intervention_results"].items():
#                 if "performance_changes" in intervention_data:
                    
#                     # Collect all performance changes
#                     all_changes = []
#                     for category, category_data in intervention_data["performance_changes"].items():
#                         if isinstance(category_data, dict) and "mean" in category_data:
#                             # Use the sample size to weight the mean appropriately
#                             sample_size = category_data.get("samples", 1)
#                             all_changes.extend([category_data["mean"]] * sample_size)
                    
#                     if len(all_changes) > 1:
#                         # Statistical tests
#                         try:
#                             t_stat, p_value = scipy_stats.ttest_1samp(all_changes, 0)
#                             effect_size = np.mean(all_changes)
                            
#                             stats_results["intervention_significance"][intervention_name] = {
#                                 "t_statistic": float(t_stat),
#                                 "p_value": float(p_value),
#                                 "effect_size": float(effect_size),
#                                 "effect_size_abs": float(abs(effect_size)),
#                                 "significant": p_value < self.config.statistical_significance_level,
#                                 "large_effect": abs(effect_size) > self.config.effect_size_threshold,
#                                 "sample_size": len(all_changes)
#                             }
                            
#                             # Identify critical layers
#                             if (p_value < self.config.statistical_significance_level and 
#                                 abs(effect_size) > self.config.effect_size_threshold):
                                
#                                 layer_idx = intervention_data.get("layer_idx")
#                                 if layer_idx is not None:
#                                     stats_results["critical_layers"].append({
#                                         "layer": layer_idx,
#                                         "intervention": intervention_name,
#                                         "effect_size": float(effect_size),
#                                         "p_value": float(p_value)
#                                     })
#                         except Exception as e:
#                             logger.warning(f"Statistical test failed for {intervention_name}: {e}")
        
#         # Analyze downstream task impacts
#         if "downstream_results" in model_results:
#             for task_name, task_data in model_results["downstream_results"].items():
#                 if "layer_0_ablation" in task_data and "baseline" in task_data:
#                     baseline_acc = task_data["baseline"].get("accuracy", 0)
#                     intervention_acc = task_data["layer_0_ablation"].get("accuracy", 0)
#                     degradation = task_data["layer_0_ablation"].get("degradation", 0)
                    
#                     stats_results["downstream_impacts"][task_name] = {
#                         "baseline_accuracy": float(baseline_acc),
#                         "intervention_accuracy": float(intervention_acc),
#                         "degradation": float(degradation),
#                         "severe_impact": degradation > 0.1,  # >10% degradation
#                         "relative_degradation": float(degradation / baseline_acc) if baseline_acc > 0 else 0
#                     }
        
#         # Summarize probing results
#         if "probing_results" in model_results and "category_classification" in model_results["probing_results"]:
#             probing_data = model_results["probing_results"]["category_classification"]
            
#             if probing_data:
#                 best_accuracy = 0
#                 best_layer = 0
                
#                 for layer_key, layer_data in probing_data.items():
#                     if isinstance(layer_data, dict) and "accuracy_mean" in layer_data:
#                         acc = layer_data["accuracy_mean"]
#                         if acc > best_accuracy:
#                             best_accuracy = acc
#                             try:
#                                 best_layer = int(layer_key.split('_')[1])
#                             except (ValueError, IndexError):
#                                 best_layer = 0
                
#                 # Get number of categories
#                 sample_layer_data = next(iter(probing_data.values()), {})
#                 num_categories = sample_layer_data.get("num_categories", 5)
#                 chance_level = 1.0 / num_categories
                
#                 stats_results["probing_summary"] = {
#                     "best_accuracy": float(best_accuracy),
#                     "best_layer": best_layer,
#                     "chance_level": float(chance_level),
#                     "above_chance": best_accuracy > chance_level + 0.05,
#                     "improvement_over_chance": float(best_accuracy - chance_level)
#                 }
        
#         # Overall assessment
#         significant_interventions = sum(1 for sig in stats_results["intervention_significance"].values() 
#                                     if sig.get("significant", False))
#         total_interventions = len(stats_results["intervention_significance"])
        
#         large_effects = sum(1 for sig in stats_results["intervention_significance"].values() 
#                         if sig.get("large_effect", False))
        
#         # Determine if P1 is functionally important
#         significance_rate = significant_interventions / total_interventions if total_interventions > 0 else 0
#         has_large_effects = large_effects > 0
#         has_downstream_impact = any(
#             impact.get("severe_impact", False) 
#             for impact in stats_results["downstream_impacts"].values()
#         )
        
#         stats_results["overall_assessment"] = {
#             "total_interventions": total_interventions,
#             "significant_interventions": significant_interventions,
#             "large_effect_interventions": large_effects,
#             "significance_rate": float(significance_rate),
#             "critical_layers_found": len(stats_results["critical_layers"]),
#             "p1_functionally_important": (
#                 significance_rate > 0.3 or  # >30% of interventions significant
#                 has_large_effects or         # Has at least one large effect
#                 has_downstream_impact        # Shows downstream task impact
#             )
#         }
        
#         return stats_results
    
#     def _run_baseline_evaluations(self, model, tokenizer) -> Dict:
#         """Run comprehensive baseline evaluations"""
#         baseline_results = {
#             "perplexity_by_category": {},
#             "attention_patterns": {},
#             "model_info": {
#                 "num_layers": self.model_manager.get_model_layers(),
#                 "vocab_size": len(tokenizer),
#                 "model_size": self.model_manager.current_model_config.get("size", "unknown")
#             }
#         }
        
#         logger.info("Measuring baseline performance across text categories")
        
#         # Group samples by category
#         samples_by_category = {}
#         for sample in self.dataset_manager.perplexity_dataset:
#             category = sample["category"]
#             if category not in samples_by_category:
#                 samples_by_category[category] = []
#             samples_by_category[category].append(sample)
        
#         # Process each category
#         for category, samples in samples_by_category.items():
#             logger.info(f"Processing {category} category ({len(samples)} samples)")
            
#             category_ppls = []
#             category_attention_patterns = []
            
#             for sample in tqdm(samples, desc=f"Baseline {category}"):
#                 text = sample["text"]
                
#                 # Calculate perplexity
#                 ppl = self.evaluation_engine.calculate_perplexity(model, tokenizer, text)
#                 if ppl != float('inf'):  # Only include valid perplexities
#                     category_ppls.append(ppl)
                
#                 # Extract attention patterns (for first 2 samples to save time)
#                 if len(category_attention_patterns) < 2:
#                     try:
#                         attention_data = self.evaluation_engine.extract_attention_patterns(model, tokenizer, text)
#                         category_attention_patterns.append(attention_data)
#                     except Exception as e:
#                         logger.warning(f"Failed to extract attention for {category}: {e}")
            
#             # Store aggregated results
#             if category_ppls:
#                 baseline_results["perplexity_by_category"][category] = {
#                     "mean": float(np.mean(category_ppls)),
#                     "std": float(np.std(category_ppls)),
#                     "median": float(np.median(category_ppls)),
#                     "min": float(np.min(category_ppls)),
#                     "max": float(np.max(category_ppls)),
#                     "samples": len(category_ppls),
#                     "all_values": category_ppls
#                 }
            
#             # Average attention patterns across samples
#             if category_attention_patterns:
#                 try:
#                     avg_p1_attention = np.mean([
#                         pattern["p1_attention_by_layer"] 
#                         for pattern in category_attention_patterns
#                     ], axis=0).tolist()
                    
#                     avg_p1_norms = np.mean([
#                         pattern["p1_hidden_norms"] 
#                         for pattern in category_attention_patterns
#                     ], axis=0).tolist()
                    
#                     baseline_results["attention_patterns"][category] = {
#                         "p1_attention_by_layer": avg_p1_attention,
#                         "p1_hidden_norms": avg_p1_norms,
#                         "peak_layer": int(np.argmax(avg_p1_attention)),
#                         "peak_value": float(np.max(avg_p1_attention))
#                     }
#                 except Exception as e:
#                     logger.warning(f"Failed to process attention patterns for {category}: {e}")
        
#         logger.info("Baseline measurement completed")
#         return baseline_results

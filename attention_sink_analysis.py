import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
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
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for attention sink analysis"""
    model_name: str = "gpt2"
    num_sink_tokens: int = 4
    max_length: int = 512
    device: str = "auto"
    output_dir: str = "./attention_sink_analysis"
    save_plots: bool = True
    save_data: bool = True

class AttentionSinkAnalyzer:
    """
    A comprehensive analyzer for attention sink phenomena in transformer models.
    
    This class provides methods to:
    1. Quantify attention sink effects across layers and heads
    2. Extract and analyze hidden state representations of sink tokens
    3. Visualize attention patterns and sink effectiveness
    4. Save results for further analysis
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing AttentionSinkAnalyzer with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("MPS available. Using Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
        return device
    
    def load_model_and_tokenizer(self) -> Tuple[torch.nn.Module, any]:
        """
        Load model and tokenizer with proper configuration
        """
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Use AutoTokenizer for better compatibility
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Handle pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                output_attentions=True,
                output_hidden_states=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if not (self.device.type == "cuda" and hasattr(model, 'hf_device_map')):
                model = model.to(self.device)
            
            model.eval()
            
            # Resize embeddings if we added tokens
            if tokenizer.pad_token == '[PAD]':
                model.resize_token_embeddings(len(tokenizer))
            
            logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def process_text(self, model: torch.nn.Module, tokenizer: any, text: str) -> Dict:
        """
        Process text and extract attention patterns and hidden states
        """
        logger.info(f"Processing text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_length
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get actual sequence length (excluding padding)
        actual_length = attention_mask.sum().item()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:actual_length])
        
        logger.info(f"Sequence length: {actual_length}, Tokens: {len(tokens)}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Extract outputs
        attentions = outputs.attentions  # Tuple of (batch, heads, seq_len, seq_len)
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
        
        # Move to CPU for analysis
        attentions = tuple(attn.cpu().float() for attn in attentions)
        hidden_states = tuple(hs.cpu().float() for hs in hidden_states)
        
        return {
            'attentions': attentions,
            'hidden_states': hidden_states,
            'input_ids': input_ids.cpu(),
            'tokens': tokens,
            'actual_length': actual_length,
            'attention_mask': attention_mask.cpu()
        }
    
    def analyze_attention_sinks(self, attentions: Tuple, actual_length: int, tokens: List[str]) -> Dict:
        """
        Comprehensive analysis of attention sink patterns
        """
        logger.info(f"Analyzing attention sinks for first {self.config.num_sink_tokens} tokens")
        
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        effective_sinks = min(self.config.num_sink_tokens, actual_length)
        
        if actual_length <= effective_sinks:
            logger.warning(f"Sequence too short ({actual_length}) for meaningful sink analysis")
            return {'error': 'sequence_too_short'}
        
        results = {
            'layer_averages': [],
            'head_patterns': [],  # Per-head analysis
            'position_analysis': [],  # How attention to each sink position varies
            'temporal_analysis': [],  # How sink attention changes across sequence positions
            'effective_sinks': effective_sinks,
            'sequence_length': actual_length,
            'tokens': tokens
        }
        
        # Analyze each layer
        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn = layer_attn[0]  # Remove batch dimension: (heads, seq_len, seq_len)
            
            # Crop to actual sequence length
            layer_attn = layer_attn[:, :actual_length, :actual_length]
            
            # Average across heads for layer-level analysis
            avg_attn = layer_attn.mean(dim=0)  # (seq_len, seq_len)
            
            # Extract attention from non-sink tokens to sink tokens
            if actual_length > effective_sinks:
                attn_to_sinks = avg_attn[effective_sinks:, :effective_sinks]  # (later_tokens, sink_tokens)
                
                # Layer average: mean attention from later tokens to any sink token
                layer_avg = attn_to_sinks.sum(dim=1).mean().item()
                results['layer_averages'].append(layer_avg)
                
                # Per-head analysis for this layer
                head_scores = []
                for head_idx in range(num_heads):
                    head_attn = layer_attn[head_idx]
                    head_attn_to_sinks = head_attn[effective_sinks:, :effective_sinks]
                    head_score = head_attn_to_sinks.sum(dim=1).mean().item()
                    head_scores.append(head_score)
                results['head_patterns'].append(head_scores)
                
                # Position analysis: attention to each sink position
                position_scores = attn_to_sinks.mean(dim=0).tolist()  # Average across all later tokens
                results['position_analysis'].append(position_scores)
                
                # Temporal analysis: how sink attention varies by query position
                temporal_scores = attn_to_sinks.sum(dim=1).tolist()  # Total sink attention per query position
                results['temporal_analysis'].append(temporal_scores)
                
                logger.info(f"Layer {layer_idx+1}: Avg attention to sinks = {layer_avg:.4f}")
            else:
                results['layer_averages'].append(0.0)
                results['head_patterns'].append([0.0] * num_heads)
                results['position_analysis'].append([0.0] * effective_sinks)
                results['temporal_analysis'].append([0.0])
        
        return results
    
    def analyze_sink_representations(self, hidden_states: Tuple, effective_sinks: int, actual_length: int) -> Dict:
        """
        Analyze the hidden state representations of sink tokens
        """
        logger.info(f"Analyzing hidden representations of {effective_sinks} sink tokens")
        
        if effective_sinks == 0:
            return {'error': 'no_sinks_to_analyze'}
        
        # Skip embedding layer, take transformer layers
        layer_outputs = hidden_states[1:]  
        
        results = {
            'sink_states_per_layer': [],
            'similarity_analysis': [],
            'norm_analysis': [],
            'effective_sinks': effective_sinks
        }
        
        for layer_idx, layer_hs in enumerate(layer_outputs):
            # Extract sink token representations
            sink_states = layer_hs[0, :effective_sinks, :]  # (effective_sinks, hidden_dim)
            results['sink_states_per_layer'].append(sink_states)
            
            # Similarity analysis: how similar are sink representations to each other?
            if effective_sinks > 1:
                similarity_matrix = F.cosine_similarity(
                    sink_states.unsqueeze(1), 
                    sink_states.unsqueeze(0), 
                    dim=2
                )
                # Average pairwise similarity (excluding diagonal)
                mask = ~torch.eye(effective_sinks, dtype=bool)
                avg_similarity = similarity_matrix[mask].mean().item()
                results['similarity_analysis'].append(avg_similarity)
            else:
                results['similarity_analysis'].append(1.0)
            
            # Norm analysis: magnitude of sink representations
            norms = torch.norm(sink_states, dim=1).tolist()
            results['norm_analysis'].append(norms)
            
            logger.info(f"Layer {layer_idx+1}: Avg sink similarity = {results['similarity_analysis'][-1]:.4f}")
        
        return results
    
    def create_visualizations(self, attention_results: Dict, representation_results: Dict, 
                            text_sample: str) -> None:
        """
        Create comprehensive visualizations of the analysis results
        """
        if 'error' in attention_results or 'error' in representation_results:
            logger.warning("Skipping visualizations due to analysis errors")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Layer-wise attention to sinks
        plt.subplot(2, 4, 1)
        layers = range(1, len(attention_results['layer_averages']) + 1)
        plt.plot(layers, attention_results['layer_averages'], 'o-', linewidth=2, markersize=6)
        plt.title('Attention to Sinks by Layer', fontsize=12, fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Average Attention Score')
        plt.grid(True, alpha=0.3)
        
        # 2. Head-wise attention patterns (heatmap)
        plt.subplot(2, 4, 2)
        head_data = np.array(attention_results['head_patterns'])
        if head_data.size > 0:
            sns.heatmap(head_data, cmap='viridis', cbar=True, 
                       xticklabels=[f'H{i+1}' for i in range(head_data.shape[1])],
                       yticklabels=[f'L{i+1}' for i in range(head_data.shape[0])])
        plt.title('Attention to Sinks by Head', fontsize=12, fontweight='bold')
        plt.xlabel('Attention Head')
        plt.ylabel('Layer')
        
        # 3. Position-wise attention (which sink positions get more attention)
        plt.subplot(2, 4, 3)
        position_data = np.array(attention_results['position_analysis'])
        if position_data.size > 0:
            avg_by_position = position_data.mean(axis=0)
            positions = range(1, len(avg_by_position) + 1)
            plt.bar(positions, avg_by_position, alpha=0.7)
            plt.title('Attention by Sink Position', fontsize=12, fontweight='bold')
            plt.xlabel('Sink Token Position')
            plt.ylabel('Average Attention Received')
            plt.xticks(positions)
        
        # 4. Temporal analysis (how sink attention varies across sequence)
        plt.subplot(2, 4, 4)
        if attention_results['temporal_analysis'] and len(attention_results['temporal_analysis'][0]) > 1:
            temporal_data = np.array(attention_results['temporal_analysis'])
            # Show middle layer as example
            mid_layer = len(temporal_data) // 2
            query_positions = range(attention_results['effective_sinks'] + 1, 
                                  attention_results['sequence_length'] + 1)
            plt.plot(query_positions, temporal_data[mid_layer], 'o-', alpha=0.7)
            plt.title(f'Sink Attention by Query Position (Layer {mid_layer+1})', 
                     fontsize=12, fontweight='bold')
            plt.xlabel('Query Token Position')
            plt.ylabel('Total Attention to Sinks')
        
        # 5. Sink representation similarity across layers
        plt.subplot(2, 4, 5)
        similarity_scores = representation_results['similarity_analysis']
        plt.plot(layers, similarity_scores, 's-', linewidth=2, markersize=6, color='red')
        plt.title('Sink Representation Similarity', fontsize=12, fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        # 6. Representation norms
        plt.subplot(2, 4, 6)
        norm_data = np.array(representation_results['norm_analysis'])
        if norm_data.size > 0:
            for sink_idx in range(norm_data.shape[1]):
                plt.plot(layers, norm_data[:, sink_idx], 'o-', 
                        label=f'Sink {sink_idx+1}', alpha=0.7)
            plt.title('Representation Norms by Layer', fontsize=12, fontweight='bold')
            plt.xlabel('Layer')
            plt.ylabel('L2 Norm')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Attention matrix visualization (sample layer)
        plt.subplot(2, 4, 7)
        if 'attentions' in locals():  # This would need to be passed separately
            # For now, create a placeholder
            plt.text(0.5, 0.5, 'Attention Matrix\n(Implementation needed)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Attention Matrix (Sample Layer)', fontsize=12, fontweight='bold')
        
        # 8. Summary statistics
        plt.subplot(2, 4, 8)
        plt.axis('off')
        stats_text = f"""
        Analysis Summary:
        
        Model: {self.config.model_name}
        Sequence Length: {attention_results['sequence_length']}
        Sink Tokens: {attention_results['effective_sinks']}
        
        Max Layer Attention: {max(attention_results['layer_averages']):.4f}
        Min Layer Attention: {min(attention_results['layer_averages']):.4f}
        
        Avg Sink Similarity: {np.mean(similarity_scores):.4f}
        
        Sample Text:
        "{text_sample[:100]}..."
        """
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_path = self.output_dir / f"attention_sink_analysis_{self.config.model_name.replace('/', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive plot saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self, attention_results: Dict, representation_results: Dict, 
                    text_sample: str) -> None:
        """
        Save analysis results to files
        """
        if not self.config.save_data:
            return
        
        # Prepare results for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results = {
            'config': {
                'model_name': self.config.model_name,
                'num_sink_tokens': self.config.num_sink_tokens,
                'max_length': self.config.max_length
            },
            'text_sample': text_sample,
            'attention_analysis': convert_tensors(attention_results),
            'representation_analysis': convert_tensors(representation_results)
        }
        
        # Save to JSON
        json_path = self.output_dir / f"results_{self.config.model_name.replace('/', '_')}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {json_path}")
        
        # Save sink representations separately (they're large)
        if 'sink_states_per_layer' in representation_results:
            torch.save(
                representation_results['sink_states_per_layer'],
                self.output_dir / f"sink_representations_{self.config.model_name.replace('/', '_')}.pt"
            )
            logger.info("Sink representations saved separately")
    
    def run_analysis(self, text: str) -> Dict:
        """
        Run complete attention sink analysis pipeline
        """
        logger.info("Starting comprehensive attention sink analysis")
        
        # Load model
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Process text
        processed_data = self.process_text(model, tokenizer, text)
        
        # Analyze attention patterns
        attention_results = self.analyze_attention_sinks(
            processed_data['attentions'],
            processed_data['actual_length'],
            processed_data['tokens']
        )
        
        # Analyze representations
        representation_results = self.analyze_sink_representations(
            processed_data['hidden_states'],
            attention_results.get('effective_sinks', 0),
            processed_data['actual_length']
        )
        
        # Create visualizations
        self.create_visualizations(attention_results, representation_results, text)
        
        # Save results
        self.save_results(attention_results, representation_results, text)
        
        logger.info("Analysis complete!")
        
        return {
            'attention_analysis': attention_results,
            'representation_analysis': representation_results,
            'processed_data': processed_data
        }

def main():
    """
    Main execution function with multiple test cases
    """
    # Configuration
    config = AnalysisConfig(
        model_name="gpt2",  # Try: "gpt2-medium", "microsoft/DialoGPT-medium"
        num_sink_tokens=4,
        max_length=512,
        save_plots=True,
        save_data=True
    )
    
    # Test texts of different types and lengths
    test_texts = {
        "narrative": """
        In the heart of the bustling city, amidst the towering skyscrapers and the ceaseless hum of traffic, 
        lay a small, forgotten park. This park, a verdant oasis named Willow Creek, was a relic from a bygone era, 
        a testament to nature's resilience. Its ancient willow trees drooped gracefully over a serene pond, 
        their leaves whispering secrets to the gentle breeze. Squirrels darted across well-worn paths, 
        and migratory birds often found refuge in its dense foliage during their long journeys. 
        Despite being surrounded by concrete and steel, Willow Creek maintained an air of tranquility, 
        a sanctuary for those seeking a momentary escape from urban chaos. The city planners had often debated 
        its fate, with proposals ranging from modern development to complete preservation. Yet, for now, it remained, 
        a cherished green lung in the metropolitan expanse.
        """,
        
        "technical": """
        The Transformer architecture revolutionized natural language processing through its self-attention mechanism. 
        Unlike recurrent neural networks, Transformers process sequences in parallel, computing attention weights 
        between all pairs of tokens simultaneously. The multi-head attention allows the model to focus on different 
        aspects of the input representation. Layer normalization and residual connections stabilize training, 
        while positional encodings provide sequence order information. The encoder-decoder structure enables 
        various tasks from translation to text generation. Recent developments like BERT and GPT have demonstrated 
        the power of pre-training large Transformer models on massive text corpora.
        """,
        
        "dialogue": """
        "Hello there!" Sarah called out as she entered the coffee shop.
        "Good morning, Sarah! The usual?" replied Tom from behind the counter.
        "Actually, I think I'll try something different today. What do you recommend?"
        "Well, our new caramel macchiato is quite popular. It has a nice balance of sweet and bitter."
        "That sounds perfect. And maybe one of those blueberry muffins too?"
        "Coming right up! How's the new job going?"
        "It's challenging but exciting. Lots to learn, but the team is really supportive."
        """
    }
    
    # Run analysis
    analyzer = AttentionSinkAnalyzer(config)
    
    # Analyze narrative text (primary analysis)
    results = analyzer.run_analysis(test_texts["narrative"])
    
    # Print summary
    attention_results = results['attention_analysis']
    if 'layer_averages' in attention_results:
        print(f"\n{'='*60}")
        print("ATTENTION SINK ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {config.model_name}")
        print(f"Sequence Length: {attention_results['sequence_length']}")
        print(f"Sink Tokens Analyzed: {attention_results['effective_sinks']}")
        print(f"Average Attention to Sinks: {np.mean(attention_results['layer_averages']):.4f}")
        print(f"Peak Layer for Sinks: Layer {np.argmax(attention_results['layer_averages']) + 1}")
        print(f"Peak Attention Score: {max(attention_results['layer_averages']):.4f}")
        
        # Position analysis
        if attention_results['position_analysis']:
            pos_data = np.array(attention_results['position_analysis'])
            avg_by_pos = pos_data.mean(axis=0)
            print(f"\nAttention by Position:")
            for i, score in enumerate(avg_by_pos):
                print(f"  Position {i+1}: {score:.4f}")
        
        print(f"\nResults saved to: {analyzer.output_dir}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

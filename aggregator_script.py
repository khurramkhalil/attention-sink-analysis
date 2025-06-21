import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionSinkAggregator:
    """
    Aggregates attention sink analysis results from multiple model-text combinations
    and creates comprehensive comparison tables.
    """
    
    def __init__(self, analysis_directory: str = "./comprehensive_sink_analysis"):
        self.analysis_dir = Path(analysis_directory)
        self.results_data = {}
        
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {analysis_directory}")
        
        logger.info(f"Initializing aggregator for directory: {self.analysis_dir}")
    
    def collect_all_results(self) -> Dict:
        """
        Traverse all subdirectories and collect results from JSON files
        """
        logger.info("Collecting results from all subdirectories...")
        
        collected_results = {}
        
        # Look for subdirectories containing results
        for subdir in self.analysis_dir.iterdir():
            if subdir.is_dir():
                # Look for results JSON file in this subdirectory
                json_files = list(subdir.glob("results_*.json"))
                
                if json_files:
                    json_file = json_files[0]  # Take the first (should be only one)
                    
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract model and text type from directory name or file data
                        dir_name = subdir.name
                        model_name = data.get('config', {}).get('model_name', 'unknown')
                        
                        # Create a unique identifier
                        combination_id = dir_name
                        
                        collected_results[combination_id] = {
                            'model_name': model_name,
                            'directory_name': dir_name,
                            'data': data,
                            'file_path': str(json_file)
                        }
                        
                        logger.info(f"✅ Loaded: {combination_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error loading {json_file}: {e}")
                        continue
        
        logger.info(f"Successfully collected {len(collected_results)} result sets")
        self.results_data = collected_results
        return collected_results
    
    def create_attention_by_position_table(self) -> pd.DataFrame:
        """
        Create table showing attention received by each sink position (1-4)
        across all model-text combinations
        """
        logger.info("Creating attention by sink position table...")
        
        rows_data = []
        
        for combination_id, result_info in self.results_data.items():
            try:
                attention_analysis = result_info['data']['attention_analysis']
                
                # Skip if there was an error in analysis
                if 'error' in attention_analysis:
                    logger.warning(f"Skipping {combination_id} due to analysis error")
                    continue
                
                # Extract position analysis data
                position_analysis = attention_analysis.get('position_analysis', [])
                
                if not position_analysis:
                    logger.warning(f"No position analysis data for {combination_id}")
                    continue
                
                # Convert to numpy array and average across layers
                position_data = np.array(position_analysis)  # Shape: (layers, positions)
                avg_by_position = position_data.mean(axis=0)  # Average across layers
                
                # Ensure we have exactly 4 positions (pad with NaN if needed)
                position_scores = [np.nan] * 4
                for i in range(min(len(avg_by_position), 4)):
                    position_scores[i] = avg_by_position[i]
                
                # Extract metadata
                model_name = result_info['model_name']
                text_type = combination_id.replace(model_name.replace('/', '_').replace('-', '_') + '_', '')
                
                row_data = {
                    'Model_Text_Combination': combination_id,
                    'Model': model_name,
                    'Text_Type': text_type,
                    'Sink_Position_1': position_scores[0],
                    'Sink_Position_2': position_scores[1],
                    'Sink_Position_3': position_scores[2],
                    'Sink_Position_4': position_scores[3],
                    'Total_Sink_Attention': np.nansum(position_scores),
                    'Sequence_Length': attention_analysis.get('sequence_length', 'N/A'),
                    'Effective_Sinks': attention_analysis.get('effective_sinks', 'N/A')
                }
                
                rows_data.append(row_data)
                
            except Exception as e:
                logger.error(f"Error processing {combination_id} for position table: {e}")
                continue
        
        df = pd.DataFrame(rows_data)
        
        if not df.empty:
            # Sort by model and text type for better readability
            df = df.sort_values(['Model', 'Text_Type'])
            
            # Add summary statistics
            logger.info(f"Position table created with {len(df)} entries")
            logger.info(f"Average attention by position:")
            for i in range(1, 5):
                col_name = f'Sink_Position_{i}'
                if col_name in df.columns:
                    avg_val = df[col_name].mean()
                    logger.info(f"  Position {i}: {avg_val:.4f}")
        
        return df
    
    def create_representation_similarity_table(self) -> pd.DataFrame:
        """
        Create table showing sink representation similarity across layers
        """
        logger.info("Creating sink representation similarity table...")
        
        rows_data = []
        
        for combination_id, result_info in self.results_data.items():
            try:
                representation_analysis = result_info['data']['representation_analysis']
                
                # Skip if there was an error in analysis
                if 'error' in representation_analysis:
                    logger.warning(f"Skipping {combination_id} due to representation analysis error")
                    continue
                
                similarity_scores = representation_analysis.get('similarity_analysis', [])
                
                if not similarity_scores:
                    logger.warning(f"No similarity analysis data for {combination_id}")
                    continue
                
                # Extract metadata
                model_name = result_info['model_name']
                text_type = combination_id.replace(model_name.replace('/', '_').replace('-', '_') + '_', '')
                
                # Create row with layer-wise similarity scores
                row_data = {
                    'Model_Text_Combination': combination_id,
                    'Model': model_name,
                    'Text_Type': text_type,
                    'Average_Similarity': np.mean(similarity_scores),
                    'Max_Similarity': np.max(similarity_scores),
                    'Min_Similarity': np.min(similarity_scores),
                    'Effective_Sinks': representation_analysis.get('effective_sinks', 'N/A')
                }
                
                # Add individual layer similarities
                for layer_idx, similarity in enumerate(similarity_scores):
                    row_data[f'Layer_{layer_idx + 1}_Similarity'] = similarity
                
                rows_data.append(row_data)
                
            except Exception as e:
                logger.error(f"Error processing {combination_id} for similarity table: {e}")
                continue
        
        df = pd.DataFrame(rows_data)
        
        if not df.empty:
            # Sort by model and text type
            df = df.sort_values(['Model', 'Text_Type'])
            logger.info(f"Similarity table created with {len(df)} entries")
            
            # Find the maximum number of layers across all models
            layer_cols = [col for col in df.columns if col.startswith('Layer_') and col.endswith('_Similarity')]
            if layer_cols:
                logger.info(f"Maximum layers found: {len(layer_cols)}")
        
        return df
    
    def create_attention_by_layer_table(self) -> pd.DataFrame:
        """
        Create table showing attention to sinks by layer across all combinations
        """
        logger.info("Creating attention to sinks by layer table...")
        
        rows_data = []
        
        for combination_id, result_info in self.results_data.items():
            try:
                attention_analysis = result_info['data']['attention_analysis']
                
                # Skip if there was an error in analysis
                if 'error' in attention_analysis:
                    logger.warning(f"Skipping {combination_id} due to attention analysis error")
                    continue
                
                layer_averages = attention_analysis.get('layer_averages', [])
                
                if not layer_averages:
                    logger.warning(f"No layer averages data for {combination_id}")
                    continue
                
                # Extract metadata
                model_name = result_info['model_name']
                text_type = combination_id.replace(model_name.replace('/', '_').replace('-', '_') + '_', '')
                
                # Create row with layer-wise attention scores
                row_data = {
                    'Model_Text_Combination': combination_id,
                    'Model': model_name,
                    'Text_Type': text_type,
                    'Average_Attention_All_Layers': np.mean(layer_averages),
                    'Peak_Layer_Attention': np.max(layer_averages),
                    'Peak_Layer_Number': np.argmax(layer_averages) + 1,
                    'Min_Layer_Attention': np.min(layer_averages),
                    'Attention_Std_Dev': np.std(layer_averages),
                    'Sequence_Length': attention_analysis.get('sequence_length', 'N/A'),
                    'Effective_Sinks': attention_analysis.get('effective_sinks', 'N/A')
                }
                
                # Add individual layer attention scores
                for layer_idx, attention_score in enumerate(layer_averages):
                    row_data[f'Layer_{layer_idx + 1}_Attention'] = attention_score
                
                rows_data.append(row_data)
                
            except Exception as e:
                logger.error(f"Error processing {combination_id} for layer attention table: {e}")
                continue
        
        df = pd.DataFrame(rows_data)
        
        if not df.empty:
            # Sort by model and text type
            df = df.sort_values(['Model', 'Text_Type'])
            logger.info(f"Layer attention table created with {len(df)} entries")
            
            # Find the maximum number of layers
            layer_cols = [col for col in df.columns if col.startswith('Layer_') and col.endswith('_Attention')]
            if layer_cols:
                logger.info(f"Maximum layers found: {len(layer_cols)}")
        
        return df
    
    def save_tables(self, output_directory: str = "./aggregated_analysis_tables"):
        """
        Generate all tables and save them as CSV files
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving aggregated tables to: {output_dir}")
        
        # 1. Attention by Sink Position Table
        logger.info("Generating attention by sink position table...")
        position_table = self.create_attention_by_position_table()
        
        if not position_table.empty:
            position_file = output_dir / "attention_by_sink_position.csv"
            position_table.to_csv(position_file, index=False)
            logger.info(f"✅ Saved: {position_file} ({len(position_table)} rows)")
        else:
            logger.warning("❌ Position table is empty - no data to save")
        
        # 2. Sink Representation Similarity Table
        logger.info("Generating sink representation similarity table...")
        similarity_table = self.create_representation_similarity_table()
        
        if not similarity_table.empty:
            similarity_file = output_dir / "sink_representation_similarity_by_layer.csv"
            similarity_table.to_csv(similarity_file, index=False)
            logger.info(f"✅ Saved: {similarity_file} ({len(similarity_table)} rows)")
        else:
            logger.warning("❌ Similarity table is empty - no data to save")
        
        # 3. Attention to Sinks by Layer Table
        logger.info("Generating attention to sinks by layer table...")
        layer_attention_table = self.create_attention_by_layer_table()
        
        if not layer_attention_table.empty:
            layer_file = output_dir / "attention_to_sinks_by_layer.csv"
            layer_attention_table.to_csv(layer_file, index=False)
            logger.info(f"✅ Saved: {layer_file} ({len(layer_attention_table)} rows)")
        else:
            logger.warning("❌ Layer attention table is empty - no data to save")
        
        # 4. Create a summary report
        self.create_summary_report(output_dir, position_table, similarity_table, layer_attention_table)
        
        return {
            'position_table': position_table,
            'similarity_table': similarity_table,
            'layer_attention_table': layer_attention_table,
            'output_directory': str(output_dir)
        }
    
    def create_summary_report(self, output_dir: Path, position_df: pd.DataFrame, 
                            similarity_df: pd.DataFrame, layer_df: pd.DataFrame):
        """
        Create a summary report with key insights
        """
        logger.info("Creating summary report...")
        
        report_lines = []
        report_lines.append("ATTENTION SINK ANALYSIS - AGGREGATED SUMMARY REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW:")
        report_lines.append("-" * 30)
        
        if not position_df.empty:
            unique_models = position_df['Model'].nunique()
            unique_text_types = position_df['Text_Type'].nunique()
            total_combinations = len(position_df)
            
            report_lines.append(f"Total model-text combinations analyzed: {total_combinations}")
            report_lines.append(f"Unique models: {unique_models}")
            report_lines.append(f"Unique text types: {unique_text_types}")
            report_lines.append("")
            
            # Model breakdown
            report_lines.append("Models analyzed:")
            for model in sorted(position_df['Model'].unique()):
                count = len(position_df[position_df['Model'] == model])
                report_lines.append(f"  - {model}: {count} text types")
            report_lines.append("")
            
            # Text type breakdown
            report_lines.append("Text types analyzed:")
            for text_type in sorted(position_df['Text_Type'].unique()):
                count = len(position_df[position_df['Text_Type'] == text_type])
                report_lines.append(f"  - {text_type}: {count} models")
        
        report_lines.append("")
        
        # Key findings - Sink Position Analysis
        if not position_df.empty:
            report_lines.append("KEY FINDINGS - SINK POSITION ANALYSIS:")
            report_lines.append("-" * 45)
            
            # Average attention by position across all combinations
            for i in range(1, 5):
                col_name = f'Sink_Position_{i}'
                if col_name in position_df.columns:
                    avg_attention = position_df[col_name].mean()
                    std_attention = position_df[col_name].std()
                    report_lines.append(f"Position {i} - Avg: {avg_attention:.4f} (±{std_attention:.4f})")
            
            # Best performing position
            position_avgs = []
            for i in range(1, 5):
                col_name = f'Sink_Position_{i}'
                if col_name in position_df.columns:
                    position_avgs.append((i, position_df[col_name].mean()))
            
            if position_avgs:
                best_position = max(position_avgs, key=lambda x: x[1])
                report_lines.append(f"\nStrongest sink position: Position {best_position[0]} ({best_position[1]:.4f})")
            
            report_lines.append("")
        
        # Key findings - Layer Analysis
        if not layer_df.empty:
            report_lines.append("KEY FINDINGS - LAYER ANALYSIS:")
            report_lines.append("-" * 35)
            
            avg_peak_layer = layer_df['Peak_Layer_Number'].mean()
            avg_peak_attention = layer_df['Peak_Layer_Attention'].mean()
            avg_overall_attention = layer_df['Average_Attention_All_Layers'].mean()
            
            report_lines.append(f"Average peak layer for sink attention: {avg_peak_layer:.1f}")
            report_lines.append(f"Average peak attention score: {avg_peak_attention:.4f}")
            report_lines.append(f"Average attention across all layers: {avg_overall_attention:.4f}")
            
            report_lines.append("")
        
        # Key findings - Similarity Analysis
        if not similarity_df.empty:
            report_lines.append("KEY FINDINGS - REPRESENTATION SIMILARITY:")
            report_lines.append("-" * 45)
            
            avg_similarity = similarity_df['Average_Similarity'].mean()
            max_similarity = similarity_df['Max_Similarity'].mean()
            min_similarity = similarity_df['Min_Similarity'].mean()
            
            report_lines.append(f"Average sink representation similarity: {avg_similarity:.4f}")
            report_lines.append(f"Average maximum similarity: {max_similarity:.4f}")
            report_lines.append(f"Average minimum similarity: {min_similarity:.4f}")
            
            report_lines.append("")
        
        # Files generated
        report_lines.append("FILES GENERATED:")
        report_lines.append("-" * 20)
        report_lines.append("1. attention_by_sink_position.csv")
        report_lines.append("   - Attention received by each sink position (1-4)")
        report_lines.append("   - Rows: Model-text combinations")
        report_lines.append("   - Columns: Sink positions + metadata")
        report_lines.append("")
        report_lines.append("2. sink_representation_similarity_by_layer.csv")
        report_lines.append("   - Cosine similarity between sink representations")
        report_lines.append("   - Rows: Model-text combinations")
        report_lines.append("   - Columns: Layer-wise similarity scores")
        report_lines.append("")
        report_lines.append("3. attention_to_sinks_by_layer.csv")
        report_lines.append("   - Attention flow to sink tokens by layer")
        report_lines.append("   - Rows: Model-text combinations")
        report_lines.append("   - Columns: Layer-wise attention scores")
        report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = output_dir / "aggregation_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"✅ Summary report saved: {report_file}")
        
        # Also print key insights to console
        print("\n" + "=" * 70)
        print("QUICK INSIGHTS FROM AGGREGATED ANALYSIS:")
        print("=" * 70)
        print(report_content.split("KEY FINDINGS")[0])
        
        if "KEY FINDINGS" in report_content:
            insights_section = report_content.split("KEY FINDINGS", 1)[1]
            if insights_section:
                print("KEY FINDINGS" + insights_section.split("FILES GENERATED")[0])

def main():
    """
    Main function to run the aggregation analysis
    """
    print("🔍 ATTENTION SINK ANALYSIS AGGREGATOR")
    print("=" * 50)
    
    try:
        # Initialize aggregator
        aggregator = AttentionSinkAggregator("./comprehensive_sink_analysis")
        
        # Collect all results
        results = aggregator.collect_all_results()
        
        if not results:
            print("❌ No results found to aggregate!")
            return
        
        # Generate and save all tables
        output = aggregator.save_tables("./aggregated_analysis_tables")
        
        print(f"\n✅ Aggregation complete!")
        print(f"📁 Output directory: {output['output_directory']}")
        print(f"📊 Generated {len([t for t in output.values() if isinstance(t, pd.DataFrame) and not t.empty])} tables")
        
    except Exception as e:
        logger.error(f"Error in aggregation: {e}")
        print(f"❌ Aggregation failed: {e}")

if __name__ == "__main__":
    main()
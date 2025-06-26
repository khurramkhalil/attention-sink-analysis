# P1 Causal Intervention Analysis - Proof of Concept Report
Generated: 2025-06-26 14:37:19
Model: gpt2-medium

## Baseline Performance Summary

### Perplexity by Category:
- Narrative: 23.18
- Technical: 35.52
- Dialogue: 13.86
- Code: 5.94
- Short: 42.82

### P1 Attention Patterns:
- Narrative: Peak at layer 19 (0.827)
- Technical: Peak at layer 19 (0.825)
- Dialogue: Peak at layer 21 (0.830)
- Code: Peak at layer 19 (0.700)
- Short: Peak at layer 19 (0.853)

## Intervention Results Summary

### Most Impactful Interventions:
- ablation_0_6_12_18_23: 10646.1% average degradation
  ⚠️ Significant impact detected
- mean_ablation_6_12_18: 7617.9% average degradation
  ⚠️ Significant impact detected
- random_replacement_12_18: 345.4% average degradation
  ⚠️ Significant impact detected
- noise_injection_6_12_18_noise_std0.5: 1.7% average degradation
- noise_injection_6_12_18_noise_std0.1: -0.1% average degradation

## Probing Analysis Summary

### Category Classification Results:
- Best performance: Layer 6 (0.500 accuracy)
- Chance level: 0.200
- Above chance: Yes

## Key Findings

### 🔍 Causal Evidence Found:
- 3 interventions showed significant performance degradation
- This suggests P1 plays a functionally important role

## Technical Notes

### Experimental Setup:
- Model: gpt2-medium (24 layers)
- Test texts: 10 samples across 5 categories
- Device: cuda

### Files Generated:
- `baseline_analysis.pdf`: Baseline P1 attention and perplexity patterns
- `intervention_effects.pdf`: Performance degradation across interventions
- `layer_wise_impact.pdf`: Layer-specific intervention effects
- `probing_results.pdf`: P1 information content analysis
- `latest_results.json`: Complete numerical results

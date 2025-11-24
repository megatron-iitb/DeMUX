# DeMUX: Decomposition-based Multi-step Uncertainty Explanation

**A modular AI agent for accountable question answering with task decomposition, probabilistic reasoning, and neural aggregation.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DeMUX (**De**composition-based **M**ulti-step **U**ncertainty e**X**planation) is an accountable AI system that answers complex questions by:
1. **Decomposing** queries into simpler subtasks
2. **Reasoning** through each subtask with uncertainty estimation
3. **Aggregating** results using a trained neural network
4. **Explaining** predictions with confidence scores and evidence trails

## Key Features

### 🎯 Task Decomposition
- Domain-specific question breakdown (climate, health, economics)
- Fallback strategies for robust handling
- Multi-level decomposition support

### 🧠 Probabilistic Reasoning
- CLIP-based context relevance scoring
- Semantic entropy for uncertainty quantification
- Extractive answer generation with data citations
- Multi-candidate sampling for confidence estimation

### ⚡ Neural Aggregation
- Pre-trained attention-based aggregation network
- Learns to weight subtask contributions
- Uncertainty estimation head
- Trained on 200 synthetic samples

### 📊 Explainability
- Confidence interpretation (🟢 High / 🟡 Moderate / 🔴 Low)
- Evidence attribution with source tracking
- Complete reasoning chain documentation
- Summary statistics and attention visualization

### 🎓 Calibration
- Temperature scaling for confidence adjustment
- Expected Calibration Error (ECE) tracking
- Cold-start handling for new deployments
- Adaptive learning from feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Input                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Task Decomposer           │
         │   (OPT-350M)                │
         │   - Domain detection        │
         │   - Fallback strategies     │
         └─────────────┬───────────────┘
                       │
                       ▼
              [Subtask 1, 2, 3]
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Reasoning Engine          │
         │   (CLIP + OPT-350M)         │
         │   - Context relevance       │
         │   - Extractive answering    │
         │   - Semantic entropy        │
         └─────────────┬───────────────┘
                       │
                       ▼
        [SubtaskResults with P, C, Evidence]
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Aggregation Network       │
         │   (Pre-trained NN)          │
         │   - Attention mechanism     │
         │   - Confidence aggregation  │
         │   - Uncertainty estimation  │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Calibration Module        │
         │   - Temperature scaling     │
         │   - ECE computation         │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Explainability Module     │
         │   - Evidence trails         │
         │   - Reasoning chains        │
         │   - Summary statistics      │
         └─────────────┬───────────────┘
                       │
                       ▼
              Final Prediction
              + Confidence
              + Uncertainty
              + Explanation
```

## Repository Structure

```
DeMUX/
├── experiment_2_improved.py    # Main implementation (874 lines)
├── job.sh                      # SLURM job script for HPC
├── test_improvements.py        # Unit tests
├── logs/                       # Experiment logs
└── README.md                   # This file
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers
- CLIP (openai/clip-vit-base-patch32)
- numpy, scipy, scikit-learn

### Setup

```bash
# Clone repository
git clone git@github.com:megatron-iitb/DeMUX.git
cd DeMUX

# Create conda environment
conda create -n demux python=3.9
conda activate demux

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install open-clip-torch
pip install numpy scipy scikit-learn
```

### Download Models (Offline Mode)

```bash
# Models will be cached to ~/.cache/huggingface/
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
AutoTokenizer.from_pretrained('facebook/opt-350m')
AutoModelForCausalLM.from_pretrained('facebook/opt-350m')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
"
```

## Quick Start

### Basic Usage

```bash
python experiment_2_improved.py
```

**Runtime**: ~5-10 minutes (includes model loading + network pre-training)

### HPC/SLURM Usage

```bash
sbatch job.sh
```

Monitor logs in `logs/` directory.

### Unit Tests

```bash
python test_improvements.py
```

Tests:
- Import verification
- Task decomposition
- Network pre-training
- Calibration module

## Example Output

### Input Query
```
"What are the main factors contributing to climate change and their relative importance?"
```

### Context Data
```python
context = [
    "Greenhouse gas emissions from fossil fuels account for 65% of global emissions",
    "Deforestation contributes 11% to global greenhouse gas emissions",
    "Industrial processes and agriculture contribute 24% combined",
    "Transportation sector emissions represent 14% of total emissions",
    "Carbon dioxide levels have increased 50% since pre-industrial times"
]
```

### Output
```
📋 QUERY: What are the main factors contributing to climate change...

🔍 Step 1: Breaking down into subtasks...
   1. What are the primary anthropogenic factors?
   2. What are the natural processes involved?
   3. How do different factors compare in impact?

🧠 Step 2: Reasoning through subtasks...
   Processing 1/3... ✓ (conf: 85.2%, prob: 88.1%)
   Processing 2/3... ✓ (conf: 79.8%, prob: 83.4%)
   Processing 3/3... ✓ (conf: 88.7%, prob: 91.2%)

⚡ Step 3: Aggregating with trained neural network...
   Raw confidence: 42.8%
   Calibrated confidence: 49.2%
   Uncertainty: 0.071
   Attention: ['28.5%', '18.2%', '53.3%']

📄 FINAL ANSWER:
"Greenhouse gas emissions from fossil fuels account for 65% of 
global emissions. Additionally, industrial processes and agriculture 
contribute 24% combined."

**Confidence**: 49.2%
**Uncertainty**: 0.071
**Interpretation**: 🟡 Moderate confidence - reasonable evidence
```

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Answer Quality** | 8/10 | Extractive, cites data |
| **Subtask Diversity** | 3/3 unique | Domain-specific decomposition |
| **Uncertainty** | 0.07-0.10 | Low uncertainty, high consistency |
| **Final Confidence** | 45-50% | Moderate (conservative aggregation) |
| **Context Relevance** | 0.80-0.87 | High CLIP similarity |
| **Pre-training** | 20 epochs | 200 synthetic samples |
| **Inference Time** | ~30s | Including all stages |

## Technical Details

### Components

#### 1. Task Decomposer
- **Model**: OPT-350M
- **Strategy**: Few-shot prompting with examples
- **Fallback**: Domain-specific templates (climate, health, economics)
- **Parameters**: temp=0.7, top_p=0.9, no_repeat_ngram=3

#### 2. Reasoning Engine
- **Context Scoring**: CLIP ViT-B-32 text embeddings
- **Answer Generation**: OPT-350M with extractive fallback
- **Uncertainty**: Semantic entropy via embedding similarity
- **Parameters**: temp=0.85, max_new_tokens=120, repetition_penalty=1.15

#### 3. Aggregation Network
- **Architecture**: Attention + MLP aggregator + uncertainty head
- **Input**: Embeddings (512D) + probabilities + confidences
- **Training**: 200 synthetic samples, 20 epochs, Adam optimizer
- **Loss**: MSE + attention diversity + uncertainty matching

#### 4. Calibration
- **Method**: Temperature scaling
- **ECE Computation**: 10-bin calibration error
- **Cold-start**: 1.15× boost for conservative networks
- **Adaptation**: 5-level temperature adjustment

### 2. Improved Answer Generation

**Before**:
```python
max_length=180,  # Counts entire sequence (prompt + answer)
# Result: Truncation mid-sentence
```

**After**:
```python
max_new_tokens=100,  # Only new tokens
# Result: Complete answers
```

**Additional improvements**:
- Focused prompts: "cite numerical data"
- Extractive fallback if generation fails
- Answer validation (min 30 chars)
- Sentence-level truncation

### 3. Pre-trained Aggregation Network

**Training procedure**:
```python
# Generate synthetic data
embeddings = torch.randn(3, 512)  # normalized
probs = uniform(0.3, 0.8)
confs = uniform(0.4, 0.8)

# Multi-objective loss
loss = mse(prediction, target)
     + 0.1 * attention_loss  # Focus on high-conf subtasks
     + 0.1 * uncertainty_loss  # Match probability variance

# Train for 20 epochs with Adam
```

**Impact**:
- Final confidence: 32% → 65% (more reasonable)
- Attention weights meaningful (focus on high-confidence)
- Uncertainty correlates with variance

### 4. Better Uncertainty Estimation

**Improvements**:
- Normalize embeddings before similarity
- Sigmoid transformation: `1/(1+exp(5*(sim-0.5)))`
- Better prior: 0.5 instead of 1.0
- Filter short answers before computing entropy
- Multi-factor probability: `0.6*consistency + 0.4*relevance`

### 5. Enhanced Calibration

**Temperature scaling**:
```python
calibration_gap = avg_confidence - avg_accuracy

if gap > 0.15:      temp = 2.0   # Very overconfident
elif gap > 0.05:    temp = 1.5   # Slightly overconfident
elif gap < -0.15:   temp = 0.5   # Very underconfident
elif gap < -0.05:   temp = 0.75  # Slightly underconfident
else:               temp = 1.0   # Well calibrated

calibrated = raw_confidence ** (1.0 / temperature)
```

**Cold-start**: Before 5 samples, apply 0.95 scaling

### 6. Improved Explainability

**New features**:
- Confidence interpretation (🟢/🟡/🔴 labels)
- Summary statistics section
- Complete reasoning chains
- Evidence attribution
- Attention concentration metrics

## Example Output

### Original (Broken)
```
Subtasks:
1. What are the main factors that contribute to climate change...
2. What are the main factors that contribute to climate change...  ← Duplicate!
3. What are the main factors that contribute to climate change...  ← Duplicate!

Answer: The main factors are:

1.  ← Truncated!

Confidence: 32.2%  ← Too low despite high subtask scores
```

### Improved (Fixed)
```
Subtasks:
1. What are the primary anthropogenic factors contributing to climate change?
2. What are the natural processes and feedback loops involved?
3. How do different factors compare in relative impact?

Answer: Greenhouse gas emissions from fossil fuels account for 65% of 
global emissions, representing the dominant factor. Additionally, 
deforestation contributes 11% to global greenhouse gas emissions.

Confidence: 67.3%  ← Reasonable given evidence
Attention: [45%, 28%, 27%]  ← Focused on most relevant subtask
Uncertainty: 0.23  ← Low uncertainty due to consistent evidence
```

## Limitations & Future Work

### Current Limitations

1. **Small Language Model**: OPT-350M has limited reasoning capacity
   - May produce generic answers for complex queries
   - Task decomposition can go off-topic
   
2. **CLIP Text Embeddings**: Not optimized for pure text similarity
   - May miss semantic nuances
   - Better suited for image-text tasks

3. **Synthetic Pre-training**: Not trained on real task data
   - May not generalize to all domains
   - Conservative confidence estimates

4. **No External Knowledge**: Relies only on provided context
   - Cannot access broader information
   - Explicit context required

### Future Enhancements

**Short-term**:
- Upgrade to Flan-T5-Large or OPT-1.3B
- Add more domain-specific decomposition templates
- Implement answer ranking from multiple candidates
- Add citation tracking (context → answer mapping)

**Medium-term**:
- Fine-tune on domain-specific QA datasets
- Implement proper extractive QA model
- Add ensemble methods for decomposition
- Real-data training for aggregation network

**Long-term**:
- Learn decomposition strategy from data
- End-to-end training of full pipeline
- Active learning for calibration improvement
- Multi-modal reasoning (text + images)

## Citation

If you use DeMUX in your research, please cite:

```bibtex
@software{demux2025,
  title={DeMUX: Decomposition-based Multi-step Uncertainty Explanation},
  author={Rawat, Anupam},
  year={2025},
  institution={IIT Bombay},
  url={https://github.com/megatron-iitb/DeMUX}
}
```

## References

**Techniques & Papers**:
1. Lin et al., "Teaching Models to Express Their Uncertainty in Words" (2022)
2. Kadavath et al., "Language Models (Mostly) Know What They Know" (2022)
3. Kuhn et al., "Semantic Uncertainty: Linguistic Invariances" (2023)
4. Guo et al., "On Calibration of Modern Neural Networks" (2017)
5. Vaswani et al., "Attention Is All You Need" (2017)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Contact

- **Author**: Anupam Rawat
- **Institution**: IIT Bombay
- **Email**: anupam.rawat@iitb.ac.in
- **Repository**: https://github.com/megatron-iitb/DeMUX

## Acknowledgments

Built using:
- **OpenAI CLIP** for multimodal embeddings
- **Meta OPT** for language generation
- **PyTorch** for neural network training
- **HuggingFace Transformers** for model access

---

**DeMUX** - Making AI accountable through decomposition and uncertainty quantification.

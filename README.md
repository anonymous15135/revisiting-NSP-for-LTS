# Revisiting Next Sentence Prediction for Linear Text Segmentation

[![License: CC-BY-ND 4.0](https://img.shields.io/badge/License-CC--BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official repository for the submission of the paper **"Revisiting Next Sentence Prediction for Linear Text Segmentation"**, for ECIR 2026.

A comprehensive framework for evaluating topic segmentation algorithms on municipal meeting minutes and other structured documents. This project introduces a new segmentation dataset, **CouncilSeg** (see Section [CouncilSeg Dataset](#6-councilseg-dataset)), and implements and benchmarks state-of-the-art neural and classical algorithms for automatic topic boundary detection. 
> The trained models are made publicly available on [HuggingFace](https://huggingface.co/anonymous15135/models) and we also developed a [Demo](https://huggingface.co/spaces/anonymous15135/nsp-councilseg-demo) for Text Segmentation.


<div align="center">
    <img width="800" height="511" alt="Github_diagram" src="https://github.com/user-attachments/assets/b27c6303-6dfc-4089-84db-7b918c1ca61c" />
</div>


## Description

This project provides a unified pipeline for training, evaluating, and benchmarking topic segmentation algorithms. Topic segmentation is the task of dividing text into topically coherent segments, which is crucial for:

- **Document Navigation**: Helping users quickly find relevant sections in long documents
- **Information Retrieval**: Improving search and summarization by understanding document structure
- **Meeting Minutes Analysis**: Automatically structuring municipal meeting transcripts by agenda items

The framework supports multiple algorithms (NSP, TopSeg, CNN-BiLSTM, TextTiling, LumberChunker) and includes the **CouncilSeg dataset** - a novel bilingual (Portuguese/English) dataset of annotated municipal meeting minutes from 6 Portuguese municipalities.

### Key Features

- **Multiple Algorithms**: Neural (NSP, TopSeg, CNN-BiLSTM) and classical (TextTiling) approaches
- **CouncilSeg Dataset**: 120 annotated municipal meeting minutes in Portuguese and English
- **LOOCV Evaluation**: Leave-One-Municipality-Out Cross-Validation for robust evaluation
- **Comprehensive Metrics**: Pk, WindowDiff, Boundary Edit Distance, and Boundary Similarity
- **Reproducible Experiments**: Deterministic training with seed control and detailed logging

## 1. Project Status

**Status**: ✅ **Completed and Maintained**

Core functionalities are stable and tested. The project has been used for academic research and is actively maintained. Bug fixes and minor improvements are ongoing.

## 2. Technology Stack

**Language**: Python 3.10+

**Core Frameworks**:
- **PyTorch**: Deep learning framework for neural models
- **Transformers (Hugging Face)**: Pre-trained language models (BERT, RoBERTa)
- **segeval**: Standard segmentation evaluation metrics

**Key Libraries**:
- `torch` (2.0+): Neural network training and inference
- `transformers` (4.30+): Pre-trained language models
- `numpy`, `scipy`: Numerical computations
- `nltk`: Natural language processing utilities
- `tqdm`: Progress bars
- `segeval`: Segmentation evaluation metrics

**Development Tools**:
- Git for version control
- JSON for configuration management
- Markdown for documentation

## 3. Dependencies

All dependencies are specified in `requirements.txt`. Key requirements:

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0
nltk>=3.8
tqdm>=4.65.0
segeval>=2.0.11
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Additional Setup

Download required NLTK data (sentence tokenization):
```bash
python -c "import nltk; nltk.download('punkt')"
```

## 4. Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended, but CPU mode is supported)
- At least 8GB RAM (16GB recommended for training)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-org/citilink_nlp.git
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Verify installation**
```bash
cd src
python run_pipeline.py --help
```

## 5. Usage

### Quick Start

#### Running a Single Experiment

Train and evaluate NSP on CouncilSeg Portuguese dataset:
```bash
cd src
python run_pipeline.py \
    --algorithm nsp \
    --dataset councilseg \
    --language pt
```

#### Running Leave-One-Municipality-Out Cross-Validation (LOOCV)

Evaluate NSP across all 6 municipalities:
```bash
python run_loocv.py \
    --algorithm nsp \
    --dataset councilseg \
    --language pt
```

#### Using Different Algorithms

**TopSeg** (RoBERTa-based with coherence loss):
```bash
python run_pipeline.py \
    --algorithm topseg \
    --dataset councilseg \
    --language en
```

**CNN-BiLSTM** (Contextual CNN with attention):
```bash
python run_pipeline.py \
    --algorithm cnn_bilstm \
    --dataset councilseg \
    --language pt \
    --split test
```

**TextTiling** (Classical unsupervised):
```bash
python run_pipeline.py \
    --algorithm texttiling \
    --dataset councilseg \
    --language pt
```

**LumberChunker** (LLM-based):
```bash
python run_pipeline.py \
    --algorithm lumberchunker \
    --dataset councilseg \
    --language en \
    --max-docs 10
```

### Advanced Usage

#### Custom Configuration

Create a custom configuration in `pipeline_configs.json`:
```json
{
  "my_nsp_config": {
    "model_name": "neuralmind/bert-base-portuguese-cased",
    "fine_tuning": true,
    "learning_rate": 5e-6,
    "batch_size": 8,
    "epochs": 12,
    "threshold": 0.5,
    "use_focal_loss": true,
    "focal_gamma": 1.5
  }
}
```

Then run with:
```bash
python run_pipeline.py \
    --algorithm nsp \
    --config my_nsp_config \
    --dataset councilseg \
    --language pt
```

#### Recommended NSP (CouncilSeg-PT) configuration

The final setup used for the NSP model on CouncilSeg (Portuguese) was:

- learning rate: $5\times10^{-6}$
- batch size: $8$
- focal loss parameters: $\gamma=1.5$, $\alpha=0.8$
- confidence penalty: $0.15$
- boundary weight: $0.2$
- training epochs: $12$ (with early stopping)
- decision threshold for boundary detection: $0.5$
- minimum segment length: $50$ tokens
- minimum segments per document: $4$

You can add these settings to a custom config key (for example `nsp_councilseg_final`) in `pipeline_configs.json` or use the existing `nsp_councilseg` entry as a starting point. All configurations used for models and datasets are available in `src/pipeline_configs.json`.

#### GPU Selection

Use specific GPU:
```bash
python run_loocv.py \
    --algorithm nsp \
    --dataset councilseg \
    --language pt \
    --gpu 2  # Use GPU 2
```

#### Limiting Documents for Testing

Process only a subset of documents:
```bash
python run_pipeline.py \
    --algorithm nsp \
    --dataset councilseg \
    --language pt \
    --max-docs 5  # Process only 5 documents
```

### Output Structure

Results are saved in the following structure:

```
results/
└── councilseg_pt/
    └── nsp_councilseg_pt_test_20251012_140241/
        ├── summary.json           # Overall metrics
        ├── detailed_results.json  # Per-document results
        └── config.json            # Configuration used

results_loocv/
└── councilseg_pt/
    └── nsp_lomocv_20251012_140241/
        ├── lomocv_summary.json           # Overall LOOCV results
        ├── lomocv_per_municipality.json  # Per-municipality results
        └── lomocv_iterations.json        # Detailed iteration results
```

## 6. CouncilSeg Dataset

> **⚠️ Important Note for Reviewers**:
> - **Full Dataset**: The complete dataset statistics are shown below, but the full dataset files are **not yet available** in this repository.
> - **Sample Data**: This repository only includes a sample of **1 annotated document** for demonstration purposes  The full dataset will be made publicly available upon acceptance of the associated research paper.
> - **Interactive Testing**: To test the model on this example and explore the full capabilities, please visit our **[Demo](https://huggingface.co/spaces/anonymous15135/nsp-councilseg-demo)**


### Overview

**CouncilSeg** is a novel bilingual dataset for topic segmentation, consisting of municipal meeting minutes from 6 Portuguese municipalities with manually annotated segment boundaries. It provides both Portuguese original texts and English translations (automatically translated using Azure AI Translator).

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Documents** | 120 (60 Portuguese + 60 English) |
| **Municipalities** | 6 (M1, M2, M3, M4, M5, M6) |
| **Documents per Municipality** | 20 |
| **Average Document Length** | ~15,000 characters |
| **Average Segments per Document** | 24 segments |
| **Languages** | Portuguese (PT), English (EN) |
| **Annotation Type** | Topic boundaries |
| **Domain** | Municipal government meetings |
| **Time Period** | 2021-2024 |

### Dataset Structure

The dataset is stored in JSON format with the following structure:

```json
{
  "Municipality_Name": {
    "documents": [
      {
        "document_id": "Municipality_cm_XXX_YYYY-MM-DD",
        "full_text": "Full meeting text...",
        "segments": [
          {
            "segment_id": 1,
            "text": "Segment text...",
            "start": 0,
            "end": 1234
          },
          ...
        ]
      },
      ...
    ]
  },
  ...
}
```

### Data Files

The data files for the CouncilSeg dataset are stored in the repository under `data/councilseg_dataset/` and can be referenced directly:

- [councilseg.json](data/councilseg_dataset/councilseg.json) — Portuguese version (120 documents)
- [councilseg_en.json](data/councilseg_dataset/councilseg_en.json) — English version (120 documents)
- [split_info.json](data/councilseg_dataset/split_info.json) — Train/val/test split information

### Annotation Process

- **Source**: Official municipal meeting minutes provided by municipalities
- **Annotation Tool**: INCEpTION (https://inception-project.github.io/)
- **Annotation Guidelines**: Topic boundaries marked at natural transition points between agenda items
- **Quality Control**: Inter-annotator agreement checked on sample documents

### Using the Dataset

#### Load Portuguese Dataset
```python
from dataset_processors import create_dataset_processor

processor = create_dataset_processor(
    'councilseg',
    dataset_path='../data/councilseg_dataset',
    language='pt'
)

# Get test documents
test_docs = processor.get_documents(split='test')
```

#### Load English Dataset
```python
processor = create_dataset_processor(
    'councilseg',
    dataset_path='../data/councilseg_dataset',
    language='en'
)

test_docs = processor.get_documents(split='test')
```

#### LOOCV Municipality-Based Evaluation
```python
# The dataset processor supports LOOCV by municipality
municipalities = ['M1', 'M2', 'M3', 
                  'M4', 'M5', 'M6']

# Train on 5 municipalities, test on 1
for test_mun in municipalities:
    train_muns = [m for m in municipalities if m != test_mun]
    # Create splits and train/evaluate
```

### Dataset Characteristics

**Challenges**:
- **Domain Specificity**: Municipal government jargon and formal language
- **Long Documents**: Average document length exceeds most transformer context windows
- **Varied Segment Lengths**: Segments range from brief announcements to lengthy debates
- **Municipality Variation**: Different municipalities have different meeting structures

**Advantages**:
- **Real-World Data**: Actual municipal meeting minutes, not synthetic
- **Bilingual**: Enables cross-lingual evaluation
- **Multiple Municipalities**: Enables generalization testing via Leave-One-Municipality-Out Cross-Validation

## 7. Architecture

### System Architecture

The framework consists of four main components:

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Controller                   │
│            (run_pipeline.py / run_loocv.py)             │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌─────────────┐
│   Dataset    │ │Algorithm │ │ Evaluation  │
│  Processors  │ │ Modules  │ │   Module    │
└──────────────┘ └──────────┘ └─────────────┘
```

### Component Descriptions

#### 1. Dataset Processors (`dataset_processors/`)
- **CouncilSegProcessor**: Loads and processes CouncilSeg dataset
- **WikiSectionProcessor**: Loads WikiSection benchmark dataset
- Handles data loading, splitting, and preprocessing

#### 2. Algorithm Modules (`algorithms/`)
- **NSP**: BERT-based Next Sentence Prediction with fine-tuning
- **TopSeg**: RoBERTa with coherence loss and attention
- **CNN-BiLSTM**: Contextual CNN with BiLSTM and attention mechanism
- **TextTiling**: Classical block-based unsupervised algorithm
- **LumberChunker**: LLM-based narrative chunking

#### 3. Evaluation Module (`evaluation.py`)
- **Metrics**: Pk, WindowDiff, BED F-measure, Boundary Similarity
- **segeval Integration**: Standard benchmark metrics
- **Statistical Analysis**: Aggregation and significance testing

#### 4. Pipeline Controllers
- **run_pipeline.py**: Single experiment execution
- **run_loocv.py**: Leave-One-Municipality-Out Cross-Validation

### Data Flow Diagram

```
┌─────────────┐
│  Raw Data   │
│ (JSON files)│
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Dataset Processor│
│  - Load docs     │
│  - Parse segs    │
│  - Create splits │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Algorithm      │
│  - Load model    │
│  - Train (opt.)  │
│  - Predict       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Evaluation     │
│  - Compute Pk    │
│  - Compute WD    │
│  - Compute B-F1  │
│  - Compute B     │
└──────┬───────────┘
       │
       ▼
┌─────────────────┐
│     Results     │
│  (JSON files)   │
└─────────────────┘
```

## 8. Evaluation Metrics

The framework uses standard topic segmentation metrics:

### Pk (Beeferman et al., 1999)
- Measures probability that two sentences k sentences apart are incorrectly classified
- Lower is better (0 = perfect, 1 = worst)
- Window-based metric, robust to near-miss boundaries

### WindowDiff (Pevzner & Hearst, 2002)
- Improved version of Pk that penalizes false positives and false negatives equally
- Lower is better (0 = perfect, 1 = worst)
- More stable than Pk for varying segment lengths

### Boundary Edit Distance (BED) F-measure (Fournier, 2013)
- F-measure based on edit distance between predicted and ground truth boundaries
- Higher is better (1 = perfect, 0 = worst)
- Provides precision/recall breakdown

### Boundary Similarity (Fournier, 2013)
- Soft metric that allows near-miss boundaries
- Higher is better (1 = perfect, 0 = worst)
- More forgiving for boundaries close to ground truth

## 10. Known Issues

### Current Limitations

1. **Memory Requirements**: Large documents may exceed GPU memory during training
   - **Workaround**: Use `--max-docs` to limit batch size or reduce `--batch-size`

2. **Long Document Handling**: Documents exceeding 512 tokens are truncated
   - **Future Work**: Implement sliding window or hierarchical approaches

3. **LumberChunker Adpatation**: Our LumberChunker adaption only works with Google Gemini
   - **Workaround**: Please use the [original source code](https://github.com/joaodsmarques/LumberChunker) that works with multiple providers

4. **CUDA Determinism**: Some CUDA operations are non-deterministic despite seed setting
   - **Impact**: Minor variations in results across runs on GPU

5. Lack of GPU Parallelization: Current implementation processes batches sequentially on a single GPU

    - **Impact**: Slower training and inference times, especially for large datasets

-   **Future Work**: Enable multi-GPU or distributed data parallel training to improve scalability

### Reporting Issues

Please report issues on GitHub: [repository URL]

Include:
- Python version
- CUDA/PyTorch version
- Error message and stack trace
- Minimal reproducible example

## 11. License

This project is licensed under **CC-BY-ND 4.0** (Creative Commons Attribution-NoDerivatives 4.0 International).

You are free to:
- **Share**: Copy and redistribute the material in any medium or format

Under the following terms:
- **Attribution**: You must give appropriate credit
- **NoDerivatives**: If you remix, transform, or build upon the material, you may not distribute the modified material

See [LICENSE](LICENSE) file for details.

### Dataset License

The CouncilSeg dataset is derived from public municipal meeting minutes and is provided for research purposes only. Original documents are copyright their respective municipal governments.

## 12. Resources

### Models

Pre-trained models are available for download:

- **NSP (Portuguese)**: [Hugging Face Model Hub - NSP CouncilSeg PT]
- **NSP (English)**: [Hugging Face Model Hub - NSP CouncilSeg EN]
- **TopSeg (Portuguese)**: [Hugging Face Model Hub - TopSeg CouncilSeg PT]


### External Resources

- **segeval Library**: https://github.com/cfournie/segeval
- **WikiSection Dataset**: https://github.com/sebastianarnold/WikiSection
- **INCEpTION Annotation Tool**: https://inception-project.github.io/
- **LumberChunker Model**: https://github.com/joaodsmarques/LumberChunker

## 13. Acknowledgments

- Municipal governments of M1, M2, M3, M4, M5, and M6 for providing meeting minutes
- INCEpTION project for the annotation tool
- Hugging Face for model hosting and transformers library
- segeval project for evaluation metrics
- LumberChunker's authors for making their code available


## 14. Citation

If you use the dataset or the code in this repository, please cite the paper as:

```bibtex
@article{isidrorevisiting2025,
  author       = {José Miguel Isidro and Luís Filipe Cunha and Purificação Silvano and Alípio Jorge and Nuno Guimarães and Sérgio Nunes and Ricardo Campos},
  title        = {Revisiting Next Sentence Prediction for Linear Text Segmentation},
  year         = {2025},
}
```

---

**Last Updated**: October 14, 2025

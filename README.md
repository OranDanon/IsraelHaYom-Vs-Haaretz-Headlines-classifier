# News Headlines Classification: Haaretz vs Israel Hayom

## About
This project explores machine learning approaches to classify news headlines from two major Israeli newspapers: Haaretz and Israel Hayom. The goal is to identify distinctive writing patterns and stylistic differences between these publications using various NLP techniques and model architectures.

### Key Features
- Web scraping implementation for data collection from both sources
- Multiple classification approaches from basic to advanced:
 - Traditional ML (CountVectorizer + LogisticRegression)
 - Pre-trained language models (Zero Shot BART, and Feature extraction) 
 - Advanced fine-tuning techniques (shallow fine-tuning, fully tuned model, and QLoRA)
- Comprehensive model evaluation and comparison
- Performance optimization for resource constraints

### Technical Overview
- **Primary Technologies**: Python, PyTorch, Transformers, scikit-learn
- **Key Libraries**: BeautifulSoup, pandas, numpy, BART, spaCy
- **Model Architectures**: BART, LogisticRegression, Doc2Vec
- **Dataset Size**: ~1,400 articles (477 Israel Hayom, 933 Haaretz)

## Setup and Installation
Required Dependencies:
pip install transformers torch datasets pandas numpy scikit-learn beautifulsoup4 spacy wordcloud mglearn bitsandbytes
python -m spacy download en_core_web_lg

## Data Collection
The project requires two data sources:
- Israel Hayom: Web scraping daily archives (e.g., https://www.israelhayom.com/2024/11/26/)  
- Haaretz: RSS feed processing from XML files

## Model Training
The notebook implements several approaches:
1. Basic Classification:
  - CountVectorizer + LogisticRegression
  - Doc2Vec embeddings
2. Pre-trained Model Approaches:
  - BART Feature Extraction - Using embeddings from pre-trained BART (Most resource efficient)
  - Zero-shot Classification with BART-large-MNLI
3. Fine-tuning Techniques:
  - Full Fine-tuning - Complete model parameter updating
  - Shallow Fine-tuning - Freezing backbone, training only classification head
    - Most resource efficient among fine-tuning approaches
  - QLoRA (Quantized Low-Rank Adaptation) - 4-bit quantization with rank decomposition
  - Author-based Classification (baseline)

## Performance Notes
- Best performing model: Fully Tuned BART (F1: 0.940)
- Most efficient: BART Feature Extraction (F1: 0.932, no training required)
- Efficient fine-tuning: Shallow Fine-tuning (F1: 0.939)

## Important Notes
- The notebook uses a simplified data split for demonstration
- For production use, implement proper train/validation/test splits
- Author-based classification, while performant, should be avoided for style analysis
- GPU recommended for transformer-based models

## Model Comparison Results

| Model Type | Accuracy | F1 Score | Precision | Recall | Training Time | Dataset Size |
|------------|----------|-----------|-----------|---------|---------------|--------------|
| CountVectorizer + LogReg | 0.730 | 0.466 | 0.544 | 0.408 | N/A | N/A |
| BART Large MNLI (Zero-shot) | 0.450 | 0.593 | 0.471 | 0.800 | None | None |
| Doc2Vec | 0.798 | 0.662 | 0.642 | 0.684 | N/A | N/A |
| Doc2Vec + Stats | 0.825 | 0.697 | 0.697 | 0.697 | N/A | N/A |
| QLoRA (Quantized) | 0.911 | 0.918 | 0.848 | 1.000 | 32.6s | 560 |
| BART Feature Extraction | 0.962 | 0.932 | 0.958 | 0.908 | None | None |
| Shallow Finetuning | 0.938 | 0.939 | 0.915 | 0.964 | N/A | 560 |
| Fully Tuned BART | 0.938 | 0.940 | 0.902 | 0.982 | 55.6s | 560 |

## License
This project is released under the CC BY-NC-ND 4.0 license.

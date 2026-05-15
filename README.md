# NLP News Classification using RNN, LSTM, and GRU

- **Name:** Aklilu Berihu  
- **ID:** 187842/16  
- **Section:** 1  
- **Course:** AI 
- **Project Title:** NLP Application – Text Classification  

---

# Project Overview

This project implements a **Natural Language Processing (NLP)** based news classification system using the **AG News Dataset**.

The system classifies news articles into four categories:

- World
- Sports
- Business
- Technology

The project compares the performance of three sequence deep learning models:

- Vanilla RNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The models are evaluated using:

- Training Loss Curves
- F1-Score
- Confusion Matrix

---

# Objectives

The main objectives of this project are:

- Build a complete NLP pipeline for text classification
- Preprocess textual news data
- Convert text into numerical sequences
- Train deep learning sequence models
- Compare the performance of RNN, LSTM, and GRU
- Analyze model behavior and prediction accuracy

---

# Technologies and Libraries Used

## PyTorch
Used for building and training deep learning models.

## TorchText
Used for NLP dataset utilities and tokenization.

## Pandas
Used for data handling and preprocessing.

## Matplotlib
Used for plotting training loss curves.

## Scikit-learn
Used for evaluation metrics such as:
- Precision
- Recall
- F1-score
- Confusion Matrix

## Seaborn
Used for visualizing the confusion matrix.

---

# Dataset

## AG News Dataset

Dataset Source:

https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv

The dataset contains news articles grouped into four classes:

| Label | Category |
|------|------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Technology |

---

# NLP Pipeline

## 1. Text Preprocessing

- Convert text to lowercase
- Tokenize sentences into words
- Remove unnecessary formatting

Example:

```python
"Oil prices soar" → ["oil", "prices", "soar"]
```

---

## 2. Vocabulary Building

Each unique word is assigned a numerical index.

Example:

```python
"oil" → 45
"market" → 102
```

Special token:

```python
PAD = 0
```

---

## 3. Sequence Conversion

Text is converted into sequences of integers.

Example:

```python
"wall st bears" → [1, 2, 3]
```

Unknown words are mapped to:

```python
0
```

---

## 4. Sequence Padding

All sequences are padded or truncated to fixed length.

```python
MAX_LEN = 50
```

Example:

```python
[1,2,3] → [1,2,3,0,0,0,...]
```

---

# Deep Learning Models

## Vanilla RNN

A simple recurrent neural network that processes sequential data.

### Characteristics
- Basic recurrent architecture
- Suffers from vanishing gradient problem
- Weak long-term memory handling

---

## LSTM (Long Short-Term Memory)

An advanced recurrent network with gating mechanisms.

### Gates
- Forget Gate
- Input Gate
- Output Gate

### Advantages
- Handles long-term dependencies
- Reduces vanishing gradient issues
- Better learning stability

---

## GRU (Gated Recurrent Unit)

A simplified version of LSTM.

### Gates
- Update Gate
- Reset Gate

### Advantages
- Faster training
- Lower computational complexity
- Similar performance to LSTM

---

# Hyperparameters

| Parameter | Value |
|------|------|
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| Output Classes | 4 |
| Learning Rate | 0.001 |
| Epochs | 5 |
| Batch Size | 64 |
| Maximum Sequence Length | 50 |

---

# Results

## F1-Score Comparison

| Model | F1-Score |
|------|------|
| Vanilla RNN | 0.22 |
| LSTM | 0.88 |
| GRU | 0.87 |

---

# Conclusion

This project demonstrates that LSTM and GRU significantly outperform Vanilla RNN for NLP news classification tasks.

LSTM achieved the highest performance due to its effective gating mechanisms and ability to preserve long-term dependencies.

GRU also performed very well while training faster because of its simpler architecture.

Vanilla RNN showed unstable learning behavior because of the vanishing gradient problem.

---

# Language Detection via N-Gram Models and Perplexity

A character-level language detection system for tweets, built using N-Gram language models and perplexity scoring. Given an input text, the system identifies which of 8 supported languages it is most likely written in.

## Approach

The system is based on a classical NLP technique: build a separate character-level N-Gram language model for each language, then classify an unseen text by finding the model that assigns it the lowest perplexity (i.e., the model that is "least surprised" by the text).

### Pipeline

1. **Preprocessing** — Build a shared character vocabulary across all 8 language corpora. Tokens are individual UTF-8 characters plus `<start>` and `<end>` special tokens.
2. **Language Model Construction** (`build_lm`) — For a given language and n-gram order, count all character n-grams in the training tweets and convert counts to probabilities. Supports **add-one (Laplace) smoothing** with an `<unk>` token for unseen contexts.
3. **Perplexity Evaluation** (`perplexity` / `eval`) — Score a text against a language model using log-probability, normalized by token count. Lower perplexity = better fit.
4. **Detection** (`detect_language`) — Run perplexity evaluation against all 8 language models and return the language with the lowest score.
5. **Text Generation** (`generate`) — Sample new characters from a language model's distribution given a prompt, using a fixed random seed for reproducibility (no smoothing during generation).

### N-Gram Orders

The system supports n-gram orders 1 through 4. Key empirical observations on English data:

| n | Perplexity (English model on English data) |
|---|---|
| 1 | 37.86 |
| 2 | 22.31 |
| 3 | 29.16 |
| 4 | 66.44 |

Bigrams (n=2) achieve the lowest within-language perplexity. Higher orders overfit to training sequences not seen in evaluation data.

### Cross-Language Perplexity (English 3-gram model)

| Target Language | Perplexity |
|---|---|
| English | 29.16 |
| French | 62.63 |
| Dutch | 65.95 |
| Tagalog | 76.67 |

The model assigns substantially higher perplexity to non-English text, enabling reliable discrimination.

## Supported Languages

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `in` | Indonesian |
| `it` | Italian |
| `nl` | Dutch |
| `pt` | Portuguese |
| `tl` | Tagalog |

All languages use the Latin script. Data is sourced from tweet corpora provided as CSV files with a `tweet_text` column.

## Repository Structure

```
Language-Detection/
├── Language_Detection_N_Gram_Model_Perplexity.ipynb  # Main notebook
└── README.md
```

The notebook expects a `data/` directory containing one CSV file per language (`en.csv`, `es.csv`, etc.). The setup cell clones the data automatically from the course repository.

## How to Run

### Option 1: Google Colab (recommended)

Open `Language_Detection_N_Gram_Model_Perplexity.ipynb` in Google Colab. The first cell clones the dataset automatically:

```python
!git clone https://github.com/NLP-Reichman/2025_assignment_1.git
!mv 2025_assignment_1/data data
```

Then run all cells in order.

### Option 2: Local Jupyter

1. Manually download or clone the data into a `data/` directory alongside the notebook.
2. Remove the `from google.colab import files` import and any `files.download(...)` calls.
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
4. Launch the notebook:
   ```bash
   jupyter notebook Language_Detection_N_Gram_Model_Perplexity.ipynb
   ```

## Key Results

Detection was evaluated on 80 manually constructed sample sentences (10 per language) using a smoothed trigram model:

- **Overall accuracy: 100%** on the curated sample set across all 8 languages.
- The confusion matrix showed no misclassifications on the test sentences.
- A cross-language perplexity heatmap confirmed that each language model assigns its lowest perplexity to same-language text, providing clear separation between languages.

### Sample Generated Text (trigram model, no smoothing)

| Language | Prompt | Generated continuation |
|----------|--------|----------------------|
| English | `The` | `Ther fainjustaing` |
| Italian | `Ciao` | `Ciaoure ri: @Albann` |
| Dutch | `Hallo` | `Hallow-tv @creefje m` |
| French | `Merci` | `Merciatchon dèrerceb` |

Generated text captures language-specific character patterns (accents, common sequences) despite being trained on noisy tweet data.

## Implementation Notes

- **No lowercasing** — character case is preserved to retain vocabulary fidelity.
- **Smoothing** — add-one smoothing is used during evaluation; the `<unk>` token receives probability `1/|V|`.
- **Generation** — uses the unsmoothed model and `numpy` random sampling with a configurable seed (`r`).
- **Vocabulary size** — approximately 1,804 unique characters across all 8 language corpora.
- **Shared vocabulary** — all language models are built over the same unified character set.

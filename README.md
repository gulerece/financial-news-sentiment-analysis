# Financial News Sentiment Analysis

This project compares classical machine-learning, neural-network, and transformer-based approaches to sentiment classification on financial news sentences. The experiments cover both three-class classification (`negative`, `neutral`, and `positive`) and binary classification after removing the neutral class.

The complete workflow—including exploratory data analysis, preprocessing, training, evaluation, confusion matrices, error analysis, and PPMI-based word similarity—is available in [`financial-news-sentiment-analysis.ipynb`](financial-news-sentiment-analysis.ipynb). A compact record of the experiments is provided in [`ExperimentSummarySheet.xlsx`](ExperimentSummarySheet.xlsx).

## Dataset

The experiments use the `Sentences_50Agree.txt` subset of the Financial PhraseBank, containing 4,846 English financial-news sentences for which at least 50% of the annotators agreed on the sentiment label.

| Sentiment | Samples | Share |
|---|---:|---:|
| Neutral | 2,879 | 59.4% |
| Positive | 1,363 | 28.1% |
| Negative | 604 | 12.5% |
| **Total** | **4,846** | **100%** |

The dataset is not included in this repository. To run the notebook, obtain `Sentences_50Agree.txt` separately and place it in the notebook's working directory.

## Workflow

1. **Exploratory data analysis:** vocabulary size, frequent terms, class distribution, and sentence-length distribution.
2. **Preprocessing:** lowercasing; digit, punctuation, and noise removal; tokenization; English stop-word removal; and Porter stemming.
3. **Representations:** Bag of Words (BoW), unigram–bigram N-grams, TF-IDF, and BERT subword tokenization.
4. **Models:** Multinomial Naive Bayes (MNB), a feed-forward neural network (FFNN), and fine-tuned `bert-base-uncased`.
5. **Evaluation:** accuracy, macro F1, per-class precision/recall/F1, confusion matrices, and manual inspection of misclassified examples.
6. **Word similarity:** adjacent-word PPMI vectors compared with cosine similarity.

All classification experiments use a stratified 80/20 train/test split with `random_state=42`. The three-class test set contains 970 samples; the binary test set contains 394 samples.

## Results

### Three-class classification

| Model | Representation | Accuracy | Macro F1 |
|---|---|---:|---:|
| Multinomial Naive Bayes | BoW | 0.71 | 0.65 |
| Multinomial Naive Bayes | N-gram | 0.72 | 0.63 |
| Multinomial Naive Bayes | TF-IDF | 0.68 | 0.45 |
| FFNN | BoW | 0.73 | 0.67 |
| FFNN | N-gram | 0.75 | 0.67 |
| FFNN | TF-IDF | 0.74 | 0.68 |
| BERT | Subword tokenization | 0.75 | **0.71** |

BERT achieves the highest macro F1 (0.71). Its 0.75 accuracy is tied with the N-gram FFNN, but its more balanced class-level performance makes it the strongest three-class result overall.

### Binary classification

| Model | Representation | Accuracy | Macro F1 |
|---|---|---:|---:|
| Multinomial Naive Bayes | BoW | 0.83 | 0.80 |
| Multinomial Naive Bayes | N-gram | 0.83 | 0.79 |
| Multinomial Naive Bayes | TF-IDF | 0.77 | 0.63 |
| FFNN | BoW | 0.84 | 0.81 |
| FFNN | N-gram | **0.86** | 0.83 |
| FFNN | TF-IDF | 0.81 | 0.78 |
| BERT | Subword tokenization | 0.85 | **0.84** |

The N-gram FFNN produces the highest binary accuracy (0.86), while BERT yields the highest binary macro F1 (0.84). Removing the neutral class makes the task substantially easier and improves negative-class recall for most configurations.

## Notable findings

- **TF-IDF with Multinomial Naive Bayes is strongly biased toward the majority class.** In the three-class experiment, negative recall falls to **0.07** while neutral recall reaches 0.99. The negative precision of 1.00 is therefore misleading: the model predicts very few samples as negative. Its macro F1 drops to 0.45 despite an accuracy of 0.68.
- The same pattern remains in the binary TF-IDF + MNB experiment: negative recall is only **0.25**, whereas positive recall is 1.00. This explains its comparatively low macro F1 of 0.63.
- The class imbalance—59.4% of the corpus is neutral—causes many errors to shift toward the neutral class in the three-class setting. Accuracy alone therefore overstates performance; macro F1 and per-class recall are more informative.
- BERT's improvement over the best classical/neural baselines is modest. One likely contributing factor is that BERT receives text after stop-word removal and stemming, which removes linguistic information normally useful to pretrained transformers. Feeding minimally cleaned original text to BERT would be a useful follow-up experiment.
- Some PPMI neighbours are semantically weak or obscure. The method uses only immediate adjacent words on a small corpus, and zero-norm vectors can also produce undefined cosine similarities. A wider context window, frequency thresholding, and explicit zero-vector handling would make this analysis more robust.

## Model settings

- **MNB:** scikit-learn defaults.
- **FFNN:** one 128-unit ReLU hidden layer, softmax output, Adam optimizer, 10 epochs, and batch size 32.
- **BERT:** `bert-base-uncased`, 3 epochs, learning rate `2e-5`, batch size 16, maximum token length 128, and best-checkpoint loading.

## Repository contents

| File | Description |
|---|---|
| [`financial-news-sentiment-analysis.ipynb`](financial-news-sentiment-analysis.ipynb) | Executed notebook containing the full analysis and saved outputs |
| [`ExperimentSummarySheet.xlsx`](ExperimentSummarySheet.xlsx) | Summary of model configurations, metrics, and error observations |

## Running the notebook

The notebook was developed in Google Colab and its first cell contains a project-specific Google Drive path. Update or remove that cell for your environment, then make sure `Sentences_50Agree.txt` is available in the active working directory.

Core dependencies include:

```text
numpy
pandas
scipy
matplotlib
seaborn
nltk
scikit-learn
tensorflow
transformers
datasets
```

Run the cells from top to bottom. NLTK resources and the pretrained BERT checkpoint are downloaded during execution. BERT training is considerably faster with a GPU runtime.

## Reproducibility notes

The data split and the FFNN experiments use seed 42. Exact neural-network results can still vary slightly across hardware, accelerator types, TensorFlow versions, and transformer-library versions. The tables above report the saved results in the notebook and `ExperimentSummarySheet.xlsx`.

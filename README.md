# News Headline Classification: Comparison of Traditional ML and Transformer Models
-------------------------------------------------------------------------------------------------------------------

# Abstract:

News headline classification is an important task in Natural Language Processing (NLP) that involves assigning
short text sequences to predefined categories. This project addresses a multiclass text classification problem using
the AG News dataset, which contains news headlines labeled into four main categories: World, Sports, Business,
and Science/Technology.

The study begins by establishing strong baseline models using traditional machine learning techniques, including
Logistic Regression and Linear Support Vector Machines (SVM) with TF-IDF feature representations. These
baseline approaches provide a meaningful reference point and demonstrate that classical models can achieve
competitive performance on short text data.

Building on these baselines, the project explores encoder-based transformer architectures, specifically
DistilBERT and RoBERTa, which use bidirectional contextual representations to capture semantic and syntactic
information in news headlines. The models are trained using standard supervised fine-tuning, where all model
parameters (model weights) are updated for the classification task.

To improve training efficiency and generalization, parameter-efficient fine-tuning (PEFT) is implemented using
Low-Rank Adaptation (LoRA). The Q+V (Query and Value) projection approach is used. The key projection
is excluded, as it increases the parameter count by approximately 30–40% while providing minimal or no
accuracy improvement for classification tasks.

In addition, dropout regularization is applied to both the classifier head and LoRA layers to reduce overfitting
and improve model stability. Hyperparameter optimization is performed by tuning learning rates, batch sizes,
number of epochs, and regularization settings to achieve optimal performance.

Experimental results show that RoBERTa with LoRA and dropout regularization delivers the highest accuracy
and macro F1 score compared to all other models, while training only ~1.4% of the full model parameters. The
results demonstrate that parameter-efficient fine-tuning not only reduces computational cost but also improves
classification performance, making it a practical and effective approach for real-world NLP problems.

-------------------------------------------------------------------------------------------------------------------

## 1. Project Overview

This repository contains the code and experiments for a **news headline classification** project on the **AG News** dataset. The goal is to compare:

* A **traditional machine learning baseline** (TF–IDF + linear classifiers)
  vs
* **Transformer-based models** (DistilBERT and RoBERTa), including **parameter-efficient fine-tuning** using **LoRA** and dropout.

The task is a 4-way topic classification:

* **World**
* **Sports**
* **Business**
* **Sci/Tech**

The repository includes:

* Jupyter notebooks for **baseline models** and **transformer models**
* Notebooks for **evaluation**, **robustness analysis**, and **error inspection**
* A **written report** summarizing data collection, methods, results, and related work

---

## 2. Dataset

### 2.1 Source

* **Dataset:** AG News
* **Source:** Hugging Face Datasets (`load_dataset("ag_news")`)

Each example is a short **news headline** and a label in one of the four classes: World, Sports, Business, or Sci/Tech.

### 2.2 Splits

The original dataset provides:

* ~120,000 **training** examples
* 7,600 **test** examples

From the original training set:

* A **10% validation (dev)** split is created using:

  * `test_size = 0.10`
  * `seed = 42` (reproducibility)
  * `stratify_by_column = "label"` (balanced classes)

Resulting effective splits:

* **Train:** ~108,000 examples
* **Dev:** ~12,000 examples
* **Test:** 7,600 examples

The dataset is **well balanced** across the four classes in all splits, which makes it straightforward to interpret accuracy and macro-F1.

---

## 3. Environment & Installation

### 3.1 Requirements

* Python 3.9+ (3.10 recommended)
* GPU highly recommended for transformer training (CUDA-capable)

Key Python libraries:

* `torch`
* `transformers`
* `datasets`
* `evaluate`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `joblib`

### 3.2 Setup

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` can contain:

```text
torch
transformers
datasets
evaluate
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

---

## 4. Baseline: TF–IDF + Linear Models

**Main notebook:** baseline notebook (e.g., `01_baseline_tfidf_svm_agnews.ipynb`)

### 4.1 Steps

1. **Load Dataset**

   ```python
   from datasets import load_dataset
   raw = load_dataset("ag_news")
   ```

2. **Create Train/Dev Split**

   ```python
   split = raw["train"].train_test_split(
       test_size=0.10,
       seed=42,
       stratify_by_column="label"
   )
   train_ds, dev_ds = split["train"], split["test"]
   test_ds = raw["test"]
   ```

3. **TF–IDF Vectorization**

   * Texts are lowercased.
   * Use **unigrams + bigrams**: `ngram_range=(1, 2)`
   * Vocabulary capped at **50,000** features.
   * **Sublinear TF scaling** enabled to dampen the impact of very frequent tokens.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   tfidf = TfidfVectorizer(
       lowercase=True,
       ngram_range=(1, 2),
       max_features=50000,
       sublinear_tf=True
   )
   Xtr = tfidf.fit_transform(train_texts)
   Xdv = tfidf.transform(dev_texts)
   ```

4. **Logistic Regression Baseline**

   * Hyperparameter search over `C` in `{0.5, 1.0, 2.0, 4.0}`.
   * Model selected based on **dev Macro-F1**.

5. **Linear SVM (LinearSVC)**

   * Hyperparameter search over `C` in `{0.5, 1.0, 2.0}`.
   * Best SVM compared against best Logistic Regression.
   * **Final baseline** = model with highest **dev Macro-F1** (in our run: LinearSVC).

6. **Evaluation & Analysis**

   * Compute and save:

     * **Classification report** (per-class precision/recall/F1)
     * **Confusion matrix** and **normalized confusion matrix**
     * **Per-class metrics** (accuracy, macro-F1, per-class F1)
   * **Robustness probe**:

     * Inject character-level typos at different rates (0–10%).
     * Measure how Macro-F1 degrades with noise.
   * **Qualitative error analysis**:

     * Table of misclassified dev examples with:

       * headline text
       * true label
       * predicted label
     * Inspect common confusion pairs: **World ↔ Business**, **Business ↔ Sci/Tech**.
   * **Margin analysis**:

     * For correctly classified examples, compute the margin between the top-2 decision scores.
     * High margin → confident predictions; low margin → borderline cases.

---

## 5. Transformer Models: DistilBERT & RoBERTa

### 5.1 DistilBERT (Full Fine-Tuning)

**Main notebook:** DistilBERT notebook (e.g., `02_distilbert_full_finetune.ipynb`)

1. **Tokenization**

   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   ```

   * Headlines are tokenized into **subword tokens**, truncated and padded to a fixed length.
   * Applied to train, dev, and test using `.map(...)` from `datasets`.

2. **Model & TrainingArguments**

   ```python
   from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

   model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-uncased",
       num_labels=4
   )

   args = TrainingArguments(
       output_dir="distilbert_agnews_seed42",
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       metric_for_best_model="macro_f1",
       num_train_epochs=3,
       learning_rate=2e-5,
       per_device_train_batch_size=32,
       per_device_eval_batch_size=64,
       weight_decay=0.01,
       seed=42,
   )
   ```

3. **Metrics with `evaluate`**

   * Accuracy, macro-F1, macro-precision, and macro-recall are computed via a `compute_metrics` function that:

     * Takes logits and labels from the Trainer
     * Applies `argmax` to obtain predictions
     * Returns a metrics dictionary.

4. **Trainer & Training**

   ```python
   trainer = Trainer(
       model=model,
       args=args,
       train_dataset=train_tok,
       eval_dataset=dev_tok,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
   )

   trainer.train()
   ```

5. **Evaluation**

   * `trainer.evaluate(dev_tok)` and `trainer.evaluate(test_tok)` to obtain dev and test metrics.
   * Metrics saved for comparison with baselines.

6. **Bottleneck Analysis**

   * The fine-tuned model is run on the dev set to collect:

     * headline text
     * true label
     * predicted label
   * Common confusion pairs are inspected:

     * World → Business / Business → World
     * Business → Sci/Tech / Sci/Tech → Business
   * Misclassified headlines are examined to understand how overlapping economic, political, and technological language causes errors.

### 5.2 RoBERTa & LoRA (Parameter-Efficient Fine-Tuning)

**Main notebook:** RoBERTa/LoRA experiments (e.g., `03_roberta_and_lora_experiments.ipynb`)

* **RoBERTa full fine-tuning**:

  * Similar setup to DistilBERT but using `roberta-base` as the backbone.
* **LoRA experiments**:

  * Base model weights are frozen.
  * Low-rank trainable adapters are inserted into attention layers.
  * Trainable parameters are reduced substantially while often matching or improving performance relative to full fine-tuning.

The experiments compare:

* Full DistilBERT vs DistilBERT+LoRA
* Full RoBERTa vs RoBERTa+LoRA

For each model, metrics such as **accuracy**, **macro-F1**, approximate **trainable parameter counts**, and **training time** are recorded.

---

## 6. Results Summary

A consolidated comparison shows:

* **TF–IDF + LinearSVC baseline**

  * ≈ 0.92 Accuracy / 0.92 Macro-F1
* **DistilBERT (full fine-tuning)**

  * ≈ 0.92 Accuracy / 0.92 Macro-F1
* **RoBERTa (full fine-tuning)**

  * ≈ 0.93 Accuracy / 0.93 Macro-F1
* **DistilBERT + LoRA**

  * ≈ 0.94 Accuracy / 0.94 Macro-F1
  * Trains only a small fraction of parameters compared to full fine-tuning.
* **RoBERTa + LoRA**

  * ≈ 0.94 Accuracy / 0.94 Macro-F1
  * Uses even fewer trainable parameters than full RoBERTa.

**Key takeaway:**

* A well-tuned **TF–IDF + LinearSVC** is a strong, simple baseline for short-text classification.
* **Transformer models with LoRA** achieve the best accuracy and macro-F1 while being more parameter-efficient and easier to fine-tune than full models.

---

## 7. How to Run the Notebooks

### Option A: Local (Jupyter)

1. Set up the Python environment and install `requirements.txt`.

2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open and run the notebooks in order, for example:

   * Baseline notebook (`01_baseline_tfidf_svm_agnews.ipynb`)
   * DistilBERT notebook (`02_distilbert_full_finetune.ipynb`)
   * RoBERTa/LoRA notebook (`03_roberta_and_lora_experiments.ipynb`)

### Option B: Google Colab

1. Upload the notebooks to Google Colab.
2. Optionally mount Google Drive if you want to save models/outputs.
3. Run the cells in order; the dataset will be downloaded automatically from Hugging Face.

---

## 8. References

1. S. González-Carvajal and E. C. Garrido-Merchán, “Comparing BERT against traditional machine learning text classification,” *arXiv preprint* arXiv:2005.13012, 2021.
2. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” *arXiv preprint* arXiv:2106.09685, 2021.
3. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” *arXiv preprint* arXiv:1910.01108, 2019.
4. Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” *arXiv preprint* arXiv:1907.11692, 2019.

# Mini AI Pipeline: News Headline Classification (AG News)

**Author:** A/P SATHESKUMAR SARVINA (Student ID: 2024148001)  
**Course:** CAS2105 — Homework 6

## 1. Introduction
This project explores a small AI pipeline for classifying news headlines. The chosen task is **news headline classification**, where each headline is assigned to one of four categories: `World`, `Sports`, `Business`, or `Sci/Tech.`

This task is interesting because accurate headline classification is widely applicable: it supports news aggregators, recommendation engines, and automated content moderation. It also illustrates the difference between simple heuristics and modern embedding-based approaches.

The project uses the **AG News dataset** due to its public availability, structured format, and suitability for fast experimentation on a manageable subset. We compare a naïve baseline* (keyword-based) with an **AI pipeline** (MiniLM embeddings + Logistic Regression), highlighting the benefits of semantic embeddings.

Overall, this project emphasizes the AI workflow: **defining the problem, designing a baseline, building an improved pipeline, evaluating results, and reflecting on outcomes.** The pipeline is lightweight, reproducible, and interpretable, capable of running efficiently on a single GPU or CPU.

Overall, this project emphasizes the AI workflow: defining the problem, designing a baseline, building an improved pipeline, evaluating results, and reflecting on outcomes. The pipeline is lightweight, reproducible, and interpretable, capable of running efficiently on a single GPU or CPU.

## 2. Task Definition
### **Task:** Classify news headlines into four categories: `World`, `Sports`, `Business`, and `Sci/Tech` using the AG News dataset.  
**Motivation:** Quick and reliable headline classification is useful for news aggregation, topic routing, personalized feeds, and automated content moderation. This project emphasizes the AI pipeline process: problem definition, baseline, model pipeline, evaluation, and reflection. It demonstrates the workflow of developing and comparing simple AI methods on a real dataset.
**Input:** Single short news headline (text).  
**Output:** One of four labels {World, Sports, Business, Sci/Tech}.  
**Success criteria:** High classification accuracy and balanced macro F1 across classes on held-out test data. The system should be reproducible and interpretable for quick experimentation.

## 3. Dataset
**Source:** AG News via Hugging Face `datasets`.  
**Subset Used:** 250 training examples per class (1000 total) and 100 test examples per class (400 total).
**Splits:** Train and test subsets, sampled to ensure balanced class representation.
**Preprocessing:** Minimal. Headlines are used in lowercase for keyword matching in the baseline. AI pipeline uses raw text; embedding model handles tokenization internally.
**Rationale for subset:** Using a small balanced subset allows fast experimentation while still demonstrating differences between baseline and AI pipeline performance.

## 4. Methods

### 4.1 Naïve Baseline
**Method description:** Keyword-based classifier, means each class has a curated list of keywords. Headlines are scored by keyword occurrences and assigned to the highest-scoring class. Ties or zero matches are resolved deterministically. 
**Why naïve:** It ignores semantics, word order, and polysemy. Cannot handle synonyms or paraphrased expressions. 
**Limitations / failure modes::** 
    - Fails on ambiguous headlines or headlines without keywords.
    - Sensitive to word choice; minor changes in phrasing can drastically alter predictions.
    - Example: “NASA launches new Mars rover” → baseline predicts `World` instead of `Sci/Tech`.

### 4.2 AI Pipeline
**Models used:** `sentence-transformers/all-MiniLM-L6-v2` to compute sentence embeddings; `sklearn.LogisticRegression` as a classifier.  
**Pipeline stages:**
    **1. Preprocessing:** raw headlines (tokenization handled internally by the embedding model).
    **2. Embedding:** encode each headline into a 384-dim vector (MiniLM).
    **3. Classifier:** logistic regression trained on embeddings.
    **4. Post-processing:** predicted label mapping back to text.

**Design choices & Justification:** Embeddings+linear classifier is inference-light, reproducible, and effective for short texts without fine-tuning. It meets the constraint to run comfortably on CPU.

## 5. Experiments & Results
**Metrics:** Accuracy, Precision (macro), Recall (macro), F1 (macro).  
**Results table** (computed by running the notebook):

| Method                  | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Notes                                                         |
| ----------------------- | -------- | ----------------- | -------------- | ---------- | ------------------------------------------------------------- |
| Baseline (keywords)     | 0.47     | 0.73              | 0.47           | 0.45       | High precision but low recall; fails on paraphrased headlines |
| AI pipeline (MiniLM+LR) | 0.88     | 0.88              | 0.88           | 0.88       | Consistent performance; captures semantic meaning             |

**Qualitative examples:** 

| Headline                                  | Gold Label | Baseline Pred | AI Pred  |
| ----------------------------------------- | ---------- | ------------- | -------- |
| “NASA launches new Mars rover”            | Sci/Tech   | World         | Sci/Tech |
| “Stock market surges after tech earnings” | Business   | World         | Business |
| “Local team wins championship final”      | Sports     | Business      | Sports   |

    These examples highlight how the AI pipeline handles semantic understanding, while the baseline relies strictly on keyword matching.

## 6. Reflection & Limitations
- **Successes:** The embedding + logistic regression pipeline significantly outperforms the baseline, handling paraphrased headlines and synonyms effectively.
- **Challenges:** Baseline fails on short, ambiguous, or synonym-rich headlines. Some classes (e.g., World vs Business) overlap conceptually, causing occasional misclassification.
- **Metric suitability:** Accuracy and macro F1 are appropriate for this balanced subset. For imbalanced datasets, per-class breakdowns or micro-averaged metrics would be more informative.
- **Future work:** Fine-tune a small transformer classifier on the full AG News dataset, expand baseline keyword coverage, explore ensemble methods, or deploy as a lightweight API.

## 7. Reproducibility & Use
- **Notebook:** `notebooks/pipeline_demo.ipynb` reproduces all steps.
- **Requirements:** `requirements.txt` includes necessary dependencies.
- **Artifacts:** Trained model and embeddings can be saved under `artifacts/` if saving steps are run.

## 8. References
- AG News dataset (Hugging Face)
- `sentence-transformers/all-MiniLM-L6-v2`
- scikit-learn documentation


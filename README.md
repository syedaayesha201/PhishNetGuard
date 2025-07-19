# PhishNetGuard: A Data Science Approach to Advanced Classification Models for Phishing Detection

**PhishNetGuard** is a comprehensive phishing detection framework developed as part of an MS Data Science research thesis at **Bahauddin Zakariya University, Multan**. It leverages classical, ensemble, and deep learning techniques to classify URLs as phishing or legitimate using two datasets with extensive URL-based and content-based features.

---

## 📌 Research Objectives

- Evaluate and compare traditional, ensemble, and deep learning models for phishing URL classification.
- Engineer advanced features including a **Composite Risk Score**.
- Improve detection accuracy using **Optuna-based hyperparameter tuning**.
- Assess model performance using standard metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 📁 Datasets Used

### 🟠 Dataset 1: `Phishing_Legitimate_full.csv`
- 10,000 labeled samples (5,000 phishing, 5,000 legitimate).
- 48 features extracted from URLs and webpage behavior.
- Covers protocol, path, domain, content, and hybrid features.

### 🔵 Dataset 2: `PhiUSIIL_Dataset.csv`
- 235,795 labeled samples with 56 initial features.
- Preprocessed and reduced to 16 key features.
- Includes lexical, behavioral, and semantic webpage attributes.

---

## 🛠️ Technologies and Tools

- **Language**: Python  
- **Development Platform**: Google Colab  
- **Libraries**:  
  - `TensorFlow`, `Keras`, `Scikit-learn`, `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`  
  - `Optuna` for hyperparameter optimization

---

## 🔎 Feature Engineering & EDA

- Introduced a **Composite Risk Score** based on six binary phishing indicators:
  - No HTTPS, IP Address usage, Fake Link in Status Bar, Disabled Right-Click, Iframes, Submit to Email
- Feature categorization into continuous and ordinal types
- Visual analysis using:
  - Pairplots
  - Heatmaps
  - Distribution fitting
  - Ordinal histograms

---

## 🤖 Models Implemented

### 🔹 Classical Models:
- Decision Tree, Logistic Regression, SVM

### 🔹 Ensemble Models:
- Random Forest, Bagging, Gradient Boosting

### 🔹 Deep Learning Models:
- RNN, LSTM, GRU, Bi-LSTM, Bi-GRU
- CNN, CNN-GRU Hybrid
- GRU with Attention Mechanism

---

## ⚙️ Model Optimization

- Applied **Optuna** for Bayesian hyperparameter tuning
- Tuned: learning rates, hidden layers, dropout, depth, estimators

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC

Best accuracy achieved: **97% using CNN-GRU with Attention**  
Strong performance across both datasets using feature selection and hybrid deep learning models.

---

## 📁 Repository Structure

```bash
PhishNetGuard/
│
├── Phishing_Legitimate_full.csv       # Dataset 1 (URL features)
├── PhiUSIIL_Dataset.csv               # Dataset 2 (hybrid features)
├── phishing_detection.ipynb           # Main project notebook
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies (optional)

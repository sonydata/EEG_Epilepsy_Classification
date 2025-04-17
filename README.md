# 🧠 EEG Epilepsy Classification

This project aims to build a robust **EEG-based binary classifier** to distinguish between patients **with and without epilepsy**, using raw clinical EEG data and machine learning pipelines. It combines signal preprocessing, feature engineering, and deep learning to explore what models best detect epilepsy from brain activity.

---

## 📊 Dataset: TUH Epilepsy Corpus v2.0.1

- **Source**: [Temple University Hospital EEG Epilepsy Corpus](https://isip.piconepress.com/projects/tuh_eeg/)
- **Structure**:
  - 200 patients total (100 epileptic / 100 non-epileptic)
  - 698 EEG sessions, ~2300 EEG files
- Labels were verified by neurologists based on clinical reports

---

## 🔄 Preprocessing Pipeline

Raw `.edf` EEG signals were cleaned and segmented using the following steps:

1. Load session from .edf (via MNE)
2. Apply bandpass filter (1–45 Hz)
3. Resample to 250 Hz
4. Select relevant EEG channels 
5. Segment into overlapping 5-second windows
6. Normalize each epoch 

Outputs are stored as pickled DataFrames and used in all downstream models.

---

## 🤖 Models Implemented

### ✅ Classical ML
- **Features extracted from**: `features_engineering.py`
- **Feature types**:
  - Time-domain: mean, variance, skewness, kurtosis
  - Frequency-domain: relative power in delta, theta, alpha, beta, gamma bands
  - Complexity: sample entropy, permutation entropy
  - Wavelet-domain: DWT 
- **Models tested**: `Logistic Regression`, `XGBoost`
- **Pipeline**: Handcrafted features → PCA → Classification -> Stratified cross-validation (by patient)


## 🧠 Deep Learning Models

### EEGNet
- Shallow compact CNN architecture tailored for EEG signal classification
- Trained on raw multi-channel segment arrays

### EpilepsyNet
- Multi-head attention architecture built over **correlation matrices** between EEG channels
- Uses full 1-minute segments split into 5-second chunks, then computes correlation matrices

### SpectroNet (2D CNN)
- CNN trained on **2D STFT spectrograms** of EEG segments
- Model WIP in `model_2dcnn.py`  
---

## 📈 Performance Snapshot

| Model        | Sensitivity | Specificity|
|--------------|-------------|----------|
| EEGNet       | 75%         | 88%     |
| EpilepsyNet  | 82%         | 71%     |
| SpectroNet   | 77%         | 71%     |
| ChronoNet    | 69%         | 64%     |

---

## 🌐 Interactive Tools & Generative Reporting

### 📊 Real-Time Demo: Epilepsy Classifier Dashboard  
Visualize EEG segment outputs from an edf file, explore predictions, and compare models, using our interactive dashboard:  
👉 **[EEG Epilepsy App on Hugging Face Spaces](https://huggingface.co/spaces/MorganBrizon/EEG_Epilepsy_App)**

### 🧾 LLM-Generated EEG Reports (Experimental)  
We developed a proof-of-concept pipeline for **automated EEG report generation** using extracted signal features and Google's **Gemini 2.0 Flash API** via **LangChain**.

- Simulates clinician-style summaries based on extracted EEG characteristics
- Prompt engineering based on the American Clinical Neurophysiology Society (ACNS) Guideline for EEG reporting 
- Source notebook: [`EEG_report_Gemini.ipynb`](notebooks/EEG_report_Gemini.ipynb)

⚠️ *Note: This component is experimental and intended for research purposes only.*

---

## 🧾 Final Presentation

The summary of this work is available here:  
📎 [`Epilepsy_Classification_PPT_final.pptx`](Epilepsy_Classification_PPT_final.pptx)

---

## 🚀 Next Steps

- Train ensemble model combining EEGNet + EpilepsyNet
- Improve segment-level labeling for seizure-specific classification
- Improve generalization with data augmentation 
- Explore model interpretability (e.g., Grad-CAMs)  

---

## 👥 Team

_Data Science & Engineering Bootcamp - Final Project Team: Sonia, Robin, Morgan, Zacharie, Eli_

---


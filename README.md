# 🔩 SAIL Steel Defect Classification System
### Internship Project — CSE | Quality Department

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | UCI Steel Plates Faults (free, no login needed) |
| **Samples** | 1,941 steel plate records |
| **Features** | 27 process parameters (geometry, luminosity, indices) |
| **Target** | 7 Defect Types (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other) |
| **Best Model** | Random Forest (~79% accuracy) |
| **Interface** | Streamlit Web Dashboard |

---

## 🚀 Quick Start (Step by Step)

### Step 1 — Install Python
- Download Python 3.10+ from https://python.org
- During install, check ✅ "Add Python to PATH"

### Step 2 — Install Libraries
Open Command Prompt / Terminal in the project folder:
```bash
pip install -r requirements.txt
```

### Step 3 — Train the Model & Generate Charts
```bash
python analysis.py
```
This will:
- Download the UCI dataset automatically
- Train 4 ML models (Random Forest, Gradient Boosting, Decision Tree, KNN)
- Generate 5 charts in the `/charts/` folder
- Save the trained model in `/models/`
- Print accuracy report in terminal

### Step 4 — Launch the Dashboard
```bash
streamlit run app.py
```
- Opens automatically in your browser at http://localhost:8501
- Three pages: Dashboard | Predict | Upload & Analyze

---

## 📁 Project Structure

```
steel_defect_project/
│
├── analysis.py          ← Run this FIRST (trains model, makes charts)
├── app.py               ← Streamlit dashboard (run after analysis.py)
├── requirements.txt     ← All Python libraries needed
├── README.md            ← This file
│
├── charts/              ← Auto-generated charts (after running analysis.py)
│   ├── 01_defect_frequency.png
│   ├── 02_feature_correlation.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrix.png
│   └── 05_feature_importance.png
│
└── models/              ← Saved ML model files
    ├── steel_defect_model.pkl
    ├── label_encoder.pkl
    └── feature_names.pkl
```

---

## 🗃️ About the Dataset

**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults  
**License:** Creative Commons CC BY 4.0 (free for academic use)

### Defect Types:
| # | Defect | Description | Count |
|---|--------|-------------|-------|
| 1 | Pastry | Surface irregularity pattern | 158 |
| 2 | Z_Scratch | Diagonal scratch marks | 190 |
| 3 | K_Scratch | Short surface scratches | 391 |
| 4 | Stains | Contamination/stain marks | 72 |
| 5 | Dirtiness | Foreign material spots | 55 |
| 6 | Bumps | Surface protrusions | 402 |
| 7 | Other_Faults | Uncategorized defects | 673 |

### Key Features (27 total):
- **Geometry:** X_Min, X_Max, Y_Min, Y_Max, Pixels_Areas, X_Perimeter, Y_Perimeter
- **Luminosity:** Sum, Min, Max, Luminosity_Index
- **Steel Info:** Steel Type (A300/A400), Plate Thickness, Conveyer Length
- **Shape Indices:** Edges_Index, Empty_Index, Square_Index, Log_of_Areas, etc.

---

## 🤖 ML Models Used

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **Random Forest** ⭐ | ~79% | Best overall, used in production |
| Gradient Boosting | ~78% | Slightly slower to train |
| Decision Tree | ~74% | Most interpretable |
| KNN | ~72% | No training phase |

---

## 📊 Dashboard Pages

### 1. Dashboard
- Metrics: total samples, features, defect types, accuracy
- Defect frequency bar chart
- Defect distribution pie chart
- Feature importance chart
- Confusion matrix
- Raw data table

### 2. Predict Defect
- Input 27 process parameters using sliders/inputs
- Click "Predict" → get defect type + confidence scores for all 7 types

### 3. Upload & Analyze
- Upload your own Excel/CSV from SAIL production logs
- Automatic defect distribution analysis
- Statistical summary

---

## 📝 How to Use SAIL's Own Data (If Available)

If your mentor gives you production/defect data:
1. Save it as `sail_data.xlsx`
2. Go to the "Upload & Analyze" tab in the dashboard
3. Upload the file — it will auto-analyze defect patterns

If the SAIL data has columns like:
`Date, Shift, Product_Type, Defect_Type, Rejection_%`

You can also replace the `X, y = load_data()` call in `app.py` with:
```python
df = pd.read_excel('sail_data.xlsx')
X = df.drop(columns=['Defect_Type', 'Date'])
y = df['Defect_Type']
```

---

## 📄 Report Outline (for Internship Submission)

```
Chapter 1: Introduction
  1.1 About SAIL & Quality Department
  1.2 Problem Statement (defect losses, rejection %)
  1.3 Objective

Chapter 2: Literature Review
  2.1 Steel defect types and causes
  2.2 ML in quality control (cite 3 papers)

Chapter 3: Dataset Description
  3.1 UCI Steel Plates Faults Dataset
  3.2 Features and target variables
  3.3 Data preprocessing

Chapter 4: Methodology
  4.1 Exploratory Data Analysis
  4.2 ML models used and why
  4.3 Evaluation metrics

Chapter 5: Results & Analysis
  5.1 Defect frequency analysis (Chart 1)
  5.2 Feature importance findings (Chart 5)
  5.3 Model comparison (Chart 3)
  5.4 Confusion matrix analysis (Chart 4)

Chapter 6: Dashboard Overview
  6.1 Screenshots of all 3 pages
  6.2 How to use the system

Chapter 7: Conclusion & Recommendations
  7.1 Key findings
  7.2 Actionable recommendations for SAIL
  7.3 Future scope

References
```

---

## 💡 Tips for Your Internship Presentation

- **Print the confusion matrix** and explain what it means in simple language
- **Highlight the feature importance chart** — "Luminosity Index is the #1 predictor of defect type"
- **Show the live dashboard** on your laptop during presentation
- **Connect to SAIL context**: "If SAIL uses this on its Durgapur/Bokaro plant logs, it can reduce manual QC effort by X%"
- **Mention the dataset source** properly — UCI ML Repository, CC BY 4.0 license

---

*Project developed as part of Summer Internship — SAIL Quality Department*  
*Dataset: UCI Steel Plates Faults | Model: Random Forest Classifier*

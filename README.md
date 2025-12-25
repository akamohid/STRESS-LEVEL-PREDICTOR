   # 🎓 Stress Level Predictor


## 🏆 Project Overview
Predict student stress levels (Low 🟢, Medium 🟡, High 🔴) using psychological, academic, and lifestyle survey data via machine learning. Empower universities to **intervene early** and support student well-being.

---

## 🔑 Key Features
- 🧹 **Data Cleaning & Preprocessing**  
- 📊 **Exploratory Data Analysis (EDA)**  
- 📉 **Dimensionality Reduction** with PCA  
- 🤖 **Model Training**:  
  - Decision Tree 🌳  
  - Random Forest 🌲  
  - Support Vector Machine ⚔️  
  - K-Nearest Neighbors 📍  
  - Logistic Regression ➗  
  - Gradient Boosting 🚀  
- 📈 **Model Evaluation**: Accuracy, Precision, Recall, F1-Score  
- 💡 **Comparison Dashboard** of all models  

---

## 📂 Repository Structure
```
STRESS-LEVEL-PREDICTOR/
├── 📁 data/                      # Raw & processed datasets
│   └── Dataset.csv            # 1,100 records × 21 features
├── 📁 docs/                      # Markdown docs
│   └── Project Report.docx    # Final project document
├── 📁 notebooks/                 # Exploratory Jupyter notebooks
│   └── Code.ipynb             # End-to-end pipeline
├── 📁 src/                       # Python scripts
│   ├── data_preprocessing.py  # Load → Clean → Scale → Split
│   ├── eda.py                 # Plots & insights
│   ├── modeling.py            # Train Decision Tree + metrics
│   └── evaluation.py          # Train & compare all models
├── 📄 .gitignore                 # Sensitive files to ignore
├── 📄 LICENSE                    # MIT License
├── 📄 README.md                  # This file
└── 📄 requirements.txt           # Python dependencies
```

---

## 💾 Installation

1. **Clone**  
   ```bash
   git clone https://github.com/akamohid/STRESS-LEVEL-PREDICTOR.git
   cd STRESS-LEVEL-PREDICTOR
   ```

2. **Setup environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

1. **Preprocess Data**  
   ```bash
   python src/data_preprocessing.py
   ```
2. **Run EDA**  
   ```bash
   python src/eda.py
   ```
3. **Train & Evaluate Decision Tree**  
   ```bash
   python src/modeling.py
   ```
4. **Compare All Models**  
   ```bash
   python src/evaluation.py
   ```
5. **Or** open the **Jupyter Notebook** for a unified, annotated workflow:  
   ```bash
   jupyter lab notebooks/Code.ipynb
   ```

---

## 📊 Results Overview

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|:--------:|:---------:|:------:|:--------:|
| Decision Tree 🌳       |  0.90    |   0.90    |  0.90  |   0.89   |
| Random Forest 🌲       |  0.89    |   0.88    |  0.89  |   0.87   |
| Gradient Boosting 🚀   |  0.88    |   0.87    |  0.88  |   0.86   |
| SVM ⚔️                 |  0.87    |   0.86    |  0.87  |   0.86   |
| KNN 📍                 |  0.85    |   0.85    |  0.85  |   0.84   |
| Logistic Regression ➗  |  0.86    |   0.86    |  0.86  |   0.86   |

> 📌 *Decision Tree delivered the best blend of performance and interpretability.*

---

## 📚 Literature Review

See [docs/literature_review.md](docs/literature_review.md) for details on:
- Filippis & Al Foysal (2024)  
- Arya et al. (2024)  
- Singh et al. (2024)  
- …and others.

---

## 📈 Git Workflow

```bash
git init
git remote remove origin 2>$null
git remote add origin https://github.com/akamohid/STRESS-LEVEL-PREDICTOR.git
git add .
git commit -m "Initial commit: Full Stress Level Predictor pipeline"
git branch -M main
git push -u origin main
```

---

## 👥 Team Members

- **Mohid Arshad** — [GitHub](https://github.com/akamohid) | [LinkedIn](https://www.linkedin.com/in/akamohid/)  
- **Mohammad Umar** — [LinkedIn](https://www.linkedin.com/in/mohammad-umar-1147a62a6/)  
- **Mohammad Hasnain** — [LinkedIn](https://www.linkedin.com/in/mohammad-hasnain-3670452a7/)  
- **Tahir Mehmood** — [LinkedIn](https://www.linkedin.com/in/tahir-mehmood-622a412a0/)

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📬 Contact & Feedback

Made with ❤️ by **Mohid Arshad**.  
✉️ Email: akamohid@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/mohid-arshad-347180235/)  
Feel free to open issues, contribute, or drop a ⭐!

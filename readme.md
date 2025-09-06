# Diabetes Detection using Random Forest

##  Dataset
- **Size:** 264 samples × 12 columns  
- **Target variable:** `Class` (3 classes: 0, 1, 2)  
- **Features:**
  - Gender, Age, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI  

This dataset represents clinical measurements and health indicators used to classify patients into different risk groups.

---

##  Model
- **Algorithm:** Random Forest Classifier (Scikit-learn)  
- **Train/Test split:** 80/20 with `random_state=42`  
- **Cross-validation:** 5-fold, mean accuracy ≈ **97%**  

Random Forest was chosen for its robustness, ability to handle mixed feature types, and built-in feature importance estimation.

---

##  Evaluation Results

### Test Set Performance
- **Accuracy:** 100% (on test set of 53 samples)  
- **Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 18      |
| 1     | 1.00      | 1.00   | 1.00     | 8       |
| 2     | 1.00      | 1.00   | 1.00     | 27      |
| **Accuracy** |       |        | **1.00** | **53** |
| **Macro avg** | 1.00 | 1.00   | 1.00     | 53      |
| **Weighted avg** | 1.00 | 1.00 | 1.00   | 53      |

---

##  Cross-Validation Results
- Fold scores: `[0.887, 1.000, 1.000, 0.981, 0.981]`  
- **Mean Accuracy:** ~97%  

This shows the model generalizes well and the dataset is highly separable. This is the reason behind the high accuracy of the model

---

##  Next Steps
- Analyze **feature importance** to identify which clinical factors most influence predictions.  
- Test with additional algorithms (e.g., Logistic Regression, XGBoost) for comparison.  
- Collect more data for better generalization beyond the current dataset.

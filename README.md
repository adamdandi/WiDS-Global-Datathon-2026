# 🌲🔥 WiDS Global Datathon 2026: Wildfire Survival Prediction

### *Predicting immediate wildfire threats using early-stage kinematic data.*

## 📌 Project Overview

**The Challenge:** When a wildfire ignites, emergency managers have a "golden window" to make life-saving decisions. Using data restricted to the **first 5 hours** of a fire, this project predicts the probability that a fire will reach a populated area (evacuation zone) within four critical time horizons: **12h, 24h, 48h, and 72h**.

**The Constraint:** The dataset is extremely small (**221 training events**) and heavily right-censored (many fires never hit). This required a rigorous "Small Data" strategy rather than standard Deep Learning approaches.

## 🏆 The Strategy: "The Committee of Experts"

Instead of a single complex model, we deployed a **Multi-Horizon Expert Framework**.
We treated each time horizon (12h, 24h, 48h, 72h) as a distinct survival problem, training specialized models for each.

### **1. Feature Engineering (The Physics Engine)**

Raw data was insufficient. We engineered **kinematic features** to teach the model the laws of motion:

* **`est_time_to_contact`**: (Distance to Town) / (Closing Speed).
* **`growth_intensity`**: (Area Growth) / (Initial Area).
* **`threat_momentum`**: (Speed × Acceleration).

### **2. The Model Tournament**

We rigorously compared two approaches using **Stratified 5-Fold Cross-Validation**:

* 🔴 **Approach A (Generalist):** One Regressor predicting exact "Time to Hit".
* 🟢 **Approach B (Specialist):** Four Classifiers (Random Forest), each answering "Will it hit in < X hours?"

**The Result:** The **Specialist approach won by a landslide** (AUC 0.95+ vs. 0.70), proving that simpler, focused models outperform complex ones on small, chaotic data.

### **3. Logical Post-Processing**

We enforced **Monotonicity constraints** on the final predictions.

* *Logic:* A fire cannot be *more* likely to hit in 12 hours than in 24 hours.
* *Algorithm:* `Prob(T < 24h) = max(Prob(T < 24h), Prob(T < 12h))`

---

## 📊 Key Results

Our validation (Stratified K-Fold) yielded exceptional stability across all horizons:

| Horizon | Model Type | AUC Score | Interpretation |
| --- | --- | --- | --- |
| **12 Hours** | Random Forest (Class) | **0.958** | Highly accurate at detecting immediate threats. |
| **24 Hours** | Random Forest (Class) | **0.989** | Near-perfect separation of "Safe" vs. "Danger". |
| **48 Hours** | Random Forest (Class) | **0.996** | Excellent long-term risk assessment. |
| **72 Hours** | Random Forest (Class) | **1.000*** | *Note: Due to high censorship, all long-surviving fires in training eventually hit.* |

---

## 🛠️ Repository Structure

```bash
├── data/
│   ├── train.csv           # Training data (Features + Targets)
│   ├── test.csv            # Test data (Features only)
│   └── metaData.csv        # Column dictionary
├── notebooks/
│   └── wildfire_analysis.ipynb  # Full analysis: EDA, Validation, & Modeling
├── submission_final.csv    # FINAL OUTPUT for leaderboard
└── README.md               # This file

```

## 🚀 Quick Start

### **1. Prerequisites**

* Python 3.8+
* `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### **2. Running the Analysis**

The entire pipeline is contained in the main notebook. It executes the following steps automatically:

1. **Audit:** Checks for censorship ratios and zero-variance columns.
2. **Split:** Sets up Stratified K-Fold (5 splits) to prevent data leakage.
3. **Engineer:** Calculates physics-based features.
4. **Train:** Trains 4 separate Random Forest Experts.
5. **Predict & Fix:** Generates probabilities and enforces monotonicity.


## ⚠️ Critical Note on Data

* **Censorship:** ~70% of the training fires *never* hit the town.
* **Sample Size:** Only ~50 fires hit within the first 12 hours.
* **Action:** We used `class_weight='balanced'` in all models to prevent the AI from ignoring the rare "Hit" events.

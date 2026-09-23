# NetGuard — Network Intrusion Detector

A machine learning-based network traffic classifier that distinguishes normal
network traffic from attack traffic using the NSL-KDD benchmark dataset.
Built as a companion project to **Guardian** (AI misinformation & deepfake
detection), extending the same detection-systems focus into network security.

## Project Description

NetGuard trains and compares two supervised classifiers — **Random Forest**
and **XGBoost** — to perform binary classification of network connection
records as either `normal` or `attack`. The project follows the standard
intrusion detection research workflow: data preprocessing, categorical
encoding, feature scaling, class-imbalance handling, model training, and
evaluation using metrics appropriate for imbalanced classification
(precision, recall, F1-score, and confusion matrices — not just accuracy).

The NSL-KDD test set is intentionally constructed to include attack types
absent from the training set, so the evaluation reflects a model's ability
to generalize to previously unseen attack patterns rather than simply
memorizing known signatures.

## Dataset

**NSL-KDD** — an improved, de-duplicated version of the classic KDD Cup 1999
intrusion detection dataset.

- Source: [Canadian Institute for Cybersecurity, UNB](https://www.unb.ca/cic/datasets/nsl.html)
- Kaggle mirror: [kaggle.com/datasets/hassan06/nslkdd](https://www.kaggle.com/datasets/hassan06/nslkdd)
- 41 features per connection record (duration, protocol type, service, byte
  counts, error rates, host-based traffic statistics, etc.)
- Labels collapsed to binary: `normal` (0) vs. `attack` (1)
- Training set: 125,973 records | Test set: 22,544 records

## Technologies Used

- **Python 3**
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — preprocessing (`StandardScaler`, one-hot encoding),
  Random Forest classifier, evaluation metrics
- **XGBoost** — gradient-boosted classifier
- **joblib** — model persistence

## Setup / Run Instructions

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `KDDTrain+.txt` and `KDDTest+.txt` from the
   [Kaggle NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
   and place them in the project root (same folder as the code file).
4. Run the notebook/script:
   ```bash
   python KING_NetGuard.py
   ```
   or open `KING_NetGuard.ipynb` in Jupyter and run all cells.
5. Trained models are saved as `.joblib` files in the project root:
   `random_forest_ids.joblib`, `xgboost_ids.joblib`, `scaler.joblib`,
   `feature_names.joblib`.

## Results

| Model | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|
| Random Forest | 0.8141 | 0.7968 | 0.7714 |
| XGBoost | 0.8261 | 0.8157 | 0.7938 |

XGBoost outperformed Random Forest across all three metrics. Both models
achieved high precision (~0.97) on the `attack` class, meaning very few
false alarms, but recall on `attack` (0.62–0.66) is lower — largely a
consequence of the test set containing attack types not seen during
training, which is by design in NSL-KDD.

### Top predictive features
- **Random Forest:** `src_bytes`, `dst_bytes`, `flag_SF`, `dst_host_srv_count`, `logged_in`
- **XGBoost:** `service_ecr_i`, `src_bytes`, `service_http`, `hot`, `dst_host_same_srv_rate`

## Key Information

- **Author:** Arjun Kaushik
- **Institution:** University School of Automation and Robotics, GGSIPU Delhi

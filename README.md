# Model Evaluation & Hyperparameter Tuning with scikit-learn

A hands-on notebook covering three skills that come up in almost every machine learning project: choosing a validation strategy, picking the right metric, and tuning hyperparameters efficiently.

## Skills covered

| Area | What I practised | Tools |
|---|---|---|
| **Cross-validation** | Hold-out vs. K-fold (K = 10 / 100 / 1000), and the reliability vs. runtime trade-off | `train_test_split`, `KFold` |
| **Classification metrics** | Accuracy, precision, recall, F1, and why a majority-class baseline can look deceptively good | `StratifiedKFold`, `DummyClassifier`, `sklearn.metrics` |
| **Hyperparameter tuning** | Exhaustive grid search, and randomized search over very large search spaces | `GridSearchCV`, `RandomizedSearchCV` |
| **Preprocessing** | Standardising features using training-set statistics only, to avoid data leakage | `StandardScaler` |

## Datasets

All datasets are bundled with or downloaded by scikit-learn:
- **California Housing**: regression (median house value)
- **Breast Cancer Wisconsin (Diagnostic)**: binary classification
- **Iris**: multi-class classification

## Key results

**Cross-validation (California Housing, R²)**

| Model | 10% hold-out | 10-fold | 100-fold | 1000-fold |
|---|---|---|---|---|
| KNeighborsRegressor | 0.672 | 0.691 | 0.691 | 0.658 |
| LinearRegression | 0.581 | 0.600 | 0.601 | 0.565 |
| DecisionTreeRegressor | 0.650 | 0.611 | 0.607 | 0.556 |

Takeaway: 10-fold CV gives a more stable estimate than a single split. Very large K multiplies the runtime (the decision tree went from ~3 s to ~400 s) and adds no useful information.

**Metrics vs. baseline (Breast Cancer, stratified 10-fold)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RandomForestClassifier | 0.956 | 0.963 | 0.969 | 0.965 |
| DummyClassifier (most frequent) | 0.627 | 0.627 | **1.000** | 0.771 |

Takeaway: the dummy model gets a perfect recall just by always predicting the positive class. You need to look at several metrics together.

**Hyperparameter search**

| Task | Search | Best parameters | Test score |
|---|---|---|---|
| Random Forest, California Housing | Grid (R²) | `n_estimators=100, min_samples_leaf=2` | R² = 0.806 |
| Distance-weighted KNN, Iris | Grid (accuracy & F1-macro) | `metric='euclidean', n_neighbors=5` | 1.00 |
| Decision tree, Breast Cancer | Randomized, 1000 iterations (accuracy & F1) | `max_depth=38, max_leaf_nodes=80, min_samples_leaf=17, min_samples_split=8` | Acc 0.947 / F1 0.958 |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook model_evaluation_and_tuning.ipynb
```

> Note: the K = 1000 cross-validation cell is slow (several minutes). Lower K to get a quick run.

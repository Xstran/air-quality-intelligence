
# 02 - HIGH-POLLUTION DAY CLASSIFICATION
# Predict whether daily PM2.5 exceeds 15 µg/m³
# Development: 2021-2022 | Final holdout: 2023


from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



# CELL 1 - PROJECT PATHS AND SOURCE MODULES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    create_modelling_features,
    MET_FEATURES,
    LAG_FEATURES,
    FEATURES_METEOROLOGY_ONLY,
    FEATURES_METEOROLOGY_LAG,
    CLASSIFICATION_TARGET,
)

from src.classification import evaluate_classifier, tune_classifier

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_daily_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# CELL 2 - LOAD PREPARED DATA


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}\n"
        "Run scripts/01_data_preparation_eda.py first."
    )

df_merged = pd.read_csv(DATA_PATH, parse_dates=["Date"])

print("Loaded prepared dataset")
print("Shape:", df_merged.shape)
print(
    "Date range:",
    df_merged["Date"].min().date(),
    "to",
    df_merged["Date"].max().date(),
)



# CELL 3 - CREATE MODELLING FEATURES


feature_data = create_modelling_features(df_merged)

print("\nFinal modelling rows:", len(feature_data))
print("Meteorological features:", len(MET_FEATURES))
print("Lag features:", len(LAG_FEATURES))

print("\nTarget class counts:")
print(feature_data[CLASSIFICATION_TARGET].value_counts())

print("\nTarget class percentage:")
print(
    (feature_data[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100)
    .round(1)
)



# CELL 4 - CHRONOLOGICAL TRAIN / TEST SPLIT


# 2021-2022 development data
train_data = feature_data[feature_data["year"].isin([2021, 2022])].copy()

# 2023 untouched final holdout
test_data = feature_data[feature_data["year"] == 2023].copy()

# Feature set 1: meteorology only
X_train_met = train_data[FEATURES_METEOROLOGY_ONLY]
X_test_met = test_data[FEATURES_METEOROLOGY_ONLY]

# Feature set 2: meteorology + PM2.5 lags
X_train_met_lag = train_data[FEATURES_METEOROLOGY_LAG]
X_test_met_lag = test_data[FEATURES_METEOROLOGY_LAG]

# Classification target
y_train = train_data[CLASSIFICATION_TARGET]
y_test = test_data[CLASSIFICATION_TARGET]

print(
    "\nTraining date range:",
    train_data["Date"].min().date(),
    "to",
    train_data["Date"].max().date(),
)
print(
    "Testing date range:",
    test_data["Date"].min().date(),
    "to",
    test_data["Date"].max().date(),
)

print("\nTraining observations:", len(train_data))
print("Testing observations:", len(test_data))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Guard against temporal leakage
assert train_data["Date"].max() < test_data["Date"].min()
print("\nNo temporal train/test overlap confirmed.")



# CELL 5 - FEATURE SETS


feature_sets = {
    "Meteorology only": {
        "X_train": X_train_met,
        "X_test": X_test_met,
    },
    "Meteorology + lag": {
        "X_train": X_train_met_lag,
        "X_test": X_test_met_lag,
    },
}



# CELL 6 - EXPANDING-WINDOW TIMESERIESSPLIT


tscv = TimeSeriesSplit(n_splits=3)

print("\nTimeSeriesSplit with 3 chronological folds:")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_met), start=1):
    train_dates = train_data.iloc[train_idx]["Date"]
    val_dates = train_data.iloc[val_idx]["Date"]
    val_positive = y_train.iloc[val_idx].sum()

    print(f"\nFold {fold}")
    print(
        "  Train:",
        train_dates.min().date(),
        "to",
        train_dates.max().date(),
        f"({len(train_idx)} observations)",
    )
    print(
        "  Validation:",
        val_dates.min().date(),
        "to",
        val_dates.max().date(),
        f"({len(val_idx)} observations)",
    )
    print(
        "  Validation high-PM2.5 days:",
        int(val_positive),
        f"({val_positive / len(val_idx) * 100:.1f}%)",
    )



# CELL 7 - HYPERPARAMETER SEARCH SPACES


rf_param_dist = {
    "n_estimators": randint(100, 601),
    "max_depth": [5, 8, 12, 15, None],
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 9),
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}

rf_class_weight_param_dist = {
    "n_estimators": randint(100, 601),
    "max_depth": [5, 8, 12, 15, None],
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 9),
    "max_features": ["sqrt", "log2", 0.5, 0.8],
    "class_weight": ["balanced", "balanced_subsample"],
}

rf_cs_param_dist = {
    "n_estimators": randint(100, 501),
    "max_depth": [8, 12, 15, None],
    "min_samples_split": randint(5, 21),
    "min_samples_leaf": randint(2, 12),
    "max_features": ["sqrt", "log2", 0.5, 0.8],
    "class_weight": [
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 4},
    ],
}

gb_param_dist = {
    "n_estimators": randint(100, 601),
    "learning_rate": loguniform(0.01, 0.2),
    "max_depth": randint(1, 6),
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 9),
    "subsample": uniform(0.6, 0.4),
    "max_features": [None, "sqrt", "log2", 0.5, 0.8],
}

gb_cs_param_dist = {
    "n_estimators": randint(150, 281),
    "learning_rate": loguniform(0.008, 0.04),
    "max_depth": [1, 2],
    "min_samples_split": randint(18, 36),
    "min_samples_leaf": randint(9, 19),
    "subsample": uniform(0.75, 0.18),
    "max_features": ["log2", "sqrt", 0.5, 0.7],
}



# CELL 8 - SELECT GRADIENT BOOSTING MINORITY WEIGHT


weight_candidates = [2.0, 2.5, 2.7, 3.0, 3.5, 3.7, 4.0, 4.5, 4.7, 5.0, 6.0]
weight_results = []

for weight in weight_candidates:
    current_sample_weight = y_train.map({0: 1.0, 1: weight}).to_numpy()

    search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions=gb_cs_param_dist,
        n_iter=50,
        scoring="f1",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(
        X_train_met_lag,
        y_train,
        sample_weight=current_sample_weight,
    )

    weight_results.append({
        "Minority weight": weight,
        "CV F1": search.best_score_,
    })

weight_results = (
    pd.DataFrame(weight_results)
    .sort_values("CV F1", ascending=False)
    .reset_index(drop=True)
)

best_weight_f1 = weight_results["CV F1"].max()

best_weights = weight_results.loc[
    np.isclose(
        weight_results["CV F1"],
        best_weight_f1,
        atol=1e-6,
        rtol=0,
    ),
    "Minority weight",
].tolist()

# If weights tie, retain the larger minority-class weight
selected_gb_weight = float(max(best_weights))

sample_weight_gb_cost = y_train.map({
    0: 1.0,
    1: selected_gb_weight,
}).to_numpy()

print("\nGradient Boosting weight search:")
print(weight_results.round(4))
print("\nSelected minority weight:", selected_gb_weight)

weight_results.to_csv(
    RESULTS_DIR / "classification_weight_search.csv",
    index=False,
)



# CELL 9 - UNTUNED BASELINE MODELS


baseline_models = {
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "sample_weight": None,
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "sample_weight": None,
    },
}



# CELL 10 - TUNE CLASSIFICATION MODELS


tuning_models = {
    "Tuned Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=1),
        "param_dist": rf_param_dist,
        "sample_weight": None,
    },
    "Tuned class-weight Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=1),
        "param_dist": rf_class_weight_param_dist,
        "sample_weight": None,
    },
    "Tuned cost-sensitive Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=1),
        "param_dist": rf_cs_param_dist,
        "sample_weight": None,
    },
    "Tuned Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_dist": gb_param_dist,
        "sample_weight": None,
    },
    "Tuned cost-sensitive Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_dist": gb_cs_param_dist,
        "sample_weight": sample_weight_gb_cost,
    },
}

tuned_searches = {}

for model_name, model_info in tuning_models.items():
    for feature_set_name, data in feature_sets.items():
        print(f"\nTuning {model_name} - {feature_set_name}")

        search = tune_classifier(
            model=model_info["model"],
            param_dist=model_info["param_dist"],
            X_train=data["X_train"],
            y_train=y_train,
            cv=tscv,
            sample_weight=model_info["sample_weight"],
            n_iter=100,
        )

        tuned_searches[(model_name, feature_set_name)] = search

        print("Best parameters:", search.best_params_)
        print("Best CV F1:", round(search.best_score_, 3))



# CELL 11 - SELECT MODEL USING 2021-2022 CV ONLY


cv_selection_rows = []

for (model_name, feature_set_name), search in tuned_searches.items():
    best_index = search.best_index_
    validation_f1 = search.best_score_
    validation_std = search.cv_results_["std_test_score"][best_index]
    training_f1 = search.cv_results_["mean_train_score"][best_index]

    cv_selection_rows.append({
        "Model": model_name,
        "Feature set": feature_set_name,
        "Training CV F1": training_f1,
        "Validation CV F1": validation_f1,
        "CV F1 standard deviation": validation_std,
        "Train-validation gap": training_f1 - validation_f1,
    })

cv_selection_df = (
    pd.DataFrame(cv_selection_rows)
    .sort_values(
        by=[
            "Validation CV F1",
            "CV F1 standard deviation",
            "Train-validation gap",
        ],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
)

selected_model_name = cv_selection_df.loc[0, "Model"]
selected_feature_set = cv_selection_df.loc[0, "Feature set"]

print("\nClassification model selection using 2021-2022 only:")
print(cv_selection_df.round(3))

print("\nFINAL CLASSIFIER SELECTED")
print("Model:", selected_model_name)
print("Feature set:", selected_feature_set)
print(
    "Validation CV F1:",
    round(cv_selection_df.loc[0, "Validation CV F1"], 3),
)

cv_selection_df.to_csv(
    RESULTS_DIR / "classification_cv_results.csv",
    index=False,
)



# CELL 12 - CLASSIFICATION THRESHOLD


# Standard threshold retained; no threshold tuning performed
selected_threshold = 0.5
print("\nClassification threshold:", selected_threshold)


# CELL 13 - FINAL 2023 EVALUATION


all_results = []
model_outputs = {}

# Evaluate untuned baselines
for model_name, model_info in baseline_models.items():
    for feature_set_name, data in feature_sets.items():
        result, output = evaluate_classifier(
            model=model_info["model"],
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=y_train,
            y_test=y_test,
            model_name=model_name,
            feature_set_name=feature_set_name,
            sample_weight=model_info["sample_weight"],
        )

        all_results.append(result)
        model_outputs[(model_name, feature_set_name)] = output

# Evaluate tuned models
for (model_name, feature_set_name), search in tuned_searches.items():
    data = feature_sets[feature_set_name]

    weights = (
        sample_weight_gb_cost
        if model_name == "Tuned cost-sensitive Gradient Boosting"
        else None
    )

    result, output = evaluate_classifier(
        model=search.best_estimator_,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=y_train,
        y_test=y_test,
        model_name=model_name,
        feature_set_name=feature_set_name,
        sample_weight=weights,
    )

    all_results.append(result)
    model_outputs[(model_name, feature_set_name)] = output

all_model_results = (
    pd.DataFrame(all_results)
    .drop_duplicates(
        subset=["Model", "Feature set"],
        keep="last",
    )
    .reset_index(drop=True)
)

all_model_results["Selected by development CV"] = (
    (all_model_results["Model"] == selected_model_name)
    & (all_model_results["Feature set"] == selected_feature_set)
)

all_model_results = (
    all_model_results
    .sort_values(
        by=["Selected by development CV", "Model", "Feature set"],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
)

print("\nFinal 2023 classification results:")
print(all_model_results.round(3))

all_model_results.to_csv(
    RESULTS_DIR / "classification_model_comparison.csv",
    index=False,
)



# CELL 14 - RETRIEVE DEVELOPMENT-SELECTED MODEL


best_key = (selected_model_name, selected_feature_set)
best_output = model_outputs[best_key]

best_model = best_output["model"]
best_prob = best_output["prob"]
best_pred = (best_prob >= selected_threshold).astype(int)
best_X_train = best_output["X_train"]
best_X_test = best_output["X_test"]

print("\nBest model:", selected_model_name)
print("Feature set:", selected_feature_set)


# CELL 15 - CONFUSION MATRIX AND CLASSIFICATION REPORT


cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
tn_selected, fp_selected, fn_selected, tp_selected = cm.ravel()

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted normal", "Predicted high"],
    yticklabels=["Actual normal", "Actual high"],
)

plt.title("High-PM2.5 Classification Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classification_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\nClassification report:")
print(
    classification_report(
        y_test,
        best_pred,
        target_names=["Normal PM2.5 day", "High PM2.5 day"],
        zero_division=0,
    )
)

print(
    "Correctly identified high days:",
    tp_selected,
    "out of",
    int(y_test.sum()),
)



# CELL 16 - PRECISION-RECALL CURVE


PrecisionRecallDisplay.from_predictions(y_test, best_prob)
plt.title("Precision-Recall Curve - Selected Classifier")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classification_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# CELL 17 - ROC CURVE


RocCurveDisplay.from_predictions(y_test, best_prob)
plt.title("ROC Curve - Selected Classifier")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classification_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# CELL 18 - FEATURE IMPORTANCE


feature_importance_df = (
    pd.DataFrame({
        "Feature": best_X_train.columns,
        "Importance": best_model.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

print("\nFeature importance:")
print(feature_importance_df.round(4))

feature_importance_df.to_csv(
    RESULTS_DIR / "classification_feature_importance.csv",
    index=False,
)

plt.figure(figsize=(8, 5))
plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"],
)
plt.gca().invert_yaxis()
plt.title("Feature Importance - Selected Classifier")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classification_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# CELL 19 - SHAP EXPLAINABILITY


print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(best_X_test)

# Handle binary-classification SHAP output
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_to_plot = shap_values[:, :, 1]
else:
    shap_values_to_plot = shap_values

shap_importance_df = (
    pd.DataFrame({
        "Feature": best_X_test.columns,
        "Mean absolute SHAP": np.abs(shap_values_to_plot).mean(axis=0),
    })
    .sort_values("Mean absolute SHAP", ascending=False)
    .reset_index(drop=True)
)

print("\nSHAP feature importance:")
print(shap_importance_df.round(4))

shap_importance_df.to_csv(
    RESULTS_DIR / "classification_shap_importance.csv",
    index=False,
)

shap.summary_plot(
    shap_values_to_plot,
    best_X_test,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "classification_shap_bar.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

shap.summary_plot(
    shap_values_to_plot,
    best_X_test,
    show=False,
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "classification_shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()



# CELL 20 - SMOTE EXPERIMENT


smote_feature_sets = {
    "Meteorology only": FEATURES_METEOROLOGY_ONLY,
    "Meteorology + lag": FEATURES_METEOROLOGY_LAG,
}

# k=5 requires at least six high-PM2.5 observations
# in every chronological training fold
for fold, (train_idx, _) in enumerate(
    tscv.split(X_train_met_lag),
    start=1,
):
    minority_count = int(y_train.iloc[train_idx].sum())

    print(
        f"Fold {fold} training high days:",
        minority_count,
    )

    assert minority_count >= 6, (
        "Not enough high-PM2.5 observations "
        "for SMOTE k_neighbors=5."
    )

smote_models = {
    "SMOTE Random Forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            bootstrap=True,
        ),
        "param_dist": {
            "model__n_estimators": randint(300, 1001),
            "model__max_depth": [5, 8, 10, 12, 15, 20, None],
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 8),
            "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_samples": [None, 0.7, 0.85],
        },
    },
    "SMOTE Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_dist": {
            "model__n_estimators": randint(100, 601),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(1, 6),
            "model__min_samples_split": randint(2, 25),
            "model__min_samples_leaf": randint(1, 12),
            "model__subsample": uniform(0.6, 0.4),
            "model__max_features": [
                None,
                "sqrt",
                "log2",
                0.5,
                0.7,
                1.0,
            ],
        },
    },
}

smote_searches = {}
smote_cv_rows = []

for model_name, model_info in smote_models.items():
    for feature_set_name, columns in smote_feature_sets.items():

        X_train_current = train_data[columns]

        # Scaling and SMOTE stay inside each CV training fold
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    sampling_strategy=1.0,
                    k_neighbors=5,
                ),
            ),
            ("model", model_info["model"]),
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_info["param_dist"],
            n_iter=150,
            scoring="f1",
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            return_train_score=True,
        )

        search.fit(X_train_current, y_train)

        key = (model_name, feature_set_name)
        smote_searches[key] = search

        best_index = search.best_index_
        training_f1 = search.cv_results_["mean_train_score"][best_index]
        validation_f1 = search.best_score_
        validation_std = search.cv_results_["std_test_score"][best_index]

        smote_cv_rows.append({
            "Model": model_name,
            "Feature set": feature_set_name,
            "Training CV F1": training_f1,
            "Validation CV F1": validation_f1,
            "CV F1 standard deviation": validation_std,
            "Train-validation gap": training_f1 - validation_f1,
        })

        print(f"\n{model_name} - {feature_set_name}")
        print("Best parameters:", search.best_params_)
        print("Validation CV F1:", round(validation_f1, 3))

smote_cv_df = (
    pd.DataFrame(smote_cv_rows)
    .sort_values(
        by=[
            "Validation CV F1",
            "CV F1 standard deviation",
            "Train-validation gap",
        ],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
)

print("\nSMOTE model selection using 2021-2022 only:")
print(smote_cv_df.round(3))

smote_cv_df.to_csv(
    RESULTS_DIR / "classification_smote_cv_results.csv",
    index=False,
)



# CELL 21 - FINAL SMOTE EVALUATION ON 2023


smote_results = []

for (model_name, feature_set_name), search in smote_searches.items():
    columns = smote_feature_sets[feature_set_name]
    X_test_current = test_data[columns]

    fitted_pipeline = search.best_estimator_
    y_pred = fitted_pipeline.predict(X_test_current)
    y_prob = fitted_pipeline.predict_proba(X_test_current)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    smote_results.append({
        "Model": model_name,
        "Feature set": feature_set_name,
        "Actual high days": int(y_test.sum()),
        "Predicted high days": int(y_pred.sum()),
        "Correctly detected high days": int(tp),
        "False positives": int(fp),
        "Missed high days": int(fn),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average precision": average_precision_score(y_test, y_prob),
    })

smote_results_df = pd.DataFrame(smote_results)

print("\nFinal 2023 SMOTE results:")
print(smote_results_df.round(3))

smote_results_df.to_csv(
    RESULTS_DIR / "classification_smote_results.csv",
    index=False,
)



# CELL 22 - FINAL SUMMARY


selected_result = all_model_results[
    all_model_results["Selected by development CV"]
].iloc[0]

print("\n========================================")
print("FINAL CLASSIFICATION SUMMARY")
print("========================================")

print("Model:", selected_model_name)
print("Feature set:", selected_feature_set)
print(
    "Balanced Accuracy:",
    round(selected_result["Balanced accuracy"], 3),
)
print(
    "Precision:",
    round(selected_result["Precision"], 3),
)
print(
    "Recall:",
    round(selected_result["Recall"], 3),
)
print(
    "F1:",
    round(selected_result["F1-score"], 3),
)
print(
    "ROC-AUC:",
    round(selected_result["ROC-AUC"], 3),
)
print(
    "Correctly detected high days:",
    int(tp_selected),
    "/",
    int(y_test.sum()),
)

print("\nResults saved to:", RESULTS_DIR)

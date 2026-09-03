from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)



def evaluate_classifier(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    feature_set_name,
    sample_weight=None
):
    # make a fresh copy of model so it does not carry over old information
    fitted_model = clone(model)

    # train the model
    if sample_weight is not None:
        fitted_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight
        )
    else:
        fitted_model.fit(X_train, y_train)

    # class predictions
    y_pred = fitted_model.predict(X_test)

    # probability of high-PM2.5 class
    y_prob = fitted_model.predict_proba(X_test)[:, 1]

    # evaluation metrics
    result = {
        "Model": model_name,
        "Feature set": feature_set_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average precision": average_precision_score(
            y_test,
            y_prob
        )
    }

    output = {
        "model": fitted_model,
        "pred": y_pred,
        "prob": y_prob,
        "X_train": X_train,
        "X_test": X_test
    }

    return result, output

def tune_classifier(
    model,
    param_dist,
    X_train,
    y_train,
    cv,
    sample_weight=None,
    n_iter=100
):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    if sample_weight is not None:
        search.fit(
            X_train,
            y_train,
            sample_weight=sample_weight
        )
    else:
        search.fit(X_train, y_train)

    return search








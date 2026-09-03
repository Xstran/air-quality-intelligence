import numpy as np

from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import RandomizedSearchCV



def evaluate_regressor(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    feature_set_name
):
    # Create a fresh copy of the model
    fitted_model = clone(model)

    # Train using development data
    fitted_model.fit(X_train, y_train)

    # Generate holdout predictions
    y_pred = fitted_model.predict(X_test)

    # Calculate regression metrics
    result = {
        "Model": model_name,
        "Feature set": feature_set_name,
        "RMSE": np.sqrt(
            mean_squared_error(y_test, y_pred)
        ),
        "MAE": mean_absolute_error(
            y_test,
            y_pred
        ),
        "R2": r2_score(
            y_test,
            y_pred
        )
    }

    # Store fitted model and predictions
    output = {
        "model": fitted_model,
        "pred": y_pred,
        "X_train": X_train,
        "X_test": X_test
    }

    return result, output



def tune_regressor(
    model,
    param_dist,
    X_train,
    y_train,
    cv,
    n_iter=100
):
    # Randomised hyperparameter search using time-aware CV
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    return search

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

from assignment_q5 import train_test_split_gestures_v3


def train_knn():
    train_df, test_df = train_test_split_gestures_v3()
    x_train = train_df.drop(["label"], axis=1)
    y_train = train_df["label"]
    x_test = test_df.drop(["label"], axis=1)
    y_test = test_df["label"]

    def objective(params):
        cv = TimeSeriesSplit(n_splits=10)
        model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                     weights=params["weights"],
                                     # p=int(params["p"]),
                                     metric=params["metric"])
        scoring = {"f1_score": make_scorer(f1_score, average="weighted", zero_division=1),
                   "precision": make_scorer(precision_score, average="weighted", zero_division=1),
                   "recall": make_scorer(recall_score, average="weighted", zero_division=1),
                   "accuracy": make_scorer(accuracy_score)}
        error = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)
        f1 = error["test_f1_score"].mean()
        return {"loss": -f1, "status": STATUS_OK}

    weights = ["uniform", "distance"]
    metrics = ['euclidean', 'manhattan']
    params = {
        "n_neighbors": hp.quniform("n_neighbors", 1, 100, 1),
        "weights": hp.choice("weights", weights),
        # "p": hp.quniform("p", 1, 5, 1),
        "metric": hp.choice("metric", metrics),
    }

    trials = Trials()
    best_params = fmin(fn=objective,
                       space=params,
                       algo=tpe.suggest,
                       max_evals=10,
                       trials=trials)

    print(f"Best n_neighbors: {best_params['n_neighbors']}")
    print(f"Best weights: {weights[best_params['weights']]}")
    # print(f"Best p: {best_params['p']}")
    print(f"Best metric: {metrics[best_params['metric']]}")

    best_knn = KNeighborsClassifier(n_neighbors=int(best_params["n_neighbors"]),
                                    weights=weights[best_params["weights"]],
                                    # p=int(best_params["p"]),
                                    metric=metrics[best_params["metric"]])
    best_knn.fit(x_train, y_train)
    y_pred = best_knn.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"F1-score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")


train_knn()


def train_knn_gridsearch():
    train_df, test_df = train_test_split_gestures_v3()
    x_train = train_df.drop(["label"], axis=1)
    y_train = train_df["label"]
    x_test = test_df.drop(["label"], axis=1)
    y_test = test_df["label"]

    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    cv = TimeSeriesSplit(n_splits=10)
    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_model, param_grid, cv=cv)
    grid_search.fit(x_train, y_train)

    best_knn_model = grid_search.best_estimator_
    y_pred = best_knn_model.predict(x_test)

    best_params = grid_search.best_params_
    print("\nBest Hyperparameters:", best_params)

    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"F1-score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")


train_knn_gridsearch()

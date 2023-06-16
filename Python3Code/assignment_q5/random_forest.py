from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from assignment_q5 import train_test_split_gestures


def train_random_forest():
    train_df, test_df = train_test_split_gestures()
    x_train = train_df.drop(["label"], axis=1)
    y_train = train_df["label"]
    x_test = test_df.drop(["label"], axis=1)
    y_test = test_df["label"]

    def objective(params):
        cv = TimeSeriesSplit(n_splits=10)
        model = RandomForestClassifier(n_estimators=int(params["n_estimators"]),
                                       max_features=params["max_features"],
                                       max_depth=int(params["max_depth"]),
                                       min_samples_split=int(params["min_samples_split"]),
                                       min_samples_leaf=int(params["min_samples_leaf"]))
        scoring = {"f1_score": make_scorer(f1_score, average="macro", zero_division=1),
                   "precision": make_scorer(precision_score, average="macro", zero_division=1),
                   "recall": make_scorer(recall_score, average="macro", zero_division=1),
                   "accuracy": make_scorer(accuracy_score)}
        error = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)
        f1 = error["test_f1_score"].mean()
        return {"loss": -f1, "status": STATUS_OK}

    n_estimators = [100]
    max_features = ["sqrt", "log2"]
    params = {
        "n_estimators": hp.choice("n_estimators", n_estimators),
        "max_features": hp.choice("max_features", max_features),
        "max_depth": hp.quniform("max_depth", 5, 40, 5),
        "min_samples_split": hp.quniform("min_samples_split", 5, 25, 5),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1)
    }

    trials = Trials()
    best_params = fmin(fn=objective,
                       space=params,
                       algo=tpe.suggest,
                       max_evals=10,
                       trials=trials)

    print(f"Best n_estimators: {n_estimators[best_params['n_estimators']]}")
    print(f"Best max_features: {max_features[best_params['max_features']]}")
    print(f"Best max_depth: {best_params['max_depth']}")
    print(f"Best min_samples_split: {best_params['min_samples_split']}")
    print(f"Best min_samples_leaf: {best_params['min_samples_leaf']}")

    best_rf = RandomForestClassifier(n_estimators=n_estimators[int(best_params["n_estimators"])],
                                     max_features=max_features[best_params['max_features']],
                                     max_depth=int(best_params['max_depth']),
                                     min_samples_split=int(best_params['min_samples_split']),
                                     min_samples_leaf=int(best_params['min_samples_leaf']))
    best_rf.fit(x_train, y_train)
    y_pred = best_rf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"F1-score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")


train_random_forest()

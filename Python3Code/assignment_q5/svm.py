from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from assignment_q5 import train_test_split_gestures_v3


def train_svm():
    train_df, test_df = train_test_split_gestures_v3()
    x_train = train_df.drop(["label"], axis=1)
    y_train = train_df["label"]
    x_test = test_df.drop(["label"], axis=1)
    y_test = test_df["label"]

    def objective(params):
        cv = TimeSeriesSplit(n_splits=10)
        model = SVC(kernel=params["kernel"],
                    C=params["C"],
                    gamma=params["gamma"])
        scoring = {"f1_score": make_scorer(f1_score, average="macro", zero_division=1),
                   "precision": make_scorer(precision_score, average="macro", zero_division=1),
                   "recall": make_scorer(recall_score, average="macro", zero_division=1),
                   "accuracy": make_scorer(accuracy_score)}
        error = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)
        f1 = error["test_f1_score"].mean()
        return {"loss": -f1, "status": STATUS_OK}

    kernel = ["rbf"]
    C = [0.1, 1., 10., 100.]
    gamma = ["auto", 0.01, 0.1, 1., 10.]
    params = {
        "kernel": hp.choice("kernel", kernel),
        "C": hp.choice("C", C),
        "gamma": hp.choice("gamma", gamma)
    }

    trials = Trials()
    best_params = fmin(fn=objective,
                       space=params,
                       algo=tpe.suggest,
                       max_evals=10,
                       trials=trials)

    print(f"Best kernel: {kernel[best_params['kernel']]}")
    print(f"Best C: {C[best_params['C']]}")
    print(f"Best gamma: {gamma[best_params['gamma']]}")

    best_rf = SVC(kernel=best_params["kernel"],
                  C=best_params["C"],
                  gamma=best_params["gamma"])
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


train_svm()

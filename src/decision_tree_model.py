from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import time

def train_decision_tree(X_train, y_train, X_test, y_test, feature_names, seed=66):
    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(f"Decision Tree Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(10, 5))
    plot_tree(
        clf, max_depth=2, feature_names=feature_names,
        class_names=["Healthy", "Sick"], filled=True, rounded=True
    )
    plt.title("Decision Tree (First Two Layers)")
    plt.show()

    return clf, acc, cm

def optimize_decision_tree(X_train, y_train, X_test, y_test, seed=66):
    param_grid = {
        "max_depth": [4, 8, 16, 32, 64],
        "min_samples_split": [4, 8, 16, 32],
        "criterion": ["gini", "entropy", "log_loss"]
    }
    base_tree = DecisionTreeClassifier(random_state=seed)
    grid = GridSearchCV(base_tree, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_tree = grid.best_estimator_
    preds = best_tree.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    cm = metrics.confusion_matrix(y_test, preds)

    print(f"\nBest Decision Tree Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", cm)
    return best_tree, acc, cm, grid.best_params_

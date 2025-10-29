from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time

def train_mlp(X_train, y_train, X_test, y_test, seed=66):
    clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(f"MLP Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", cm)
    return clf, acc, cm

def compare_speed(X_train, y_train, X_test, y_test, seed=66):
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(random_state=seed)
    mlp = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, random_state=seed)

    # Fit timing
    t0 = time.time(); tree.fit(X_train, y_train); t1 = time.time()
    t2 = time.time(); mlp.fit(X_train, y_train); t3 = time.time()

    # Predict timing
    p0 = time.time(); tree.predict(X_test); p1 = time.time()
    p2 = time.time(); mlp.predict(X_test); p3 = time.time()

    print(f"Decision Tree Train Time: {t1-t0:.4f}s | Predict Time: {p1-p0:.6f}s")
    print(f"MLP Train Time: {t3-t2:.4f}s | Predict Time: {p3-p2:.6f}s")

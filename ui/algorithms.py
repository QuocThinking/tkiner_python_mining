from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

def run_algorithm(data, algo, k=3):
    X = data[["math_score", "physics_score", "chemistry_score"]].values
    y = data["label"].values

    if algo == "Naive Bayes":
        model = GaussianNB()
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        result = {
            "text": f"Naive Bayes Accuracy: {accuracy:.2f}",
            "confusion_matrix": cm.tolist()  # Trả về ma trận dạng list
        }

    elif algo == "KNN":
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        result = {
            "text": f"KNN (k={k}) Accuracy: {accuracy:.2f}",
            "confusion_matrix": cm.tolist()
        }

    elif algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        labels = model.labels_
        result = {
            "text": f"K-Means Clustering (k={k}) completed",
            "confusion_matrix": None  # K-Means không có ma trận nhầm lẫn
        }

    return result
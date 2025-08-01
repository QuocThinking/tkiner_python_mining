import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def run_algorithm(data, algo, k=3, use_reduct=False):
    """
    Chạy thuật toán phân loại hoặc phân cụm trên dữ liệu học sinh.
    
    Args:
        data: DataFrame chứa các cột math_score, physics_score, chemistry_score, label
        algo: Tên thuật toán ("Naive Bayes", "KNN", "K-Means", "Decision Tree")
        k: Số láng giềng (KNN) hoặc số cụm (K-Means)
        use_reduct: Boolean, áp dụng giảm chiều dữ liệu bằng PCA nếu True
    
    Returns:
        dict: Kết quả bao gồm text, confusion_matrix (nếu có), và các chỉ số bổ sung
    """
    # Kiểm tra dữ liệu đầu vào
    required_columns = ["math_score", "physics_score", "chemistry_score", "label"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Dữ liệu thiếu một hoặc nhiều cột: math_score, physics_score, chemistry_score, label")
    
    if data[required_columns].isnull().any().any():
        raise ValueError("Dữ liệu chứa giá trị null")
    
    valid_labels = {"Giỏi", "Trung bình", "Yếu"}
    if not set(data["label"]).issubset(valid_labels):
        raise ValueError("Nhãn không hợp lệ, chỉ chấp nhận: Giỏi, Trung bình, Yếu")
    
    # Chuẩn bị dữ liệu
    X = data[["math_score", "physics_score", "chemistry_score"]].values
    y = data["label"].values
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Áp dụng reduct nếu được yêu cầu
    if use_reduct:
        pca = PCA(n_components=2)
        X_scaled = pca.fit_transform(X_scaled)
        reduct_info = {
            "reduced_features": X_scaled,
            "explained_variance_ratio": pca.explained_variance_ratio_
        }
    else:
        reduct_info = None
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled if use_reduct else X, y, test_size=0.2, random_state=42, stratify=y)
    
    result = {
        "text": "",
        "confusion_matrix": None,
        "classification_report": None,
        "train_accuracy": None,
        "test_accuracy": None,
        "execution_time": None,
        "additional_info": {},
        "reduct_info": reduct_info
    }
    
    start_time = time.time()
    
    try:
        if algo == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train, y_train)
            
            # Dự đoán và đánh giá
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            
            result.update({
                "text": f"Naive Bayes - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}",
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })
        
        elif algo == "KNN":
            if k <= 0:
                raise ValueError("k phải là số nguyên dương")
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            
            # Dự đoán và đánh giá
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            
            result.update({
                "text": f"KNN (k={k}) - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}",
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })
        
        elif algo == "K-Means":
            if k <= 0:
                raise ValueError("k phải là số nguyên dương")
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(X_scaled if use_reduct else X)
            
            labels = model.labels_
            silhouette = silhouette_score(X_scaled if use_reduct else X, labels) if len(set(labels)) > 1 else 0.0
            cluster_centers = model.cluster_centers_.tolist()
            
            # Ánh xạ cụm với nhãn gốc
            cluster_label_map = {}
            for cluster in range(k):
                cluster_mask = labels == cluster
                if cluster_mask.sum() > 0:
                    most_common_label = pd.Series(y[cluster_mask]).mode()[0]
                    cluster_label_map[cluster] = most_common_label
            
            result.update({
                "text": f"K-Means (k={k}) - Silhouette Score: {silhouette:.2f}",
                "confusion_matrix": None,
                "additional_info": {
                    "cluster_centers": cluster_centers,
                    "cluster_label_map": cluster_label_map,
                    "cluster_distribution": pd.Series(labels).value_counts().to_dict()
                }
            })
        
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42, max_depth=5)
            model.fit(X_train, y_train)
            
            # Dự đoán và đánh giá
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Tầm quan trọng đặc trưng (dựa trên Gini Index)
            feature_names = ["math_score", "physics_score", "chemistry_score"]
            if use_reduct:
                feature_names = [f"PC{i+1}" for i in range(2)]
            feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            
            result.update({
                "text": f"Decision Tree - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}",
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "additional_info": {"feature_importance": feature_importance}
            })
        
        else:
            raise ValueError(f"Thuật toán {algo} không được hỗ trợ")
        
        result["execution_time"] = time.time() - start_time
        return result
    
    except Exception as e:
        raise ValueError(f"Lỗi khi chạy thuật toán {algo}: {str(e)}")
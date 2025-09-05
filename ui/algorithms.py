from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def run_association_rules(data, min_support=0.1, min_confidence=0.5):
    """
    Chạy thuật toán Apriori để tìm luật kết hợp từ dữ liệu.
    
    Args:
        data: DataFrame chứa các cột số (features) và tùy chọn cột 'label'
        min_support: Ngưỡng hỗ trợ tối thiểu
        min_confidence: Ngưỡng độ tin cậy tối thiểu
    
    Returns:
        dict: Kết quả bao gồm luật kết hợp và các chỉ số thống kê
    """
    features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    if len(features) == 0:
        raise ValueError("Không có cột features số nào")
    
    if data[features].isnull().any().any():
        raise ValueError("Dữ liệu chứa giá trị null")
    
    if len(data) == 0:
        raise ValueError("Dữ liệu rỗng, không thể chạy Association Rules")
    
    print(f"Debug: Running Association Rules with data shape={data.shape}, features={features}")
    
    def discretize_scores(score):
        if score < 5:
            return "<5"
        elif 5 <= score <= 8:
            return "5-8"
        else:
            return ">8"
    
    transactions = []
    has_label = 'label' in data.columns
    for _, row in data.iterrows():
        transaction = [f"{col}_{discretize_scores(row[col])}" for col in features]
        if has_label:
            transaction.append(f"Label_{row['label']}")
        transactions.append(transaction)
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        raise ValueError("Không tìm thấy tập phổ biến với ngưỡng hỗ trợ hiện tại")
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="confidence", ascending=False).head(5)
    
    start_time = time.time()
    result = {
        "text": f"Association Rules - Tìm {len(rules)} quy tắc với min_support={min_support}, min_confidence={min_confidence}",
        "rules": rules.to_dict() if not rules.empty else "Không có quy tắc nào",
        "execution_time": time.time() - start_time
    }

    print(f"Debug: Frequent itemsets found: {len(frequent_itemsets)}")
    print(f"Debug: Rules generated: {len(rules)}")
    
    return result

def run_algorithm(data, algo, k=3, use_reduct=False, n_components=2):
    """
    Chạy thuật toán phân loại, phân cụm hoặc luật kết hợp trên dữ liệu.
    
    Args:
        data: DataFrame chứa các cột số (features) và tùy chọn cột 'label'
        algo: Tên thuật toán ("Naive Bayes", "KNN", "K-Means", "Decision Tree", "ID3", "Association Rules")
        k: Số láng giềng (KNN) hoặc số cụm (K-Means)
        use_reduct: Boolean, áp dụng giảm chiều dữ liệu bằng PCA nếu True
        n_components: Số thành phần chính khi giảm chiều
    
    Returns:
        dict: Kết quả bao gồm text, confusion_matrix (nếu có), và các chỉ số bổ sung
    """
    features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    if len(features) == 0:
        raise ValueError("Không có cột features số nào")
    
    if data[features].isnull().any().any():
        raise ValueError("Dữ liệu chứa giá trị null")
    
    has_label = 'label' in data.columns
    if algo in ["Naive Bayes", "KNN", "Decision Tree", "ID3"] and not has_label:
        raise ValueError(f"Thuật toán {algo} yêu cầu cột 'label'")
    
    X = data[features].values
    y = data["label"].values if has_label else None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if use_reduct and algo != "Association Rules":
        n_components = min(n_components, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        reduct_info = {
            "reduced_features": X_scaled,
            "explained_variance_ratio": pca.explained_variance_ratio_
        }
        print(f"Debug: reduct_info created - use_reduct={use_reduct}, algo={algo}, n_components={n_components}")
    else:
        reduct_info = None
        print(f"Debug: reduct_info is None - use_reduct={use_reduct}, algo={algo}")
    
    start_time = time.time()
    result = {"reduct_info": reduct_info}
    
    if algo in ["Naive Bayes", "KNN", "Decision Tree", "ID3"]:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled if use_reduct else X, y, test_size=0.2, random_state=42)
        
        if algo == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train, y_train)
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
        
        elif algo in ["Decision Tree", "ID3"]:
            criterion = 'entropy' if algo == "ID3" else 'gini'
            model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion=criterion)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            feature_names = features if not use_reduct else [f"PC{i+1}" for i in range(n_components)]
            feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            result.update({
                "text": f"{algo} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}",
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "additional_info": {"feature_importance": feature_importance}
            })
    
    elif algo == "K-Means":
        if k <= 0:
            raise ValueError("k phải là số nguyên dương")
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled if use_reduct else X)
        labels = model.labels_
        silhouette = silhouette_score(X_scaled if use_reduct else X, labels) if len(set(labels)) > 1 else 0.0
        cluster_centers = model.cluster_centers_.tolist()
        
        cluster_label_map = {}
        if has_label:
            for cluster in range(k):
                cluster_mask = labels == cluster
                if cluster_mask.sum() > 0:
                    most_common_label = pd.Series(y[cluster_mask]).mode()[0]
                    cluster_label_map[cluster] = most_common_label
        else:
            cluster_label_map = {i: f"Cluster {i}" for i in range(k)}
        
        result.update({
            "text": f"K-Means (k={k}) - Silhouette Score: {silhouette:.2f}",
            "confusion_matrix": None,
            "additional_info": {
                "cluster_centers": cluster_centers,
                "cluster_label_map": cluster_label_map,
                "cluster_distribution": pd.Series(labels).value_counts().to_dict()
            }
        })
    
    elif algo == "Association Rules":
        temp_result = run_association_rules(data, min_support=0.1, min_confidence=0.5)
        result.update(temp_result)
    
    else:
        raise ValueError(f"Thuật toán {algo} không được hỗ trợ")
    
    print(f"Debug: Final result for algo={algo}, reduct_info={result['reduct_info']}")
    result["execution_time"] = time.time() - start_time
    return result
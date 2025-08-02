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
from itertools import combinations

def run_algorithm(data, algo, k=3, use_reduct=False, minsupp=0.5, minconf=0.7):
    """
    Chạy thuật toán phân loại, phân cụm, luật kết hợp hoặc tập rút gọn trên dữ liệu.
    
    Args:
        data: DataFrame chứa dữ liệu
        algo: Tên thuật toán ("Naive Bayes", "KNN", "K-Means", "Decision Tree", "Association Rules", "Reduct")
        k: Số láng giềng (KNN) hoặc số cụm (K-Means)
        use_reduct: Boolean, áp dụng giảm chiều dữ liệu bằng PCA nếu True
        minsupp: Ngưỡng độ hỗ trợ tối thiểu cho luật kết hợp
        minconf: Ngưỡng độ tin cậy tối thiểu cho luật kết hợp
    
    Returns:
        dict: Kết quả bao gồm text, confusion_matrix (nếu có), và các chỉ số bổ sung
    """
    
    if algo in ["Association Rules", "Apriori"]:
        return run_association_rules(data, minsupp=0.2, minconf=0.5)
    elif algo in ["Reduct", "Rough Set"]:
        return run_reduct_algorithm(data)
    else:
        # Code hiện tại cho các thuật toán ML
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
        X_train, X_test, y_train, y_test = train_test_split(X_scaled if use_reduct else X, y, test_size=3, random_state=42, stratify=y)
        
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

def run_association_rules(data, minsupp=0.5, minconf=0.7):
    """
    Thuật toán Apriori để tìm tập phổ biến và luật kết hợp.
    
    Args:
        data: DataFrame chứa dữ liệu giao dịch (dạng nhị phân hoặc itemset)
        minsupp: Ngưỡng độ hỗ trợ tối thiểu
        minconf: Ngưỡng độ tin cậy tối thiểu
    
    Returns:
        dict: Kết quả chứa tập phổ biến và luật kết hợp
    """
    start_time = time.time()
    
    try:
        # Chuyển đổi dữ liệu thành dạng giao dịch nhị phân
        if 'transaction_id' in data.columns:
            # Dữ liệu dạng transaction_id, item
            transactions = data.groupby('transaction_id')['item'].apply(list).values
        else:
            # Dữ liệu đã ở dạng ma trận nhị phân
            transactions = []
            for _, row in data.iterrows():
                transaction = []
                for col in data.columns:
                    if row[col] == 1:
                        transaction.append(col)
                transactions.append(transaction)
        
        # Tìm tất cả các item duy nhất
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        all_items = sorted(list(all_items))
        
        # Thuật toán Apriori
        frequent_itemsets = apriori_algorithm(transactions, all_items, minsupp)
        
        # Tìm tập phổ biến tối đại
        maximal_itemsets = find_maximal_itemsets(frequent_itemsets)
        
        # Tạo luật kết hợp
        association_rules = generate_association_rules(frequent_itemsets, transactions, minconf)
        
        # Tính thống kê
        total_transactions = len(transactions)
        total_frequent_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
        
        result_text = f"Apriori Algorithm Results:\n"
        result_text += f"- Total transactions: {total_transactions}\n"
        result_text += f"- Minimum support: {minsupp}\n"
        result_text += f"- Minimum confidence: {minconf}\n"
        result_text += f"- Total frequent itemsets: {total_frequent_itemsets}\n"
        result_text += f"- Maximal frequent itemsets: {len(maximal_itemsets)}\n"
        result_text += f"- Association rules generated: {len(association_rules)}\n"
        
        execution_time = time.time() - start_time
        
        return {
            "text": result_text,
            "execution_time": execution_time,
            "frequent_itemsets": frequent_itemsets,
            "maximal_itemsets": maximal_itemsets,
            "association_rules": association_rules,
            "additional_info": {
                "total_transactions": total_transactions,
                "algorithm": "Apriori",
                "parameters": {
                    "minsupp": minsupp,
                    "minconf": minconf
                }
            }
        }
        
    except Exception as e:
        raise ValueError(f"Lỗi khi chạy thuật toán Association Rules: {str(e)}")

def apriori_algorithm(transactions, all_items, minsupp):
    """Thuật toán Apriori chính"""
    frequent_itemsets = {}
    total_transactions = len(transactions)
    
    # Tìm tập phổ biến kích thước 1
    L1 = {}
    for item in all_items:
        support = sum(1 for transaction in transactions if item in transaction) / total_transactions
        if support >= minsupp:
            L1[frozenset([item])] = support
    
    frequent_itemsets[1] = L1
    k = 2
    
    # Tiếp tục tìm tập phổ biến kích thước k
    while frequent_itemsets.get(k-1):
        # Sinh ứng cử viên kích thước k
        candidates = generate_candidates(list(frequent_itemsets[k-1].keys()), k)
        
        # Tính support cho từng ứng cử viên
        Lk = {}
        for candidate in candidates:
            support = sum(1 for transaction in transactions if candidate.issubset(set(transaction))) / total_transactions
            if support >= minsupp:
                Lk[candidate] = support
        
        if Lk:
            frequent_itemsets[k] = Lk
            k += 1
        else:
            break
    
    return frequent_itemsets

def generate_candidates(prev_frequent, k):
    """Sinh ứng cử viên kích thước k từ tập phổ biến kích thước k-1"""
    candidates = []
    n = len(prev_frequent)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Kết hợp hai tập có (k-2) phần tử chung đầu tiên
            set1 = sorted(list(prev_frequent[i]))
            set2 = sorted(list(prev_frequent[j]))
            
            if set1[:-1] == set2[:-1]:
                candidate = frozenset(set1 + [set2[-1]])
                candidates.append(candidate)
    
    return candidates

def find_maximal_itemsets(frequent_itemsets):
    """Tìm tập phổ biến tối đại"""
    all_frequent = []
    for k in frequent_itemsets:
        all_frequent.extend(frequent_itemsets[k].keys())
    
    maximal = []
    for itemset in all_frequent:
        is_maximal = True
        for other in all_frequent:
            if itemset != other and itemset.issubset(other):
                is_maximal = False
                break
        if is_maximal:
            maximal.append(itemset)
    
    return maximal

def generate_association_rules(frequent_itemsets, transactions, minconf):
    """Tạo luật kết hợp từ tập phổ biến"""
    rules = []
    total_transactions = len(transactions)
    
    for k in frequent_itemsets:
        if k > 1:  # Chỉ tạo luật từ tập có ít nhất 2 phần tử
            for itemset in frequent_itemsets[k]:
                itemset_support = frequent_itemsets[k][itemset]
                
                # Tạo tất cả các tập con không rỗng làm antecedent
                items = list(itemset)
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Tính confidence
                        antecedent_support = 0
                        for size in frequent_itemsets:
                            if antecedent in frequent_itemsets[size]:
                                antecedent_support = frequent_itemsets[size][antecedent]
                                break
                        
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            if confidence >= minconf:
                                rules.append({
                                    'antecedent': set(antecedent),
                                    'consequent': set(consequent),
                                    'support': itemset_support,
                                    'confidence': confidence
                                })
    
    return rules

def run_reduct_algorithm(data):
    """
    Thuật toán tìm tập rút gọn (Reduct) dựa trên lý thuyết tập thô.
    
    Args:
        data: DataFrame chứa các thuộc tính điều kiện và thuộc tính quyết định
    
    Returns:
        dict: Kết quả chứa tập rút gọn và thông tin xấp xỉ
    """
    start_time = time.time()
    
    try:
        # Kiểm tra dữ liệu đầu vào
        if 'decision' not in data.columns:
            raise ValueError("Dữ liệu phải có cột 'decision' làm thuộc tính quyết định")
        
        condition_attrs = [col for col in data.columns if col in ['math_score', 'physics_score', 'chemistry_score']]
        if len(condition_attrs) == 0:
            raise ValueError("Dữ liệu phải có ít nhất một thuộc tính điều kiện")
        print(f"Condition Attributes: {condition_attrs}")
        print(f"Data Columns: {data.columns}")
        
        # Tạo ma trận phân biệt
        discernibility_matrix = create_discernibility_matrix(data, condition_attrs)
        
        # Tìm hàm phân biệt
        discernibility_function = create_discernibility_function(discernibility_matrix)
        
        # Rút gọn hàm phân biệt để tìm reducts
        reducts = find_reducts(discernibility_function, condition_attrs)
        
        # Tính các xấp xỉ cho từng lớp quyết định
        decision_classes = data['decision'].unique()
        approximations = {}
        
        for decision_class in decision_classes:
            class_objects = set(data[data['decision'] == decision_class].index)
            lower_approx, upper_approx = compute_approximations(data, condition_attrs, class_objects)
            
            approximations[decision_class] = {
                'lower_approximation': lower_approx,
                'upper_approximation': upper_approx,
                'boundary_region': upper_approx - lower_approx,
                'accuracy': len(lower_approx) / len(upper_approx) if len(upper_approx) > 0 else 1.0
            }
        
        # Tính độ phụ thuộc tổng thể
        total_lower = sum(len(approx['lower_approximation']) for approx in approximations.values())
        dependency_degree = total_lower / len(data)
        
        result_text = f"Rough Set Reduct Analysis:\n"
        result_text += f"- Total objects: {len(data)}\n" 
        result_text += f"- Condition attributes: {len(condition_attrs)}\n"
        result_text += f"- Decision classes: {len(decision_classes)}\n"
        result_text += f"- Dependency degree: {dependency_degree:.3f}\n"
        result_text += f"- Number of reducts found: {len(reducts)}\n"
        
        if reducts:
            min_reduct_size = min(len(reduct) for reduct in reducts)
            result_text += f"- Minimal reduct size: {min_reduct_size}\n"
        
        execution_time = time.time() - start_time
        
        return {
            "text": result_text,
            "execution_time": execution_time,
            "reducts": [list(reduct) for reduct in reducts],
            "discernibility_matrix": discernibility_matrix,
            "approximations": approximations,
            "dependency_degree": dependency_degree,
            "additional_info": {
                "condition_attributes": condition_attrs,
                "decision_classes": list(decision_classes),
                "algorithm": "Rough Set Reduct"
            }
        }
        
    except Exception as e:
        raise ValueError(f"Lỗi khi chạy thuật toán Reduct: {str(e)}")

def create_discernibility_matrix(data, condition_attrs):
    """Tạo ma trận phân biệt"""
    n = len(data)
    matrix = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            if data.iloc[i]['decision'] != data.iloc[j]['decision']:
                # Chỉ quan tâm các đối tượng có quyết định khác nhau
                diff_attrs = []
                for attr in condition_attrs:
                    if data.iloc[i][attr] != data.iloc[j][attr]:
                        diff_attrs.append(attr)
                
                if diff_attrs:
                    matrix[(i, j)] = set(diff_attrs)
    
    print(f"Discernibility Matrix: {matrix}")  # Thêm dòng này để debug
    return matrix

def create_discernibility_function(discernibility_matrix):
    """Tạo hàm phân biệt từ ma trận phân biệt"""
    if not discernibility_matrix:
        return []
    
    # Mỗi phần tử trong ma trận tạo thành một clause (tuyển)
    clauses = list(discernibility_matrix.values())
    return clauses

def find_reducts(discernibility_function, condition_attrs):
    print(f"Discernibility Function: {discernibility_function}")  # Debug
    if not discernibility_function:
        return [set(condition_attrs)]  # Nếu không có hàm phân biệt, tất cả attributes đều cần thiết
    
    # Sử dụng thuật toán đơn giản để tìm minimal hitting set
    reducts = []
    
    # Tìm tất cả các tập con tối thiểu có thể cover tất cả clauses
    all_attrs = set(condition_attrs)
    
    # Thử từ kích thước nhỏ nhất
    for size in range(1, len(condition_attrs) + 1):
        for attr_combination in combinations(condition_attrs, size):
            attr_set = set(attr_combination)
            
            # Kiểm tra xem attr_set có cover được tất cả clauses không
            if all(not clause.isdisjoint(attr_set) for clause in discernibility_function):
                # Kiểm tra tính tối thiểu
                is_minimal = True
                for existing_reduct in reducts:
                    if existing_reduct.issubset(attr_set):
                        is_minimal = False
                        break
                
                if is_minimal:
                    # Loại bỏ các reducts không tối thiểu
                    reducts = [r for r in reducts if not attr_set.issubset(r)]
                    reducts.append(attr_set)
        
        # Nếu đã tìm được reduct, không cần tìm kích thước lớn hơn
        if reducts:
            break
    
    return reducts if reducts else [set(condition_attrs)]

def compute_approximations(data, attributes, target_set):
    """Tính xấp xỉ dưới và xấp xỉ trên cho một tập đối tượng"""
    # Tạo các lớp tương đương dựa trên attributes
    equivalence_classes = {}
    
    for idx, row in data.iterrows():
        # Tạo signature cho đối tượng dựa trên các attributes
        signature = tuple(row[attr] for attr in attributes)
        
        if signature not in equivalence_classes:
            equivalence_classes[signature] = set()
        equivalence_classes[signature].add(idx)
    
    # Tính xấp xỉ dưới và trên
    lower_approximation = set()
    upper_approximation = set()
    
    for eq_class in equivalence_classes.values():
        if eq_class.issubset(target_set):
            # Lớp tương đương hoàn toàn thuộc target_set
            lower_approximation.update(eq_class)
        
        if not eq_class.isdisjoint(target_set):
            # Lớp tương đương có giao với target_set
            upper_approximation.update(eq_class)
    
    return lower_approximation, upper_approximation
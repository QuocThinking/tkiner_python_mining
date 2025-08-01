import pandas as pd
import numpy as np

def generate_student_data(n=200, seed=42):
    """
    Tạo dữ liệu học sinh với điểm số và nhãn, lưu vào CSV và SQL.
    
    Args:
        n: Số lượng mẫu dữ liệu
        seed: Hạt giống ngẫu nhiên để tái tạo dữ liệu
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu học sinh
    """
    np.random.seed(seed)
    
    data = {
        "math_score": [],
        "physics_score": [],
        "chemistry_score": [],
        "label": []
    }
    
    labels = ["Giỏi", "Trung bình", "Yếu"]
    label_weights = [0.3, 0.4, 0.3]  # Tỷ lệ: 30% Giỏi, 40% Trung bình, 30% Yếu
    
    for _ in range(n):
        label = np.random.choice(labels, p=label_weights)
        
        if label == "Giỏi":
            math = np.random.uniform(8.0, 10.0)
            physics = math + np.random.normal(0, 0.5)
            chemistry = math + np.random.normal(0, 0.5)
        elif label == "Trung bình":
            math = np.random.uniform(5.0, 8.0)
            physics = math + np.random.normal(0, 0.7)
            chemistry = math + np.random.normal(0, 0.7)
        else:  # Yếu
            math = np.random.uniform(0.0, 5.0)
            physics = math + np.random.normal(0, 1.0)
            chemistry = math + np.random.normal(0, 1.0)
        
        math = round(max(0.0, min(10.0, math)), 1)
        physics = round(max(0.0, min(10.0, physics)), 1)
        chemistry = round(max(0.0, min(10.0, chemistry)), 1)
        
        avg_score = (math + physics + chemistry) / 3
        if avg_score >= 8:
            final_label = "Giỏi"
        elif avg_score >= 5:
            final_label = "Trung bình"
        else:
            final_label = "Yếu"
        
        data["math_score"].append(math)
        data["physics_score"].append(physics)
        data["chemistry_score"].append(chemistry)
        data["label"].append(final_label)
    
    df = pd.DataFrame(data)
    
    # Lưu vào file CSV
    df.to_csv("students_new.csv", index=False)
    
    # Tạo câu lệnh SQL
    sql_statements = [
        "CREATE DATABASE IF NOT EXISTS data_mining_project;",
        "USE data_mining_project;",
        "",
        "CREATE TABLE IF NOT EXISTS students (",
        "    id INT AUTO_INCREMENT PRIMARY KEY,",
        "    math_score FLOAT NOT NULL,",
        "    physics_score FLOAT NOT NULL,",
        "    chemistry_score FLOAT NOT NULL,",
        "    label VARCHAR(20) NOT NULL",
        ");",
        "",
        "INSERT INTO students (math_score, physics_score, chemistry_score, label) VALUES"
    ]
    
    for i, row in df.iterrows():
        sql_line = f"({row['math_score']}, {row['physics_score']}, {row['chemistry_score']}, '{row['label']}')"
        if i < len(df) - 1:
            sql_line += ","
        else:
            sql_line += ";"
        sql_statements.append(sql_line)
    
    # Lưu vào file SQL
    with open("students_new.sql", "w", encoding="utf-8") as f:
        f.write("\n".join(sql_statements))
    
    return df

if __name__ == "__main__":
    df = generate_student_data()
    print("Đã tạo file students_new.csv và students_new.sql với dữ liệu mẫu:")
    print(df.head())
    print("\nPhân phối nhãn:")
    print(df["label"].value_counts())
import pandas as pd
import random

def generate_student_data(n=50):
    data = {
        "math_score": [],
        "physics_score": [],
        "chemistry_score": [],
        "label": []
    }
    labels = ["Giỏi", "Trung bình", "Yếu"]
    
    for _ in range(n):
        # Sinh điểm ngẫu nhiên từ 0.0 đến 10.0, bước 0.5
        math = round(random.uniform(0, 10), 1)
        physics = round(random.uniform(0, 10), 1)
        chemistry = round(random.uniform(0, 10), 1)
        # Gán nhãn dựa trên điểm trung bình
        avg_score = (math + physics + chemistry) / 3
        if avg_score >= 8:
            label = "Giỏi"
        elif avg_score >= 5:
            label = "Trung bình"
        else:
            label = "Yếu"
        data["math_score"].append(math)
        data["physics_score"].append(physics)
        data["chemistry_score"].append(chemistry)
        data["label"].append(label)
    
    df = pd.DataFrame(data)
    df.to_csv("students.csv", index=False)
    return df

if __name__ == "__main__":
    df = generate_student_data()
    print("Đã tạo file students.csv với dữ liệu mẫu:")
    print(df.head())
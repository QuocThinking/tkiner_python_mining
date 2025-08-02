CREATE DATABASE data_mining_project;
USE data_mining_project;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    math_score FLOAT NOT NULL,
    physics_score FLOAT NOT NULL,
    chemistry_score FLOAT NOT NULL,
    label VARCHAR(20) NOT NULL
);

INSERT INTO students (math_score, physics_score, chemistry_score, label) VALUES
(8.5, 7.8, 8.0, 'Giỏi'),
(6.5, 6.0, 5.5, 'Trung bình'),
(4.0, 3.5, 4.5, 'Yếu'),
(9.0, 8.5, 8.8, 'Giỏi'),
(7.0, 6.8, 6.5, 'Trung bình'),
(5.0, 4.8, 5.0, 'Yếu'),
(8.0, 7.5, 7.8, 'Giỏi'),
(6.0, 5.5, 6.0, 'Trung bình');

ALTER TABLE students 
ADD COLUMN decision VARCHAR(20);

-- Cập nhật giá trị cho cột decision
-- Dựa trên logic: Nếu có ít nhất 2 môn >= 7 thì "Đậu", ngược lại "Rớt"
UPDATE students 
SET decision = CASE 
    WHEN (CASE WHEN math_score >= 7 THEN 1 ELSE 0 END +
          CASE WHEN physics_score >= 7 THEN 1 ELSE 0 END +
          CASE WHEN chemistry_score >= 7 THEN 1 ELSE 0 END) >= 2 
    THEN 'Đậu'
    ELSE 'Rớt'
END;

SET SQL_SAFE_UPDATES = 0;
UPDATE students
SET decision = CASE
    WHEN (CASE WHEN math_score >= 7 THEN 1 ELSE 0 END +
          CASE WHEN physics_score >= 7 THEN 1 ELSE 0 END +
          CASE WHEN chemistry_score >= 7 THEN 1 ELSE 0 END) >= 2
    THEN 'Đậu'
    ELSE 'Rớt'
END
WHERE id IS NOT NULL;

-- Kiểm tra kết quả
SELECT * FROM students;
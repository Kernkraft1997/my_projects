# 1. Загружаем библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 2. Загружаем данные
iris = load_iris()
X = iris.data  # признаки
y = iris.target  # целевая переменная (0, 1, 2)

# 3. Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Создаём и обучаем модель (логистическая регрессия)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Предсказываем на тестовых данных
y_pred = model.predict(X_test)

# 6. Оцениваем качество
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Точность (accuracy): {accuracy:.2f}")
print("\n📊 Отчёт по классам:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. Матрица ошибок (визуально)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Предсказано')
plt.ylabel('Реальность')
plt.title('Матрица ошибок')
plt.show()

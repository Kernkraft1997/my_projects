# Это твой первый ML-скрипт
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. СОЗДАЁМ ДАННЫЕ (как будто это реальные замеры)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 4, 5, 7, 8, 10, 11, 13])

# 2. СОЗДАЁМ МОДЕЛЬ И ОБУЧАЕМ
model = LinearRegression()
model.fit(X, y)

# 3. СМОТРИМ, ЧТО МОДЕЛЬ ПОНЯЛА
print(f"Коэффициент (наклон): {model.coef_[0]:.2f}")
print(f"Смещение (точка пересечения): {model.intercept_:.2f}")

# 4. ПРЕДСКАЗЫВАЕМ ДЛЯ НОВОГО ЗНАЧЕНИЯ
new_X = np.array([[9]])
pred = model.predict(new_X)
print(f"Предсказание для X=9: {pred[0]:.2f}")

# 5. РИСУЕМ ГРАФИК
plt.scatter(X, y, color="blue", label="Реальные данные")
plt.plot(X, model.predict(X), color="red", label="Модель")
plt.scatter(new_X, pred, color="green", marker="*", s=200, label="Предсказание")
plt.xlabel("X (признак)")
plt.ylabel("y (цель)")
plt.title("Первая ML-модель")
plt.legend()
plt.grid(True)
plt.show()

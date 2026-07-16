# ==========================================
# 30 Programs in Support Vector Machine (SVM)
# ==========================================

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error


# 1. SVM Classification using Iris Dataset

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("1. Iris SVM Accuracy:",
      accuracy_score(y_test, prediction))


# 2. SVM with RBF Kernel

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("2. RBF Kernel Accuracy:",
      accuracy_score(y_test, prediction))


# 3. SVM with Polynomial Kernel

model = SVC(kernel="poly")

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("3. Polynomial Kernel Accuracy:",
      accuracy_score(y_test, prediction))


# 4. SVM with Sigmoid Kernel

model = SVC(kernel="sigmoid")

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("4. Sigmoid Kernel Accuracy:",
      accuracy_score(y_test, prediction))


# 5. Breast Cancer Classification

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("5. Breast Cancer Accuracy:",
      accuracy_score(y_test, prediction))


# 6. Wine Dataset Classification

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42
)

model = SVC()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("6. Wine Accuracy:",
      accuracy_score(y_test, prediction))


# 7. Support Vector Regression Example

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([10,20,30,40,50])

svr = SVR()

svr.fit(X,y)

print("7. SVR Prediction:",
      svr.predict([[6]]))


# 8. Linear SVM

model = SVC(kernel="linear")

model.fit(iris.data, iris.target)

print("8. Linear SVM completed")


# 9. RBF SVM

model = SVC(kernel="rbf")

model.fit(iris.data, iris.target)

print("9. RBF SVM completed")


# 10. Polynomial SVM

model = SVC(kernel="poly")

model.fit(iris.data, iris.target)

print("10. Polynomial SVM completed")


# 11-30 Placeholder Programs

programs = [
    "11. Hyperparameter tuning using GridSearchCV",
    "12. Cross validation with SVM",
    "13. SVM using StandardScaler",
    "14. SVM using MinMaxScaler",
    "15. Binary classification using SVM",
    "16. Multi-class classification",
    "17. One-vs-Rest SVM",
    "18. One-vs-One SVM",
    "19. SVM probability prediction",
    "20. SVM confusion matrix",
    "21. SVM classification report",
    "22. SVM accuracy comparison",
    "23. SVM kernel comparison",
    "24. SVM parameter C tuning",
    "25. Gamma parameter tuning",
    "26. SVM model saving using pickle",
    "27. SVM model loading",
    "28. SVM prediction on new data",
    "29. SVM pipeline implementation",
    "30. Complete SVM machine learning project"
]


for program in programs:
    print(program)
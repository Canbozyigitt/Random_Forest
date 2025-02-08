import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veri setini yükleme
data = pd.read_csv('train.csv')

# Eksik değerleri doldurma
data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())
data.loc[:, 'Fare'] = data['Fare'].fillna(data['Fare'].median())
data.loc[:, 'Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Gerekli sütunları seçme ve kategorik değişkenleri sayısallaştırma
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Özellik ve hedef değişkenleri tanımlama
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression modeli oluşturma ve eğitme
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Tahmin ve doğruluk skoru
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Logistic Regression Doğruluk Oranı: {accuracy_logreg:.4f}')

# Random Forest modeli oluşturma ve eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Tahmin ve doğruluk skoru
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Doğruluk Oranı: {accuracy_rf:.4f}')

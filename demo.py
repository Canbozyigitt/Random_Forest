import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri setini yükle (banka.csv dosyasının bulunduğu dizinde olduğundan emin olun)
df = pd.read_csv("banka.csv")

# 2. Veri setinin ilk satırlarını gözlemle
print("Veri setinin ilk 5 satırı:")
print(df.head())

# 3. Hedef değişken: 'y' sütunu (örneğin: 'yes' - müşteri abone, 'no' - abone değil)
#    Diğer sütunlar; yaş, iş, medeni durum, eğitim, kredi bilgileri, vs. olabilir.
#    Kategorik sütunları dummy değişkenlere çeviriyoruz.
df_processed = pd.get_dummies(df, drop_first=True)

# 4. Özellikler (X) ve hedef (y) değişkenini belirle
#    Hedef değişkenin dummies dönüşümünden sonra adı 'y_yes' olarak gelecektir.
X = df_processed.drop("y_yes", axis=1)
y = df_processed["y_yes"]

# 5. Veriyi eğitim ve test olarak böl (örnek: %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Lojistik regresyon modelini oluştur ve eğit
model = LogisticRegression(max_iter=1000)  # max_iter parametresi modelin yakınsaması için arttırıldı
model.fit(X_train, y_train)

# 7. Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# 8. Model performansını değerlendir
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy (Doğruluk): {accuracy:.2f}")
print("Confusion Matrix (Hata Matrisi):")
print(cm)
print("Classification Report (Sınıflandırma Raporu):")
print(report)

# 9. Hata matrisini görselleştir (Seaborn ile)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

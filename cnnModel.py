import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

# 1. Verileri Yükleme
def load_binary_data(data_paths, img_size=(400, 400)):
    """
    Verileri yükler ve etiketler.
    Akciğer çökmesi (1) ve diğer sınıflar (0) olarak etiketler.
    Args:
        data_paths (list): Veri yolları listesi.
        img_size (tuple): Görüntülerin boyutlandırılması (varsayılan: 400x400).
    Returns:
        np.array: Görüntüler.
        np.array: Etiketler.
    """
    images = []
    labels = []
    for folder in data_paths:
        for file in os.listdir(folder):
            if file.endswith(".png"):  # Yalnızca .png dosyalarını oku
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri ölçekli oku
                img = cv2.resize(img, img_size)  # Giriş boyutuna yeniden boyutlandır
                images.append(img)
                label = 1 if "cokme" in folder else 0  # Akciğer çökmesi (1) ve diğerleri (0)
                labels.append(label)
    return np.array(images), np.array(labels)

# Veri yolları
train_paths = [
    "C:\\Users\\tenma\\Desktop\\dataSet\\cokme\\cokmesEgtmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\saglikli\\saglikliEgtmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\verem\\veremEgtmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\zature\\zatureEgtmProcessed"
]
test_paths = [
    "C:\\Users\\tenma\\Desktop\\dataSet\\cokme\\cokmeTestProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\saglikli\\saglikliTestProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\verem\\veremTestProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\zature\\zatureTestProcessed"
]
validation_paths = [
    "C:\\Users\\tenma\\Desktop\\dataSet\\cokme\\cokmeDogrlmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\saglikli\\saglikliDogrlmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\verem\\veremDogrlmProcessed",
    "C:\\Users\\tenma\\Desktop\\dataSet\\zature\\zatureDogrlmProcessed"
]

# Eğitim, test ve doğrulama verilerini yükle
X_train, y_train = load_binary_data(train_paths)
X_test, y_test = load_binary_data(test_paths)
X_val, y_val = load_binary_data(validation_paths)

# Kanal Boyutunu Ekleyerek CNN'e Uygun Hale Getirme
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# 2. Model Tanımlama
model = Sequential()

# Evrişimsel Katmanlar
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(400, 400, 1)))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid: İkili sınıflandırma

model.compile(optimizer='adam', 
              loss=BinaryCrossentropy(), 
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_accuracy:.2f}")

# Modeli Kaydetme
model.save("cnn_model0412400.h5")  # Modeli 'cnn_model03' dosyasına kaydeder
print("Model başarıyla kaydedildi!")


plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
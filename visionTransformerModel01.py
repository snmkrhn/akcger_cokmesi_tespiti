import os
import numpy as np
import tensorflow as tf
from transformers import AutoImageProcessor, TFAutoModelForImageClassification
import cv2
import matplotlib.pyplot as plt

# 1. Verileri Yükleme
def load_binary_data(data_paths, img_size=(224, 224)):
    """
    Verileri yükler ve etiketler.
    Args:
        data_paths (list): Veri yolları listesi.
        img_size (tuple): Görüntü boyutlandırma (varsayılan: 224x224).
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
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Renkli olarak oku
                img = cv2.resize(img, img_size)  # Görüntüleri yeniden boyutlandır
                images.append(img)
                label = 1 if "cokme" in folder else 0  # Etiket belirleme
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
X_train, y_train = load_binary_data(train_paths, img_size=(224, 224))
X_test, y_test = load_binary_data(test_paths, img_size=(224, 224))
X_val, y_val = load_binary_data(validation_paths, img_size=(224, 224))

# Normalize etme (0-1 aralığına getirme)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# 2. Vision Transformer Ön İşleme
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

X_train_vit = image_processor(images=list(X_train), return_tensors="tf", do_rescale=False)["pixel_values"]
X_val_vit = image_processor(images=list(X_val), return_tensors="tf", do_rescale=False)["pixel_values"]
X_test_vit = image_processor(images=list(X_test), return_tensors="tf", do_rescale=False)["pixel_values"]

# 3. Vision Transformer Modeli
vit_model = TFAutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=1)

# 4. Modeli Eğitme
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

history = vit_model.fit(
    X_train_vit, y_train,
    validation_data=(X_val_vit, y_val),
    epochs=5,
    batch_size=16
)

# 5. Performansı Değerlendirme
test_loss, test_accuracy = vit_model.evaluate(X_test_vit, y_test)
print(f"Test Doğruluğu: {test_accuracy:.2f}")

# 6. Modeli Kaydetme
vit_model.save("visionTransformer0404255.keras")
print("Model başarıyla kaydedildi!")

# 6. Eğitim ve Doğrulama Kaybı Grafiği
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid()
plt.show()

# Eğitim ve Doğrulama Doğruluğu Grafiği
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()
plt.show()

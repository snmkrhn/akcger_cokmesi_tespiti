import os
import cv2
import numpy as np

# Kaynak ve hedef klasörlerin yolu
source_folder = "C:\\Users\\abdur\\Desktop\\testing\\zature\\zatureTest"
output_folder = "C:\\Users\\abdur\\Desktop\\testing\\zature\\zatureDogrlmProcessed"

# Hedef klasör yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Tüm görüntüleri işle
for filename in os.listdir(source_folder):
    if filename.endswith(".jpeg")or filename.endswith(".png"):
        # Görüntüyü yükle
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        # 1. Boyutlandırma (224x224)
        resized_image = cv2.resize(image, (400, 400))

        # 2. RGB'den Gri Seviyeye Dönüştürme
        if len(resized_image.shape) == 3:  # RGB ise
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:  # Zaten gri seviye ise
            gray_image = resized_image
            

        # 3. Histogram Eşitleme
        equalized_image = cv2.equalizeHist(gray_image)

        # 4. Piksel Normalizasyonu
        normalized_image = equalized_image / 255.0  # [0, 1] aralığına çeker

        # 5. PNG Formatında Kaydetme
        output_path = os.path.join(output_folder, filename.replace(".jpeg", ".png"))
        normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)  # Tekrar [0, 255] aralığına çeker
        cv2.imwrite(output_path, normalized_image_uint8)

        print(f"İşlenen görüntü kaydedildi: {output_path}")

print("Tüm görüntüler işlendi ve kaydedildi.")

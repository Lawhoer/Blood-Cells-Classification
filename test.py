from keras._tf_keras.keras import utils
from keras._tf_keras.keras.models import load_model
import numpy as np
import random
import os

import matplotlib.pyplot as plt



path = os.path.join("data/bloodcells_dataset")

BATCH_SIZE = 100 # model 4 = 16 , diğerleri 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Load the dataset
dataset, test_dataset = utils.image_dataset_from_directory(directory=path,
                                                                 image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                 batch_size=BATCH_SIZE,
                                                                 label_mode="int",
                                                                 validation_split=0.2,
                                                                 subset="both",
                                                                 shuffle=True,
                                                                 seed=5)

class_names = test_dataset.class_names
model = load_model('More_Complex.h5')

# Tüm resimleri ve etiketleri saklamak için boş listeler oluştur
all_images = []
all_labels = []

for images, labels in test_dataset:
    all_images.append(images.numpy())  # Tüm resimleri numpy olarak sakla
    all_labels.append(labels.numpy())  # Tüm etiketleri sakla

all_images = np.concatenate(all_images, axis=0)  # Listeyi birleştir
all_labels = np.concatenate(all_labels, axis=0)  # Listeyi birleştir

# Rastgele 100 örnek seç
random_indices = random.sample(range(len(all_images)), 1000)  # Rastgele 100 indeks seç
random_images = all_images[random_indices]
random_labels = all_labels[random_indices]

# Model tahminleri
predictions = model.predict(random_images)  # Model tahminleri
predicted_labels = predictions.argmax(axis=1)  # En yüksek olasılıklı sınıflar

# Doğruluk hesaplama için başlangıç değerleri
correct_predictions = 0
total_images = 0

# Her bir görüntü için döngü
for i in range(len(random_images)):  # Rastgele seçilen 100 resim için döngü
    true_label = class_names[random_labels[i]]  # Gerçek sınıf
    predicted_label = class_names[predicted_labels[i]]  # Tahmin edilen sınıf

    # Doğruluk kontrolü
    if true_label == predicted_label:
        correct_predictions += 1
    total_images += 1

    # Görselleştirme
    plt.figure(figsize=(6, 3))  # Görsel boyutlarını ayarla

    # Resmi göster
    plt.subplot(1, 2, 1)  # 1 satır, 2 kolon, 1. görsel
    plt.imshow(random_images[i].astype("uint8"))
    plt.axis("off")  # Eksenleri kapat

    # Metin kısmı
    plt.subplot(1, 2, 2)  # 1 satır, 2 kolon, 2. görsel (metin kısmı)
    plt.axis("off")  # Eksenleri kapat
    plt.text(0, 0.6, f"Gerçek Adı:\n{true_label}", fontsize=12, color="black")
    plt.text(0, 0.3, f"Modelin Tahmini:\n{predicted_label}", fontsize=12, color="blue")

    # Görselleştirmeyi göster
    plt.tight_layout()
    plt.show()

# Doğruluk oranını hesapla
accuracy = (correct_predictions / total_images) * 100
print(f"Toplam Resim: {total_images}")
print(f"Doğru Tahminler: {correct_predictions}")
print(f"Doğruluk Oranı: %{accuracy:.2f}")
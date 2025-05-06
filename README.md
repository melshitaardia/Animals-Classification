# Proyek Klasifikasi Gambar: Animal Classification

Proyek ini merupakan bagian dari submission akhir untuk pelatihan klasifikasi gambar menggunakan TensorFlow dan MobileNetV2. Model yang dikembangkan mampu mengklasifikasikan gambar hewan ke dalam beberapa kategori.

## 👩‍💻 Penulis

- **Nama:** Melshita Ardia Kirana  
- **Email:** mc006d5x1408@student.devacademy.id  
- **ID Dicoding:** MC006D5X1408  

## 📁 Struktur Direktori Output

```
Submission
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
```

## 🚀 Fitur

- Pelatihan model klasifikasi gambar menggunakan MobileNetV2
- Augmentasi data menggunakan `ImageDataGenerator`
- Konversi model ke format:
  - TensorFlow SavedModel (untuk cloud/TF serving)
  - TensorFlow Lite (untuk mobile/embedded)
  - TensorFlow.js (untuk web/browser)
- Inference gambar tunggal dengan visualisasi hasil prediksi

## 🧪 Inference Contoh

```python
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

img_path = 'your_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# Tampilkan gambar dan hasil prediksi
plt.imshow(img)
plt.title(f"Prediction: {class_names[predicted_class]} ({confidence:.2f})")
plt.axis('off')
plt.show()
```

## 📦 Instalasi

```bash
pip install -r requirements.txt
```

## 📌 Catatan

- Dataset diunggah melalui Kaggle API
- Model disimpan dalam berbagai format untuk fleksibilitas deployment
- Proyek dikembangkan dan diuji di Google Colab

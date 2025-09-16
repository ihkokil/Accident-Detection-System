# 🚨 Accident Detection System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

A **real-time accident detection system** that analyzes video or CCTV footage to detect road accidents using **Deep Learning**. This project aims to reduce casualties by enabling faster emergency responses.

---

## 📺 Demonstration

![Demo](https://github.com/ihkokil/Accident-Detection-System/raw/main/Demo.gif)

---

## ❓ What is Accident Detection System?

Road accidents are a serious global problem, resulting in significant loss of life every year. This system detects accidents from **CCTV or video footage** in real-time using **machine learning models** trained on accident datasets.  

It’s designed to provide **quick and automated accident detection** which can be integrated with emergency alert systems in the future.

---

## 🛠️ Features

- Real-time accident detection from live camera or video files.
- Deep Learning-based classification using pre-trained models.
- Frame-by-frame prediction with confidence scores.
- Modular design for easy extension (e.g., alarm integration, notification system).

---

## ⚡ Prerequisites

- Python ≥ 3.7  
- Jupyter Notebook installed (`pip install notebook`)  
- Basic knowledge of **Python**, **Machine Learning**, and **Deep Learning** helpful for contributions.  
- A webcam or video file for testing.  

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ihkokil/Accident-Detection-System.git
cd Accident-Detection-System
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download dataset

The dataset is not included in the repository. You must download it from Kaggle:

👉 [Accident Detection from CCTV Footage Dataset](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage/data?select=data)

Place the dataset folders (`train`, `val`, `test`) into the `data/` directory like this:

```
Accident-Detection-System/
├── data/
│   ├── train/
│   │   ├── Accident
│   │   └── Non Accident
│   ├── val/
│   │   ├── Accident
│   │   └── Non Accident
│   └── test/
│       ├── Accident
│       └── Non Accident
```

### 4️⃣ Train the model

Run the Jupyter notebook to train the model and generate weights:

```bash
jupyter notebook accident-classification.ipynb
```

This will create the following files:

* `model.json` – Model architecture
* `model_weights.h5.keras` – Trained weights (**not included in repo, must be generated**)

### 5️⃣ Run the detection

```bash
python main.py
```

* For live camera detection, ensure your webcam is connected.
* For video file detection, you can use `cars.mp4` or `test.mp4`.

---

## 🗂️ Project Structure

```
Accident-Detection-System/
├── data/                         # Dataset (must be downloaded separately)
│   ├── train/Accident
│   ├── train/Non Accident
│   ├── val/Accident
│   ├── val/Non Accident
│   ├── test/Accident
│   └── test/Non Accident
├── accident-classification.ipynb # Notebook to train the model
├── camera.py                     # Captures video & runs detection frame by frame
├── cars.mp4                      # Sample video for testing
├── test.mp4                      # Another sample video
├── Demo.gif                      # Demonstration GIF
├── detection.py                  # Loads the model & performs predictions
├── main.py                       # Entry point for running the system
├── model.json                    # Generated model architecture
├── model_weights.h5.keras        # Generated model weights (not included)
├── model.png                     # Model architecture diagram
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🔮 Future Work

* Integrate **automatic emergency alerts** to police or ambulance services.
* Add **accident severity detection** to prioritize emergency response.
* Deploy as a **cloud-based monitoring system** for smart city traffic surveillance.
* Improve accuracy with **larger datasets** and advanced architectures like YOLOv8 or Faster R-CNN.

---

## 📬 Contact

👨‍💻 Developed by **Md. Iqbal Haider Khan (@ihkokil)**
🌐 [@ihkokil](https://www.linkedin.com/in/ihkokil/)

---

## 📄 License

This project is licensed under the **MIT License**.
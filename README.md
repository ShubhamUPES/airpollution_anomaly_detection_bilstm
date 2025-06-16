### Air Pollution Anomaly Detection using BiLSTM

Detecting unusual patterns in air pollution data using a Bidirectional LSTM (BiLSTM) model. This project applies deep learning to time-series data from environmental sensors to identify spikes or anomalies in pollution levels.

## 🧠 Project Goal
Build a robust sequence-based anomaly detection system that can:
- Learn from temporal pollution trends
- Flag abnormal pollutant behavior (e.g., sudden PM2.5 spikes)
- Assist in early warning or automated monitoring systems

## 🔧 Tech Stack

- Python, PyTorch
- BiLSTM for time-series modeling
- Sliding window preprocessing
- Matplotlib for visualizing anomalies
- Trained and tested on real air quality sensor datasets

## 📊 Key Highlights

- End-to-end pipeline: data cleaning → windowing → BiLSTM training → anomaly detection
- Evaluated using reconstruction error thresholds
- Modular code for easy experimentation and tuning

## 🚀 Run It Yourself

1. Clone the repo:
   git clone https://github.com/ShubhamUPES/airpollution_anomaly_detection_bilstm
   cd airpollution_anomaly_detection_bilstm

2. Install dependencies:
   pip install -r requirements.txt

3. Preprocess and train:
   python src/train.py

4. Detect anomalies:
   python src/evaluate.py
   
📎 About Me

👨‍💻 Shubham Sahu
🔗 [LinkedIn](https://www.linkedin.com/in/shubham-sahu-892751262/)

> This project showcases my interest in AI for environmental monitoring and real-world time-series anomaly detection using deep learning.



import sys
import random
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === Load and preprocess data ===
df = pd.read_csv(r"C:\Users\pc\Documents\university file\class project_ traffic\Metro_Interstate_Traffic_Volume.csv")
df.dropna(inplace=True)

# Binarize target based on median
median_volume = df['traffic_volume'].median()
df['high_traffic'] = (df['traffic_volume'] > median_volume).astype(int)

# Encode categorical data
label_enc = LabelEncoder()
df['weather_main_enc'] = label_enc.fit_transform(df['weather_main'])

# Select features
features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main_enc']
X = df[features]
y = df['high_traffic']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Accuracy (optional to print)
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# === GUI using PyQt6 ===
class TrafficPredictionApp(QWidget):
    def __init__(self, samples, true_labels, predicted_labels):
        super().__init__()
        self.setWindowTitle("Traffic Prediction Results")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        label = QLabel("10 Random Test Predictions:")
        layout.addWidget(label)

        list_widget = QListWidget()
        for i, sample in samples.iterrows():
            idx = list(samples.index).index(i)
            true_label = true_labels[idx]
            pred_label = predicted_labels[idx]
            correct = "✅ Correct" if true_label == pred_label else "❌ Incorrect"
            item_text = f"Sample {i}: Predicted={pred_label}, Actual={true_label} — {correct}"
            list_widget.addItem(QListWidgetItem(item_text))

        layout.addWidget(list_widget)
        self.setLayout(layout)

def main():
    random_indices = random.sample(range(len(X_test)), 10)
    samples = X_test.iloc[random_indices]
    true_labels = y_test.iloc[random_indices].values
    predicted_labels = clf.predict(samples)

    app = QApplication(sys.argv)
    window = TrafficPredictionApp(samples, true_labels, predicted_labels)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

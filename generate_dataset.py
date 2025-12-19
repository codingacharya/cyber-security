import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

attack_types = [
    "Normal", "DoS", "DDoS", "Malware",
    "Phishing", "BruteForce", "InsiderThreat", "ZeroDay"
]

records = []

start_time = datetime.now()

def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"

for attack in attack_types:
    for _ in range(1500):  # 1500 samples per class → balanced dataset
        timestamp = start_time + timedelta(seconds=random.randint(1, 100000))
        
        if attack == "Normal":
            packet_size = np.random.randint(200, 600)
            cpu = np.random.randint(10, 30)
            entropy = np.random.uniform(2.5, 3.5)
            anomaly = np.random.uniform(0.0, 0.2)
        elif attack in ["DoS", "DDoS"]:
            packet_size = np.random.randint(1000, 1600)
            cpu = np.random.randint(70, 95)
            entropy = np.random.uniform(6.5, 8.0)
            anomaly = np.random.uniform(0.8, 1.0)
        elif attack == "Malware":
            packet_size = np.random.randint(700, 1200)
            cpu = np.random.randint(60, 90)
            entropy = np.random.uniform(7.5, 9.5)
            anomaly = np.random.uniform(0.85, 1.0)
        elif attack == "Phishing":
            packet_size = np.random.randint(500, 800)
            cpu = np.random.randint(30, 50)
            entropy = np.random.uniform(6.0, 7.5)
            anomaly = np.random.uniform(0.6, 0.8)
        elif attack == "BruteForce":
            packet_size = np.random.randint(400, 700)
            cpu = np.random.randint(40, 65)
            entropy = np.random.uniform(4.5, 6.0)
            anomaly = np.random.uniform(0.5, 0.7)
        elif attack == "InsiderThreat":
            packet_size = np.random.randint(600, 900)
            cpu = np.random.randint(50, 75)
            entropy = np.random.uniform(6.0, 8.0)
            anomaly = np.random.uniform(0.7, 0.9)
        else:  # ZeroDay
            packet_size = np.random.randint(900, 1500)
            cpu = np.random.randint(80, 99)
            entropy = np.random.uniform(8.5, 10.0)
            anomaly = np.random.uniform(0.95, 1.0)

        records.append([
            timestamp,
            random_ip(),
            random_ip(),
            random.choice(["TCP", "UDP", "HTTP"]),
            packet_size,
            random.choice([22, 80, 443, 3389, 8080]),
            np.random.randint(0, 20),
            np.random.randint(1, 100),
            cpu,
            np.random.randint(20, 1200),
            entropy,
            anomaly,
            attack
        ])

columns = [
    "timestamp", "src_ip", "dst_ip", "protocol",
    "packet_size", "port", "failed_logins",
    "login_frequency", "cpu_usage", "api_calls",
    "file_entropy", "anomaly_score", "attack_type"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("cyber_threat_dataset.csv", index=False)

print("✅ Dataset generated: cyber_threat_dataset.csv")
print("Total records:", len(df))

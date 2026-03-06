import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# PATHS
# ==========================
RAW_ECG_PATH = "raw_ecg"
OUTPUT_PATH = "ecg_images"

NORMAL_DIR = os.path.join(OUTPUT_PATH, "normal")
ARR_DIR = os.path.join(OUTPUT_PATH, "arrhythmia")

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ARR_DIR, exist_ok=True)

# ==========================
# PARAMETERS (FAST MODE)
# ==========================
WINDOW_SIZE = 360 * 5     # 5 seconds
STEP = 360 * 5            # no overlap
MAX_WINDOWS = 200         # LIMIT images per record

NORMAL_BEATS = ["N"]

# ==========================
# PROCESS RECORDS
# ==========================
for record in os.listdir(RAW_ECG_PATH):
    if record.endswith(".dat"):
        record_name = record.replace(".dat", "")
        print(f"🔄 Processing record: {record_name}")

        signal, fields = wfdb.rdsamp(os.path.join(RAW_ECG_PATH, record_name))
        ecg_signal = signal[:, 0]

        annotation = wfdb.rdann(
            os.path.join(RAW_ECG_PATH, record_name), "atr"
        )

        count = 0

        for i in range(0, len(ecg_signal) - WINDOW_SIZE, STEP):

            count += 1
            if count > MAX_WINDOWS:
                break

            window = ecg_signal[i:i + WINDOW_SIZE]

            label = "normal"
            for sample, symbol in zip(annotation.sample, annotation.symbol):
                if i <= sample <= i + WINDOW_SIZE:
                    if symbol not in NORMAL_BEATS:
                        label = "arrhythmia"
                        break

            plt.figure(figsize=(4, 2))
            plt.plot(window, color="black")
            plt.axis("off")

            filename = f"{record_name}_{i}.png"
            save_dir = NORMAL_DIR if label == "normal" else ARR_DIR

            plt.savefig(
                os.path.join(save_dir, filename),
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close()

print("✅ ECG conversion completed successfully")

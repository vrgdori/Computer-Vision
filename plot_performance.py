import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "./results/performance_log.csv"

os.makedirs("./results/graphs", exist_ok=True)

df = pd.read_csv(CSV_PATH)

if df.empty:
    print("No performance data available.")
    exit()

# 1. Feldolgozási idő képkockánként
plt.figure(figsize=(12, 5))

plt.plot(
    df["frame_id"],
    df["processing_time_ms"],
    label="Processing time"
)

plt.axhline(
    df["processing_time_ms"].mean(),
    linestyle="--",
    label="Average"
)

plt.xlabel("Frame ID")
plt.ylabel("Processing time (ms)")
plt.title("Obstacle Detection Processing Time")
plt.legend()
plt.grid(True)

plt.savefig("./results/graphs/processing_time.png")
plt.show()


# 2. Feldolgozási FPS képkockánként
plt.figure(figsize=(12, 5))

plt.plot(
    df["frame_id"],
    df["processing_fps"],
    label="Processing FPS"
)

plt.xlabel("Frame ID")
plt.ylabel("FPS")
plt.title("Obstacle Detection Processing FPS")
plt.legend()
plt.grid(True)

plt.savefig("./results/graphs/processing_fps.png")
plt.show()


plt.figure(figsize=(10, 5))

plt.hist(
    df["processing_time_ms"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Processing time (ms)")
plt.ylabel("Frequency")
plt.title("Processing Time Distribution")
plt.grid(True)

plt.savefig("./results/graphs/processing_distribution.png")
plt.show()


plt.figure(figsize=(12, 5))

plt.plot(
    df["frame_id"],
    df["total_pipeline_time_ms"],
    label="Total pipeline time"
)

plt.xlabel("Frame ID")
plt.ylabel("Time (ms)")
plt.title("Total Pipeline Processing Time")
plt.legend()
plt.grid(True)

plt.savefig("./results/graphs/total_pipeline_time.png")
plt.show()

print("All performance graphs generated successfully!")

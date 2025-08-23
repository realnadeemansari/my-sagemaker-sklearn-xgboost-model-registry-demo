import subprocess
import threading
import time

def start_mlflow_ui():
    subprocess.Popen(
        ["mlflow", "ui", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

# run in background thread
print("Starting MLflow UI")
threading.Thread(target=start_mlflow_ui, daemon=True).start()

# give UI time to spin up
time.sleep(60)

print("Starting pipeline...")

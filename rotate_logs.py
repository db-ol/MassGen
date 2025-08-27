import os
import shutil
from datetime import datetime

def rotate_logs():
    """Rotate log files to prevent them from getting too large."""
    
    log_files = [
        "agent_outputs/benchmark.txt",
        "agent_outputs/benchmark.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            # Check file size (10MB limit)
            if os.path.getsize(log_file) > 10 * 1024 * 1024:
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{log_file}.{timestamp}"
                shutil.move(log_file, backup_file)
                print(f"Rotated {log_file} to {backup_file}")

if __name__ == "__main__":
    rotate_logs()

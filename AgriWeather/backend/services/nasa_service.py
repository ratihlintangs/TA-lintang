import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # backend/
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_PATH = BASE_DIR / "nasa_update.log"


def run_update_script():
    """Run backend/scripts/update_nasa_daily.py using the same Python executable.
    Returns (success_bool, stdout+stderr)
    """
    script = SCRIPTS_DIR / "update_nasa_daily.py"
    if not script.exists():
        return False, f"Script not found: {script}"

    try:
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(SCRIPTS_DIR), timeout=120)
        out = proc.stdout.strip() + "\n" + proc.stderr.strip()
        # append to central log
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(out + "\n")
        return proc.returncode == 0, out
    except Exception as e:
        return False, str(e)


def read_log_tail(lines: int = 200):
    if not LOG_PATH.exists():
        return "No log file"
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return "\n".join(data[-lines:])
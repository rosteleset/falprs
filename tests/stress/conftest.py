import sys
from pathlib import Path

# The runner's modules import each other by name, as when stress.py runs
sys.path.insert(0, str(Path(__file__).resolve().parent))

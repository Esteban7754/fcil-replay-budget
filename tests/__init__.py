"""Los módulos del experimento viven en src/ y se importan por su nombre (import fl_core)."""
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

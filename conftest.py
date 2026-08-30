"""Menaruh root proyek di sys.path agar `import fcv` bekerja dari tests/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

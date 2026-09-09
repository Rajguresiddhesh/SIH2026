"""Make the repo root importable so `import ml` / `import legal_metrology_ml`
work under pytest without an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

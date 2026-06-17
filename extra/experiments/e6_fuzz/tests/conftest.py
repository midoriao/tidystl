"""Put the e6_fuzz dir on sys.path so tests can ``import generator`` etc."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

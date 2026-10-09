import sys
from pathlib import Path

# Ensure marketing-analytics root is at the front of sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

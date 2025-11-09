# Ensure the project root is on sys.path so `import offbeat` works during tests
import os
import sys

# Project root is the parent directory of this file's parent (tests/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import sys
from pathlib import Path

p = Path(__file__).parents[1]
sys.path.append(str(p))
print(p)
print(sys.path)
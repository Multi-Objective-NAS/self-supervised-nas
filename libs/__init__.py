import pathlib
import sys

_paths = [
    'nasbench',
]

for p in _paths:
    if p not in sys.path:
        sys.path.append(str(pathlib.Path(__file__).parent / p))
print(sys.path)

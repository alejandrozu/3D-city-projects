import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Pose and dress the two 1.5 m marble figures (MakeHuman CC0 base mesh) -> data/statues.pkl"""
import sys, pickle; sys.path.insert(0, ROOT + '/src')
from statue import build_pair
pickle.dump(build_pair(), open(ROOT + '/data/statues.pkl', 'wb'))

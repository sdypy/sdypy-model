import numpy as np
import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import sdypy.model

def test_basic():
    assert 1 == 1
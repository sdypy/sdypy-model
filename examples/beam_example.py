import os
import sys
import numpy as np

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from sdypy.model.beam.beam import *


n_elements = 10
beam_length = 500 # mm
beam_ro = 7850 * 1e-12
beam_Young = 180 * 1e3

profile_height = 15
profile_width = 30

length = np.ones(n_elements) * beam_length/(n_elements)
density = np.ones(n_elements) * beam_ro
Young = np.ones(n_elements) * beam_Young
n_nodes = n_elements + 1

beam_obj = Beam(org=None, conec=None, 
        length=length, width=profile_width, 
        height=profile_height, density=density, 
        Young=Young, n_nodes=n_nodes, added_masses=None, mass_locations=None)

nat_freq, vec = beam_obj.solve()

print(nat_freq)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from sdypy.model import Beam


n_elements = 20
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

fig, axs = plt.subplots(3, 2, figsize=(10, 7))
for i in range(6):
    axs.flatten()[i].plot(vec[::2, i])
    axs.flatten()[i].set_title(f'Mode {i+1} ({nat_freq[i]:.2f} Hz)')
plt.tight_layout()
plt.show()
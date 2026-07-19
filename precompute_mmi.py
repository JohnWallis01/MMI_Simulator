import MMI_SParams as mmi
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
centre_wavelength = 1.45
bandwidth = 0.1
subbandwidth = 0.0002
points_per_subband = 3
points_per_band = 30

points = np.linspace(centre_wavelength - bandwidth/2, centre_wavelength + bandwidth/2, points_per_band)
smatricies = []
wavelengths = []
try:
    for point in tqdm(points):
        subband = np.linspace(point - (points_per_subband-1)*(subbandwidth)/2, point + (points_per_subband-1)*(subbandwidth)/2, points_per_subband)
        for subpoint in subband:
            S = mmi.MMI_SMatrix(subpoint, precomputed=False, example=False)
            wavelengths.append(subpoint)
            smatricies.append(S)
finally:
    smatricies = np.array(smatricies)
    wavelengths = np.array(wavelengths)
    # print(wavelengths)
    np.savez("precomputed_mmi.npz", smatricies=smatricies, wavelengths=wavelengths)

"""Example: the optimised 2x2 MMI 3-dB coupler (devices/mmi_2x2).

Loads the device config, gets its S-matrix, and applies it to an input mode to
show the 50:50 split with 90 deg quadrature.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import MMI_SParams as mmi

CONFIG = os.path.join(_ROOT, "devices", "mmi_2x2", "config.json")


def print_mode(v, label):
    for i, a in enumerate(np.asarray(v).flatten()):
        print(f"    port {i+1}: |a|={abs(a):.4f}  power={abs(a)**2:.4f}  "
              f"∠{np.degrees(np.angle(a)):7.2f}°")


if __name__ == "__main__":
    wl = 1.55
    S = mmi.MMI_SMatrix(CONFIG, wl)                    # precomputed; precomputed=False runs live
    print(f"2x2 MMI S-matrix @ {wl*1000:.0f} nm:")
    print(np.round(S, 4))
    print(f"  |S31| (through) = {20*np.log10(abs(S[0,0])):+.2f} dB")
    print(f"  |S41| (cross)   = {20*np.log10(abs(S[1,0])):+.2f} dB")
    print(f"  quadrature ∠S31-∠S41 = {np.degrees(np.angle(S[0,0]/S[1,0])):.2f}°")

    print("\nDrive port 1 with unit amplitude:")
    a_in = np.array([[1], [0]])
    b_out = S @ a_in
    print_mode(b_out, "output")
    print(f"  total output power = {np.sum(np.abs(b_out)**2):.4f}")

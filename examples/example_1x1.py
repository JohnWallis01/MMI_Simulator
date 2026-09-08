"""Example: the 1x1 relay MMI (devices/mmi_1x1).

A centred symmetric MMI with a 0 dB self-image at 1550 nm. Its S21 follows the 2x2
through path (S31) in magnitude (+3 dB), dispersion and phase up to a constant
offset. This prints the 1x1 |S21| next to the 2x2 |S31| to show the 3 dB lift.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import MMI_SParams as mmi

CFG_1x1 = os.path.join(_ROOT, "devices", "mmi_1x1", "config.json")
CFG_2x2 = os.path.join(_ROOT, "devices", "mmi_2x2", "config.json")

if __name__ == "__main__":
    wl = 1.55
    S = mmi.MMI_SMatrix(CFG_1x1, wl)                   # 1x1 -> shape (1,1)
    s21 = S[0, 0]
    print(f"1x1 relay S-matrix @ {wl*1000:.0f} nm: {np.round(S,4)}")
    print(f"  |S21| = {20*np.log10(abs(s21)):+.2f} dB (insertion loss {-20*np.log10(abs(s21)):.2f} dB), "
          f"∠S21 = {np.degrees(np.angle(s21)):.2f}°")

    print("\n1x1 |S21| (0 dB) vs 2x2 |S31| (-3 dB) across the band:")
    for wl in [1.45, 1.50, 1.55, 1.60, 1.65]:
        s21 = mmi.MMI_SMatrix(CFG_1x1, wl)[0, 0]
        s31 = mmi.MMI_SMatrix(CFG_2x2, wl)[0, 0]
        print(f"  {wl*1000:.0f} nm: 1x1 |S21|={20*np.log10(abs(s21)):+.2f} dB   "
              f"2x2 |S31|={20*np.log10(abs(s31)):+.2f} dB   "
              f"(1x1 − [S31+3dB] = {20*np.log10(abs(s21)) - (20*np.log10(abs(s31))+3):+.2f} dB)")

"""Example: query MMI S-parameters from a device config.

The config-driven API (MMI_SParams.MMI_SMatrix / DeviceSMatrix) takes a device
config.json and returns the S-matrix at a wavelength, interpolated from that
device's precomputed band by default (no simulation). Pass precomputed=False to
run the eigenmode engine live instead.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)          # so `import MMI_SParams` resolves from the repo root
import numpy as np
import MMI_SParams as mmi

DEVICES = {
    "2x2 coupler": os.path.join(_ROOT, "devices", "mmi_2x2", "config.json"),
    "1x1 relay":   os.path.join(_ROOT, "devices", "mmi_1x1", "config.json"),
}


def describe(S, label):
    print(f"  {label}: shape {S.shape}")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = S[i, j]
            print(f"    S[{i+1}{j+1}] = {abs(v):.4f} ∠{np.degrees(np.angle(v)):7.2f}°  "
                  f"({20*np.log10(abs(v)):+.2f} dB)")


if __name__ == "__main__":
    wl = 1.55
    for name, cfg_path in DEVICES.items():
        cfg = mmi.load_config(cfg_path)
        print(f"\n{name}  ({cfg['name']}, kind={cfg['kind']}, "
              f"W={cfg['mmi']['width']} um, L={cfg['mmi']['length']} um)")
        S = mmi.MMI_SMatrix(cfg_path, wl)          # precomputed (fast, default)
        describe(S, f"S-matrix @ {wl*1000:.0f} nm")

    # A small wavelength sweep of the 2x2 through/cross split
    print("\n2x2 split vs wavelength:")
    for wl in [1.50, 1.53, 1.55, 1.57, 1.60]:
        S = mmi.MMI_SMatrix(DEVICES["2x2 coupler"], wl)
        thru, cross = abs(S[0, 0]), abs(S[1, 0])
        print(f"  {wl*1000:.0f} nm: |through|={20*np.log10(thru):+.2f} dB  "
              f"|cross|={20*np.log10(cross):+.2f} dB  "
              f"imbalance={20*np.log10(thru/cross):+.2f} dB")

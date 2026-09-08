"""Example: the thermo-optic MZI node (devices/mzi_node) as a tunable coupler.

Pass heater temperatures (K rise above ambient) as
temperatures={"middle": <coupling>, "input": <phase>}. The middle heater sets the
split ratio (3/6/10 dB); the input heater sets the external (inter-port) phase.
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import MMI_SParams as mmi

CFG = os.path.join(_ROOT, "devices", "mzi_node", "config.json")

SETTINGS = {"3 dB": (48.7, 97.6), "6 dB": (32.5, 97.7), "10 dB": (19.9, 97.7)}

if __name__ == "__main__":
    wl = 1.55
    print(f"MZI node (tunable coupler) @ {wl*1000:.0f} nm:")
    for name, (Tmid, Tin) in SETTINGS.items():
        S = mmi.MMI_SMatrix(CFG, wl, temperatures={"middle": Tmid, "input": Tin})
        bar, cross = abs(S[0, 0])**2, abs(S[1, 0])**2
        diff = np.degrees(np.angle(S[0, 0]) - np.angle(S[0, 1]))
        print(f"  {name:5s} (middle={Tmid:.1f} K, input={Tin:.1f} K): "
              f"split {100*bar/(bar+cross):4.1f}:{100*cross/(bar+cross):4.1f}  "
              f"loss {-10*np.log10(bar+cross):.2f} dB  diff-phase {((diff+180)%360)-180:+.0f}°")

    # the full exported two-heater dataset is available for external use:
    w, Tm, Ti, S = mmi.load_mzi_grid(CFG)
    print(f"\nexported grid: S{S.shape}  over {w.min()*1000:.0f}-{w.max()*1000:.0f} nm, "
          f"ΔT {Tm.min():.0f}-{Tm.max():.0f} K (both heaters)")

# MMI Simulator

Meep-based S-parameter simulator for silicon-nitride MMI devices, with a
config-driven API and precomputed broadband bands.

## Layout

```
MMI_SParams.py        core: config API + eigenmode/pulsed engine (+ legacy CW)
devices/
  mmi_2x2/            optimised broadband 3-dB 2x2 coupler (W=18, L=266 um)
    config.json         device definition (geometry, ports, band)
    precomputed.npz     precomputed S-parameters, 1400-1700 nm
  mmi_1x1/            centred 1x1 relay, 0 dB self-image (W=14.6, L=266 um)
    config.json
    precomputed.npz
  mmi_node/ (mzi_node) thermo-optic 2x2 MZI node (two MMIs + s-bends + 2 heaters)
    config.json
    precomputed.npz     full two-heater S-grid (wl x Tmid x Tin, 15-100 K)
    sbend.npz           the s-bend transmission band it is built from
examples/
  example.py          load a config, query the S-matrix
  example_2x2.py      2x2 coupler: split + quadrature
  example_1x1.py      1x1 relay: 0 dB, tracks the 2x2 S31 + 3 dB
  example_mzi.py      MZI node: tunable 3/6/10 dB coupler via heater temperatures
```

## Usage

```python
import MMI_SParams as mmi

# Config-driven (preferred): pass a device config path and a wavelength (um).
S = mmi.MMI_SMatrix("devices/mmi_2x2/config.json", 1.55)   # 2x2 -> (2,2) matrix
S = mmi.MMI_SMatrix("devices/mmi_1x1/config.json", 1.55)   # 1x1 -> (1,1) matrix

# Default: interpolate the device's precomputed band (fast, no simulation).
# precomputed=False runs the eigenmode engine live:
S = mmi.MMI_SMatrix("devices/mmi_2x2/config.json", 1.55, precomputed=False)

# Thermo-optic MZI node (kind="mzi_node"): pass heater temperatures (K rise).
# middle heater -> coupling ratio; input heater -> external (inter-port) phase.
S = mmi.MMI_SMatrix("devices/mzi_node/config.json", 1.55,
                    temperatures={"middle": 48.7, "input": 97.6})   # ~3 dB coupler
wl, Tmid, Tin, Sgrid = mmi.load_mzi_grid("devices/mzi_node/config.json")  # full dataset
```

An `mzi_node` device is assembled on the fly from its base MMI + s-bend bands plus
the analytic heater phases (fast, exact for magnitude/split/loss and relative
phase). Its `precomputed.npz` is a dense exported grid S(wl, Tmid, Tin) over
15-100 K on each heater -- read it with `load_mzi_grid`; don't linearly interpolate
it over wavelength (the propagation phase winds >180 deg per grid step).

A **device config** (`devices/*/config.json`) defines the material indices,
waveguide/taper/MMI dimensions, the port layout (input/output waveguide
positions, which input is driven, and how the S-matrix is assembled), the
eigenmode band settings, and the precomputed-band filename. The same eigenmode
engine (`RunDeviceEigenmodeBand`) runs for any port layout, so new devices are
added by writing a config + generating its `precomputed.npz` (via
`RunDeviceEigenmodeBand`).

The engine launches one broadband pulse of the fundamental mode into the driven
input and reads DFT mode monitors at every output, giving the whole band in one
transient-free run. Materials are refractive indices `n` (SiN 1.974, SiO2 1.4657);
meep uses `epsilon = n^2`.

Running Meep here needs the WSL conda env (`~/miniconda3/envs/mp/bin/python`);
the precomputed path needs only numpy/scipy.

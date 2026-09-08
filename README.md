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
examples/
  example.py          load a config, query the S-matrix
  example_2x2.py      2x2 coupler: split + quadrature
  example_1x1.py      1x1 relay: 0 dB, tracks the 2x2 S31 + 3 dB
ai_workspace/         geometry-optimisation / analysis scripts (see that folder)
dummyMMI/             1x1 relay design + field/comparison scripts
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
```

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

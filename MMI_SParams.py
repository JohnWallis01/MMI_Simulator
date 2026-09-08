import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.interpolate import PchipInterpolator

mp = None  # meep is imported lazily (see _ensure_meep); some callers inject it as MMI_SParams.mp


def _ensure_meep():
    """Import meep into the module global `mp` on first use (keeps the module
    importable, e.g. for the precomputed path, on machines without meep)."""
    global mp
    if mp is None:
        import meep as _meep
        mp = _meep
    return mp

#CONSTANTS
    #a note on units c=1 and a=1um so all meep units are in microns (including time)
# These are refractive INDICES (SiN ~1.97, SiO2 ~1.47). meep materials take
# permittivity, so epsilon = n**2. (Previously the index values were passed
# straight into mp.Medium(epsilon=...), which modelled n=sqrt(1.974)~1.40 etc.)
SIN_index = 1.974
SIO2_index = 1.4657
SIN_epsilon = SIN_index**2
SIO2_epsilon = SIO2_index**2

#device parameters
sm_waveguide_length=50
sm_waveguide_width=1
sm_waveguide_spacing=6    # = mm_waveguide_width/3: inputs at +-W/6 (paired interference)
# Geometry tuned (corrected n=1.974/1.4657 indices) for a broadband 3-dB coupler
# centred at 1550nm. W was narrowed 24->18 (with L and spacing co-scaled) to flatten
# the band: |imbalance|<0.5dB over ~1400-1720nm, <0.7dB loss over 1450-1650nm.
# See ai_workspace/optimize_geometry.py and tune_width.py. (Originals: L360/W24/s8.)
mm_waveguide_length=266
mm_waveguide_width=18
taper_length=50
taper_output_width=5

def GenerateTaperGeometry(taper_length, taper_input_width, taper_output_width, start_position_x, start_position_y, direction, core_eps=None):
    core_eps = SIN_epsilon if core_eps is None else core_eps
    if direction == 'right':
        taper_verticies = [mp.Vector3(start_position_x, start_position_y - taper_input_width/2),
                                    mp.Vector3(start_position_x + taper_length, start_position_y - taper_output_width/2),
                                    mp.Vector3(start_position_x + taper_length, start_position_y + taper_output_width/2),
                                    mp.Vector3(start_position_x, start_position_y + taper_input_width/2)]
    elif direction == 'left':
        taper_verticies = [mp.Vector3(start_position_x, start_position_y - taper_input_width/2),
                                    mp.Vector3(start_position_x - taper_length, start_position_y - taper_output_width/2),
                                    mp.Vector3(start_position_x - taper_length, start_position_y + taper_output_width/2),
                                    mp.Vector3(start_position_x, start_position_y + taper_input_width/2)]
    else:
        raise ValueError("Direction must be 'right' or 'left'")
    taper = mp.Prism(taper_verticies,height=mp.inf,material=mp.Medium(epsilon=core_eps))
    return taper

def GenerateWaveguideGeometry(waveguide_length, waveguide_width, start_position_x, start_position_y, core_eps=None):
    core_eps = SIN_epsilon if core_eps is None else core_eps
    waveguide_verticies = [mp.Vector3(start_position_x, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y + 0.5*waveguide_width),
                                         mp.Vector3(start_position_x, start_position_y + 0.5*waveguide_width)]
    waveguide = mp.Prism(waveguide_verticies,height=mp.inf,material=mp.Medium(epsilon=core_eps))
    return waveguide

def GenerateMMIGeometry(   sm_waveguide_length, sm_waveguide_width,
                                                sm_waveguide_spacing,
                                                mm_waveguide_length, mm_waveguide_width, 
                                                taper_length, taper_output_width):
    """Generates the geometry for a 2x2 MMI coupler with the specified parameters."""
    cell_x_um = mm_waveguide_length + 2*sm_waveguide_length + 2*taper_length
    cell_y_um = np.ceil(1.2 * mm_waveguide_width)

    P1_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, -cell_x_um/2 + sm_waveguide_length, sm_waveguide_spacing/2, 'right')
    P2_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, -cell_x_um/2 + sm_waveguide_length, -sm_waveguide_spacing/2, 'right')
    P3_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, cell_x_um/2 - sm_waveguide_length, sm_waveguide_spacing/2, 'left')
    P4_taper = GenerateTaperGeometry(taper_length, sm_waveguide_width, taper_output_width, cell_x_um/2 - sm_waveguide_length, -sm_waveguide_spacing/2, 'left')

    P1_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, -cell_x_um/2, sm_waveguide_spacing/2)
    P2_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, -cell_x_um/2, -sm_waveguide_spacing/2)
    P3_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, cell_x_um/2 - sm_waveguide_length, sm_waveguide_spacing/2)
    P4_waveguide = GenerateWaveguideGeometry(sm_waveguide_length, sm_waveguide_width, cell_x_um/2 - sm_waveguide_length, -sm_waveguide_spacing/2)
    
    mm_waveguide = GenerateWaveguideGeometry(mm_waveguide_length, mm_waveguide_width, -cell_x_um/2 + sm_waveguide_length + taper_length, 0)
    cell = mp.Vector3(cell_x_um,cell_y_um,0)
    geometry = [P1_taper, P2_taper, P3_taper, P4_taper, P1_waveguide, P2_waveguide, P3_waveguide, P4_waveguide, mm_waveguide]
    # geometry = [P1_waveguide, P2_waveguide, P3_waveguide, P4_waveguide, mm_waveguide]

    return geometry, cell


def RunMMISimulation(wavelength_um, resolution):
    geometry, cell = GenerateMMIGeometry(   sm_waveguide_length=sm_waveguide_length,
                                                    sm_waveguide_width=sm_waveguide_width,
                                                    sm_waveguide_spacing=sm_waveguide_spacing,
                                                    mm_waveguide_length=mm_waveguide_length,
                                                    mm_waveguide_width=mm_waveguide_width,
                                                    taper_length=taper_length,
                                                    taper_output_width=taper_output_width)
    cell_x_um = cell[0]
    cell_y_um = cell[1]
    frequency = 1/wavelength_um
    sources = [
    mp.GaussianBeamSource(
        src=mp.ContinuousSource(frequency),
        center=mp.Vector3(-cell_x_um/2 + 1.5, -sm_waveguide_spacing/2),
        size=mp.Vector3(0,5*sm_waveguide_width),
        beam_kdir=mp.Vector3(1,0),
        beam_w0=0.8*sm_waveguide_width,
        beam_E0=mp.Vector3(0,0,1),
        )
    ]
    pml_layers = [mp.PML(1.0)]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        default_material=mp.Medium(epsilon=SIO2_epsilon))

    sim.run(until=SIN_index*cell_x_um) #light has just reached the end of the device (n*L transit)
    return sim

def VisualiseDevice(sim, LOG=True):
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Ez)
    fig = plt.figure(figsize=(100,30))
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    if LOG:
        plt.imshow(np.log(np.abs(ez_data.transpose())+1), alpha=0.9, )
    else:
        plt.imshow(ez_data.transpose(), cmap="RdBu", alpha=0.9, )

    plt.xticks((np.arange(0, sim.cell_size[0] * sim.resolution, 100)))
    plt.savefig("MMI_Device.png", dpi=300)


def VisualiseTransverseFields(sim):
    pass

def VisualiseLongitudinalFields(sim):
    pass


def GetSParameters(sim, waveguide_spacing):
    resolution = sim.resolution
    ez_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Ez)

    #longitudinal field profiles
    P3_ypos = ez_data.shape[1]//2 - waveguide_spacing//2 * resolution
    P4_ypos = ez_data.shape[1]//2 + waveguide_spacing//2 * resolution
    P1_ypos = ez_data.shape[1]//2 - waveguide_spacing//2 * resolution
    P3_Efield = ez_data[-13*resolution:-3 *resolution , P3_ypos]
    P4_Efield = ez_data[-13*resolution:-3*resolution, P4_ypos]
    P1_Efield = ez_data[3*resolution: 13*resolution, P1_ypos]

    window = np.kaiser(len(P3_Efield),6)
    edge_size =     int(np.ceil(1.5*resolution))
    # window = np.ones_like(output1)
    hP1 = sp.hilbert(P1_Efield*window)
    hP3 = sp.hilbert(P3_Efield*window)
    hP4 = sp.hilbert(P4_Efield*window)

    def HilbertSParam(hx, hy, edge):
        Hxy = hx/hy
        Sxy = np.mean(Hxy[edge:-edge])
        return Sxy
    
    S31 = HilbertSParam(hP3, hP1, edge_size)
    S41 = HilbertSParam(hP4, hP1, edge_size)
    return S31, S41

def PrintSParameters(Sxy, name="Sxy"):
    amplitude_dB = 20*np.log10(np.abs(Sxy))
    phase_deg = np.degrees(np.angle(Sxy))
    print(f"{name}: {Sxy:.4f}, Amplitude: {amplitude_dB:.2f} dB, ∠: {phase_deg:.2f} degrees")
    
def main(wavelength_um=1.55, resolution=12, visualise=False):
    #global import meep
    global mp
    import meep as mp
    sim = RunMMISimulation(wavelength_um, resolution)
    if visualise:
        VisualiseDevice(sim)
        print("Wavelength: {:.4f} nm".format(wavelength_um * 1000))
        PrintSParameters(S31, name="S31")
        PrintSParameters(S41, name="S41")
    S31, S41 = GetSParameters(sim, sm_waveguide_spacing)
    return np.array([[S31, S41], [S41, S31]])


#precomputed S-parameter data (covers ~1.4-1.6um) sits in these files alongside this script
_PRECOMPUTED_SPARAM_FILES = ["14-15um.npz", "15-16um.npz"]
_precomputed_fit_cache = None

def _LoadPrecomputedSMatrixFit():
    """Loads and fits the precomputed S-matrix data to build interpolators vs wavelength_um."""
    global _precomputed_fit_cache
    if _precomputed_fit_cache is not None:
        return _precomputed_fit_cache

    base_dir = os.path.dirname(os.path.abspath(__file__))
    wavelengths = []
    smatricies = []
    for filename in _PRECOMPUTED_SPARAM_FILES:
        data = np.load(os.path.join(base_dir, filename))
        wavelengths.append(data['wavelengths'])
        smatricies.append(data['smatricies'])
    wavelengths = np.concatenate(wavelengths)
    smatricies = np.concatenate(smatricies)

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    smatricies = smatricies[order]

    # the raw data is sampled in tight triplets (~0.0002um apart: a band-centre point
    # plus two flanking points, meant for estimating group delay via finite difference)
    # separated by much larger gaps (~0.003um) between triplets/bands. The two files
    # also overlap by one triplet at their shared boundary. Group consecutive samples
    # into clusters wherever the gap is small, then take each cluster's centre sample
    # (not its complex mean!) as the representative point for that band: the physical
    # propagation phase over the 360um multimode section winds fast enough that phase
    # can swing tens of degrees across a single triplet, so averaging the complex
    # values within a cluster partially cancels them and biases |S| low by several
    # tenths of a dB (verified against the raw per-point magnitude). The centre sample
    # is an actual measurement, so it carries no such bias.
    gaps = np.diff(wavelengths)
    cluster_breaks = np.nonzero(gaps > 0.001)[0] + 1
    cluster_ids = np.zeros(len(wavelengths), dtype=int)
    cluster_ids[cluster_breaks] = 1
    cluster_ids = np.cumsum(cluster_ids)
    n_clusters = cluster_ids[-1] + 1

    cluster_wavelengths = np.zeros(n_clusters)
    cluster_smatricies = np.zeros((n_clusters, 2, 2), dtype=complex)
    for c in range(n_clusters):
        indices = np.nonzero(cluster_ids == c)[0]
        centre_index = indices[len(indices)//2]
        cluster_wavelengths[c] = wavelengths[centre_index]
        cluster_smatricies[c] = smatricies[centre_index]

    # Fit magnitude (dB) and unwrapped phase (radians) separately rather than real/imag:
    # magnitude(dB) is the slowly-varying, visually-checkable envelope, while phase is
    # sensitive to noise wherever |S| dips low - separating them lets each be smoothed
    # on its own terms instead of a real/imag fit distorting one to accommodate the other.
    # Only (0,0)=S31 and (1,0)=S41 are independent; (0,1)/(1,1) mirror them (see the
    # symmetry note above main()'s return in this file).
    fits = {}
    for i, j in [(0, 0), (1, 0)]:
        cluster_vals = cluster_smatricies[:, i, j]
        mag_db = 20*np.log10(np.abs(cluster_vals))
        phase_rad = np.unwrap(np.angle(cluster_vals))
        phase_rad = _DespikePhase(phase_rad)
        mag_fit = PchipInterpolator(cluster_wavelengths, mag_db)
        phase_fit = PchipInterpolator(cluster_wavelengths, phase_rad)
        fits[(i, j)] = (mag_fit, phase_fit)
    fits[(0, 1)] = fits[(1, 0)]
    fits[(1, 1)] = fits[(0, 0)]

    _precomputed_fit_cache = (cluster_wavelengths, fits)
    return _precomputed_fit_cache


def _DespikePhase(phase_rad, window=5, n_sigma=3.0):
    """Pulls outlier samples back toward the local trend (Hampel filter).

    A handful of the precomputed points carry simulation noise that shows up as an
    isolated jump against the otherwise-smooth unwrapped phase trend. Interpolating
    through such a point exactly (as PchipInterpolator does) reproduces the jump as a
    spurious spike in the fitted curve. This detects samples that sit more than
    n_sigma robust-standard-deviations from their local (windowed) median and snaps
    them to that median before fitting.
    """
    n = len(phase_rad)
    corrected = phase_rad.copy()
    half_window = window // 2
    for k in range(n):
        lo, hi = max(0, k - half_window), min(n, k + half_window + 1)
        neighbours = np.concatenate([phase_rad[lo:k], phase_rad[k+1:hi]])
        median = np.median(neighbours)
        mad = np.median(np.abs(neighbours - median)) * 1.4826  # ~= robust std for gaussian noise
        if mad < 1e-9:
            continue
        if np.abs(phase_rad[k] - median) > n_sigma * mad:
            corrected[k] = median
    return corrected


def FittedSMatrix(wavelength_um):
    """Returns the S-matrix at wavelength_um, fitted from the precomputed data (~1.4-1.6um)."""
    fit_wavelengths, fits = _LoadPrecomputedSMatrixFit()
    wl_min, wl_max = fit_wavelengths[0], fit_wavelengths[-1]
    if not (wl_min <= wavelength_um <= wl_max):
        raise ValueError(
            f"No precomputed S-parameter data at {wavelength_um:.4f}um; "
            f"precomputed data only covers {wl_min:.4f}-{wl_max:.4f}um."
        )
    S = np.zeros((2, 2), dtype=complex)
    for (i, j), (mag_fit, phase_fit) in fits.items():
        S[i, j] = 10**(mag_fit(wavelength_um)/20) * np.exp(1j*phase_fit(wavelength_um))
    return S


# ---------------------------------------------------------------------------
# Eigenmode / pulsed engine (the default). One broadband Gaussian pulse of the
# fundamental waveguide mode is launched into the input port; DFT mode monitors
# in the input and both output waveguides record the S-parameters across the
# whole band in a single run, and the fields are run until they decay
# (stop_when_fields_decayed). Unlike the CW-snapshot method (main() above), this
# is transient-free, so it does not carry the settling ripple, and it covers the
# entire band at once. S31 = b3(+)/a1(+), S41 = b4(+)/a1(+): forward mode
# coefficient at each output waveguide over the forward (incident) coefficient at
# the input waveguide (the ratio cancels the source spectrum).
# ---------------------------------------------------------------------------

_EIGENMODE_BAND_UM = (1.4, 1.7)        # default band for a live eigenmode run
_EIGENMODE_MON_H = 6.0                  # monitor / source transverse height (um)


def RunEigenmodeBand(nfreq=120, resolution=12, decay_by=1e-3,
                     band_um=_EIGENMODE_BAND_UM):
    """Broadband pulsed/eigenmode S-parameters across band_um in ONE simulation.

    Returns (wavelengths_um ascending, S31 array, S41 array).
    """
    global mp
    import meep as mp

    geometry, cell = GenerateMMIGeometry(sm_waveguide_length=sm_waveguide_length,
                                         sm_waveguide_width=sm_waveguide_width,
                                         sm_waveguide_spacing=sm_waveguide_spacing,
                                         mm_waveguide_length=mm_waveguide_length,
                                         mm_waveguide_width=mm_waveguide_width,
                                         taper_length=taper_length,
                                         taper_output_width=taper_output_width)
    cell_x = cell[0]
    y_port = sm_waveguide_spacing / 2.0          # bottom = -y_port (input & through), top = +y_port (cross)
    x_src = -cell_x/2 + 8                          # source, inside the input waveguide
    x_in = -cell_x/2 + 20                          # input monitor, downstream of the source
    x_out = cell_x/2 - 20                          # output monitors, inside the output waveguides

    fmin, fmax = 1.0/band_um[1], 1.0/band_um[0]
    fcen = 0.5*(fmin + fmax)
    df = 1.25*(fmax - fmin)                        # slightly wider than the band for flat coverage

    sources = [mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_src, -y_port),
        size=mp.Vector3(0, _EIGENMODE_MON_H),
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_match_freq=True)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0)],
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        default_material=mp.Medium(epsilon=SIO2_epsilon))

    freqs = np.linspace(fmin, fmax, nfreq)
    mon_in = sim.add_mode_monitor(freqs, mp.FluxRegion(center=mp.Vector3(x_in, -y_port),
                                                       size=mp.Vector3(0, _EIGENMODE_MON_H)))
    mon_thru = sim.add_mode_monitor(freqs, mp.FluxRegion(center=mp.Vector3(x_out, -y_port),
                                                         size=mp.Vector3(0, _EIGENMODE_MON_H)))
    mon_cross = sim.add_mode_monitor(freqs, mp.FluxRegion(center=mp.Vector3(x_out, +y_port),
                                                          size=mp.Vector3(0, _EIGENMODE_MON_H)))

    # The device is ~560 um long: the pulse needs ~700 t.u. to reach the outputs,
    # well after the source ends. Run a fixed traversal interval first so the
    # pulse actually arrives, THEN watch the output field decay to steady state
    # (calling stop_when_fields_decayed before the pulse arrives stops it at ~0).
    sim.run(until_after_sources=1000)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(x_out, -y_port), decay_by))

    def fwd(mon):
        r = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z)
        return r.alpha[0, :, 0]                    # band 0, all freqs, +x direction

    a1, b3, b4 = fwd(mon_in), fwd(mon_thru), fwd(mon_cross)
    S31 = b3 / a1
    S41 = b4 / a1
    wl = 1.0 / freqs
    order = np.argsort(wl)
    return wl[order], S31[order], S41[order]


# precomputed eigenmode band (covers 1.4-1.6um) sits alongside this script
_PRECOMPUTED_EIGENMODE_FILE = "precomputed_eigenmode.npz"
_eigenmode_fit_cache = None
_live_eigenmode_cache = None


def _BuildSMatrixFit(wavelengths, smatricies):
    """PCHIP interpolators (magnitude dB + unwrapped phase) for S31 and S41.

    The eigenmode data is smooth (transient-free), so no despiking is needed --
    contrast _LoadPrecomputedSMatrixFit for the noisier CW data.
    """
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    smatricies = smatricies[order]
    fits = {}
    for i, j in [(0, 0), (1, 0)]:
        vals = smatricies[:, i, j]
        mag_db = 20*np.log10(np.abs(vals))
        phase_rad = np.unwrap(np.angle(vals))
        fits[(i, j)] = (PchipInterpolator(wavelengths, mag_db),
                        PchipInterpolator(wavelengths, phase_rad))
    fits[(0, 1)] = fits[(1, 0)]
    fits[(1, 1)] = fits[(0, 0)]
    return wavelengths, fits


def _EvalSMatrixFit(wavelength_um, fit_wavelengths, fits, what):
    wl_min, wl_max = fit_wavelengths[0], fit_wavelengths[-1]
    if not (wl_min <= wavelength_um <= wl_max):
        raise ValueError(
            f"No {what} at {wavelength_um:.4f}um; it only covers "
            f"{wl_min:.4f}-{wl_max:.4f}um.")
    S = np.zeros((2, 2), dtype=complex)
    for (i, j), (mag_fit, phase_fit) in fits.items():
        S[i, j] = 10**(mag_fit(wavelength_um)/20) * np.exp(1j*phase_fit(wavelength_um))
    return S


def _LoadPrecomputedEigenmodeFit():
    global _eigenmode_fit_cache
    if _eigenmode_fit_cache is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data = np.load(os.path.join(base_dir, _PRECOMPUTED_EIGENMODE_FILE))
        _eigenmode_fit_cache = _BuildSMatrixFit(data['wavelengths'], data['smatricies'])
    return _eigenmode_fit_cache


def EigenmodeSMatrix(wavelength_um):
    """S-matrix at wavelength_um, interpolated from the precomputed eigenmode band."""
    fit_wavelengths, fits = _LoadPrecomputedEigenmodeFit()
    return _EvalSMatrixFit(wavelength_um, fit_wavelengths, fits, "precomputed eigenmode data")


def LiveEigenmodeSMatrix(wavelength_um, resolution=12):
    """S-matrix at wavelength_um from a live eigenmode run. The whole band is
    computed once and cached in-process, so repeated calls are cheap."""
    global _live_eigenmode_cache
    cache = _live_eigenmode_cache
    if cache is None or not (cache[0][0] <= wavelength_um <= cache[0][-1]):
        wl, S31, S41 = RunEigenmodeBand(resolution=resolution)
        smat = np.zeros((len(wl), 2, 2), dtype=complex)
        smat[:, 0, 0] = S31
        smat[:, 1, 0] = S41
        smat[:, 0, 1] = S41
        smat[:, 1, 1] = S31
        _live_eigenmode_cache = cache = (wl, *_BuildSMatrixFit(wl, smat)[1:], smat)
    wl, fits = cache[0], cache[1]
    return _EvalSMatrixFit(wavelength_um, wl, fits, "live eigenmode band")


# ===========================================================================
# CONFIG-DRIVEN DEVICE API
# ---------------------------------------------------------------------------
# A device is described by a JSON config (see devices/*/config.json): material
# indices, waveguide/taper/MMI dimensions, and the port layout (input & output
# waveguide y-positions, which input is driven, and how the returned S-matrix is
# assembled). The same eigenmode/pulsed engine as above runs for any port layout,
# so both the 2x2 coupler and the 1x1 relay use one code path. Each device folder
# also holds a precomputed.npz band, interpolated by default (no simulation).
#
# Config schema:
#   kind:      "2x2" or "1x1" (how MMI_SMatrix assembles the returned matrix)
#   material:  {core_index, clad_index}
#   waveguide: {sm_length, sm_width, taper_length, taper_output_width}
#   mmi:       {length, width}
#   ports:     inputs/outputs = [{name, y}, ...]; drive = <input name>;
#              for 2x2 also through/cross = <output names>
#   eigenmode: {band_um, nfreq, resolution, decay_by, monitor_height,
#               source_offset, monitor_offset}
#   precomputed: filename (relative to the config) of the precomputed band
#     (keys: wavelengths (N,), S (N, n_outputs) complex transmission per output)
# ===========================================================================

_device_fit_cache = {}
_device_live_cache = {}


def load_config(path):
    """Load a device config JSON; records its directory for resolving files."""
    with open(path) as f:
        cfg = json.load(f)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def _cfg(config):
    return load_config(config) if isinstance(config, str) else config


def BuildDeviceGeometry(config):
    """Geometry for any port layout: input waveguides+tapers on the left, the MMI
    slab in the centre, output tapers+waveguides on the right."""
    _ensure_meep()
    wg, mm = config["waveguide"], config["mmi"]
    core_eps = config["material"]["core_index"]**2
    sm_len, sm_w = wg["sm_length"], wg["sm_width"]
    tp_len, tp_w = wg["taper_length"], wg["taper_output_width"]
    L, W = mm["length"], mm["width"]
    cell_x = L + 2*sm_len + 2*tp_len
    cell_y = float(np.ceil(1.2 * W))
    g = []
    for p in config["ports"]["inputs"]:
        g.append(GenerateWaveguideGeometry(sm_len, sm_w, -cell_x/2, p["y"], core_eps=core_eps))
        g.append(GenerateTaperGeometry(tp_len, sm_w, tp_w, -cell_x/2 + sm_len, p["y"], 'right', core_eps=core_eps))
    for p in config["ports"]["outputs"]:
        g.append(GenerateTaperGeometry(tp_len, sm_w, tp_w, cell_x/2 - sm_len, p["y"], 'left', core_eps=core_eps))
        g.append(GenerateWaveguideGeometry(sm_len, sm_w, cell_x/2 - sm_len, p["y"], core_eps=core_eps))
    g.append(GenerateWaveguideGeometry(L, W, -cell_x/2 + sm_len + tp_len, 0, core_eps=core_eps))
    return g, mp.Vector3(cell_x, cell_y, 0)


def RunDeviceEigenmodeBand(config, nfreq=None, resolution=None, decay_by=None, band_um=None):
    """Broadband eigenmode transmission of a config device: drive the config's
    input, monitor every output. Returns (wl ascending, S) with S shape
    (n_outputs, N) = forward output coeff / forward input coeff per output."""
    _ensure_meep()
    eig = config["eigenmode"]
    nfreq = int(nfreq or eig["nfreq"])
    resolution = int(resolution or eig["resolution"])
    decay_by = float(decay_by or eig["decay_by"])
    band_um = tuple(band_um or eig["band_um"])
    monh = eig.get("monitor_height", 6.0)
    soff = eig.get("source_offset", 8.0)
    moff = eig.get("monitor_offset", 20.0)

    geometry, cell = BuildDeviceGeometry(config)
    cell_x = cell[0]
    ports = config["ports"]
    drive = next(p for p in ports["inputs"] if p["name"] == ports["drive"])
    x_src, x_in, x_out = -cell_x/2 + soff, -cell_x/2 + moff, cell_x/2 - moff
    fmin, fmax = 1.0/band_um[1], 1.0/band_um[0]
    fcen, df = 0.5*(fmin + fmax), 1.25*(fmax - fmin)

    sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=df),
                                  center=mp.Vector3(x_src, drive["y"]), size=mp.Vector3(0, monh),
                                  eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)], geometry=geometry,
                        sources=sources, resolution=resolution,
                        default_material=mp.Medium(epsilon=config["material"]["clad_index"]**2))
    freqs = np.linspace(fmin, fmax, nfreq)
    mon_in = sim.add_mode_monitor(freqs, mp.FluxRegion(center=mp.Vector3(x_in, drive["y"]),
                                                       size=mp.Vector3(0, monh)))
    mon_out = [sim.add_mode_monitor(freqs, mp.FluxRegion(center=mp.Vector3(x_out, p["y"]),
                                                         size=mp.Vector3(0, monh)))
               for p in ports["outputs"]]
    sim.run(until_after_sources=1000)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(x_out, ports["outputs"][0]["y"]), decay_by))

    def fwd(mon):
        return sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z).alpha[0, :, 0]
    a1 = fwd(mon_in)
    S = np.array([fwd(m) / a1 for m in mon_out])   # (n_out, N)
    wl = 1.0 / freqs
    order = np.argsort(wl)
    return wl[order], S[:, order]


def AssembleSMatrix(config, S_row):
    """Assemble the device S-matrix from the per-output transmission row."""
    kind = config["kind"]
    out_names = [p["name"] for p in config["ports"]["outputs"]]
    if kind == "1x1":
        return np.array([[S_row[0]]], dtype=complex)
    if kind == "2x2":
        thru = S_row[out_names.index(config["ports"]["through"])]
        cross = S_row[out_names.index(config["ports"]["cross"])]
        return np.array([[thru, cross], [cross, thru]], dtype=complex)
    raise ValueError(f"unknown device kind {kind!r}")


def _fit_columns(wl, S):
    """PCHIP (magnitude dB + unwrapped phase) per output column of S (N, n_out)."""
    order = np.argsort(wl); wl = wl[order]; S = S[order]
    fits = []
    for k in range(S.shape[1]):
        vals = S[:, k]
        fits.append((PchipInterpolator(wl, 20*np.log10(np.abs(vals))),
                     PchipInterpolator(wl, np.unwrap(np.angle(vals)))))
    return wl, fits


def _eval_columns(wl, fits, wavelength_um, what):
    if not (wl[0] <= wavelength_um <= wl[-1]):
        raise ValueError(f"No {what} at {wavelength_um:.4f}um; it only covers "
                         f"{wl[0]:.4f}-{wl[-1]:.4f}um.")
    return np.array([10**(mf(wavelength_um)/20) * np.exp(1j*pf(wavelength_um))
                     for mf, pf in fits])


def _load_device_fit(config):
    key = config["_dir"] + "/" + config["precomputed"]
    if key not in _device_fit_cache:
        data = np.load(os.path.join(config["_dir"], config["precomputed"]))
        _device_fit_cache[key] = _fit_columns(data["wavelengths"], data["S"])
    return _device_fit_cache[key]


def PrecomputedDeviceSMatrix(config, wavelength_um):
    wl, fits = _load_device_fit(config)
    S_row = _eval_columns(wl, fits, wavelength_um, f"precomputed data for {config['name']}")
    return AssembleSMatrix(config, S_row)


def LiveDeviceSMatrix(config, wavelength_um, **overrides):
    key = config["_dir"]
    cache = _device_live_cache.get(key)
    if cache is None or not (cache[0][0] <= wavelength_um <= cache[0][-1]):
        wl, S = RunDeviceEigenmodeBand(config, **overrides)   # S (n_out, N)
        _device_live_cache[key] = cache = _fit_columns(wl, S.T)
    wl, fits = cache
    S_row = _eval_columns(wl, fits, wavelength_um, f"live band for {config['name']}")
    return AssembleSMatrix(config, S_row)


def DeviceSMatrix(config, wavelength_um, precomputed=True):
    """S-matrix of a config-described device at wavelength_um (um). config may be a
    path to a config.json or a loaded dict. precomputed=True interpolates the
    device's precomputed.npz; precomputed=False runs the eigenmode engine live."""
    config = _cfg(config)
    return (PrecomputedDeviceSMatrix(config, wavelength_um) if precomputed
            else LiveDeviceSMatrix(config, wavelength_um))


def MMI_SMatrix(config_or_wavelength, wavelength_um=None, example=False,
                precomputed=True, engine="eigenmode"):
    """S-matrix of an MMI device.

    Config-driven form (preferred): MMI_SMatrix(config, wavelength_um) where config
    is a path to a device config.json (or a loaded dict). precomputed=True (default)
    interpolates the device's precomputed band; precomputed=False runs it live.

    Legacy form: MMI_SMatrix(wavelength_um, ...) uses the module-global default 2x2
    device (kept for older scripts). engine="cw" selects the legacy Hilbert method.
    example=True returns the ideal coupler matrix for the device kind.
    """
    if isinstance(config_or_wavelength, (str, dict)):
        config = _cfg(config_or_wavelength)
        if example:
            return (np.array([[1.0]]) if config["kind"] == "1x1"
                    else np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))
        return DeviceSMatrix(config, wavelength_um, precomputed=precomputed)

    # ---- legacy float-wavelength form (default 2x2 module-global device) ----
    wl = config_or_wavelength
    if example:
        return np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    if precomputed:
        return FittedSMatrix(wl) if engine == "cw" else EigenmodeSMatrix(wl)
    return main(wavelength_um=wl, resolution=12, visualise=False) if engine == "cw" else LiveEigenmodeSMatrix(wl)


if __name__ == "__main__":
    wavelength_um = 1.55 - 6*0.0001761
    resolution = 12
    main(wavelength_um=wavelength_um, resolution=resolution, visualise=True)
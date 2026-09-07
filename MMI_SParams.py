import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.interpolate import PchipInterpolator

#CONSTANTS
    #a note on units c=1 and a=1um so all meep units are in microns (including time)
SIN_epsilon = 1.974
SIO2_epsilon = 1.4657

#device parameters
sm_waveguide_length=50
sm_waveguide_width=1
sm_waveguide_spacing=8
mm_waveguide_length=360
mm_waveguide_width=24
taper_length=50
taper_output_width=5

def GenerateTaperGeometry(taper_length, taper_input_width, taper_output_width, start_position_x, start_position_y, direction):
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
    taper = mp.Prism(taper_verticies,height=mp.inf,material=mp.Medium(epsilon=SIN_epsilon))
    return taper

def GenerateWaveguideGeometry(waveguide_length, waveguide_width, start_position_x, start_position_y):
    waveguide_verticies = [mp.Vector3(start_position_x, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y - 0.5*waveguide_width),
                                         mp.Vector3(start_position_x + waveguide_length, start_position_y + 0.5*waveguide_width),
                                         mp.Vector3(start_position_x, start_position_y + 0.5*waveguide_width)]
    waveguide = mp.Prism(waveguide_verticies,height=mp.inf,material=mp.Medium(epsilon=SIN_epsilon))
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

    sim.run(until=SIN_epsilon*cell_x_um) #light has just reached the end of the device
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

_EIGENMODE_BAND_UM = (1.4, 1.6)        # default band for a live eigenmode run
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


def MMI_SMatrix(wavelength_um, example=False, precomputed=True, engine="eigenmode"):
    """S-matrix of the 2x2 MMI at wavelength_um.

    engine="eigenmode" (default): pulsed/eigenmode extraction (transient-free).
    engine="cw": the legacy ContinuousSource + Hilbert-snapshot method.
    precomputed=True (default): interpolate saved band data (fast, no simulation).
    precomputed=False: run the chosen engine live.
    example=True: return the ideal 1/sqrt(2)*[[1,1j],[1j,1]] matrix.
    """
    if example:
        return np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    if precomputed:
        if engine == "cw":
            return FittedSMatrix(wavelength_um)
        return EigenmodeSMatrix(wavelength_um)
    if engine == "cw":
        return main(wavelength_um=wavelength_um, resolution=12, visualise=False)
    return LiveEigenmodeSMatrix(wavelength_um)


if __name__ == "__main__":
    wavelength_um = 1.55 - 6*0.0001761
    resolution = 12
    main(wavelength_um=wavelength_um, resolution=resolution, visualise=True)
import cv2
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sys import argv


# FUNCTIONS

"""Read image as grayscale"""
def imread_gs(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


"""Gaussian normalization of spectrum"""
def gnorm(spec: np.ndarray) -> np.ndarray:
    return (spec - np.mean(spec))/np.std(spec)


"""Parse a line from the reference spectrum"""
def readline_ref(line: str) -> tuple[float, float]:
    x, y = line.split('\t')
    return (float(x), float(y))


# CONSTANTS

WIDTH = 15
SIGNAL_MIN_BRIGHTNESS = 50
SIGNAL_GRADIENT_CUTOFF = 1
SMOOTHING_FACTOR = 2
PROMINENCE_MAXIMA = 10
PROMINENCE_MINIMA = 10
REFERENCE_PATH = 'data/sapphire_ref_cm++.txt'
PROMINENCE_REF = .1
CUTOFF_REF = 10
BADNESS_THRESHOLD = 5e-4
DEBUG_MODE = False

# MAIN
if __name__ == '__main__':
    path = argv[1]
    try:
        f0 = imread_gs(path)
    except:
        print('File could not be found')
        exit()

    # FIND AREA OF INTEREST

    ## first iteration center line
    f0_max1 = np.where(f0.sum(axis=1) == f0.sum(axis=1).max())[0][0]

    center_line = f0[f0_max1, :]
    ## find laser, use it to find signal start after filter
    laser_loc = np.where(center_line == max(center_line))[0][0]
    laser_left = laser_loc < len(center_line)/2

    ## flip image such that the laser is on the right-hand side of the image
    if laser_left:
        f0 = np.flip(f0, axis=1)
        center_line = f0[f0_max1, ::-1]
        laser_loc = np.where(center_line == max(center_line))[0][0]

    ## use gradient to find the laser onset
    center_gradient = np.gradient(gaussian_filter(center_line, 5))
    laser_onset_loc = np.where(center_gradient[:laser_loc] == 0)[0][-1]

    ## use gradient to the left of the laser to find where the signal starts
    gradient_condition = center_gradient[:laser_onset_loc] > -SIGNAL_GRADIENT_CUTOFF
    brightness_condition = center_line[:laser_onset_loc] > SIGNAL_MIN_BRIGHTNESS
    signal_loc = np.where(gradient_condition & brightness_condition)[0][-1]

    ## cut off the part of the image without the signal
    f0 = f0[:, :signal_loc]

    ## second iteration max
    f0_max = np.where(f0.sum(axis=1) == f0.sum(axis=1).max())[0][0]

    ## center cut (flip such that signal starts on the right) and spectrum
    f0 = np.flip(f0[f0_max-WIDTH:f0_max+WIDTH, :], axis=1)
    f0_s = f0.sum(axis=0, dtype=np.int64)


    # FIND PEAKS (to calibrate) AND MINIMA (to fit background)
    f0_smoothed = gaussian_filter(f0_s, SMOOTHING_FACTOR)
    f0_peaks  = find_peaks( f0_smoothed, prominence=PROMINENCE_MAXIMA)[0]
    f0_minima = find_peaks(-f0_smoothed, prominence=PROMINENCE_MINIMA)[0]
    if DEBUG_MODE:
        with open('/tmp/peaks', 'w') as f:
            f.write(','.join([str(s) for s in f0_peaks]))


    # GET REFERENCE SPECTRUM OF CORUNDUM (SAPPHIRE GLASS) AND PEAKS
    with open(REFERENCE_PATH, 'r') as f:
        sap_ref = [ readline_ref(l) for l in f.readlines()[1+CUTOFF_REF:] ]

    ref_wvns, ref_vals = np.array(sap_ref).transpose()
    ref_peaks = [ ref_wvns[p] for p in find_peaks(ref_vals, prominence=PROMINENCE_REF)[0] ]

    # CALIBRATE BY FITTING THE FIRST FEW PEAKS TO REFERENCE
    f0_fit = np.poly1d(np.polyfit(f0_peaks[:3], ref_peaks[:3], 1))  # linear fit
    if np.abs(f0_fit.coef[0] - 2.5) > 0.05:
        print('fitting only 2...')
        f0_fit = np.poly1d(np.polyfit(f0_peaks[:2], ref_peaks[1:3], 1))

    ###################################################################################
    #f0_fit = np.poly1d(np.polyfit(f0_peaks[:3], ref_peaks[:3], 2))  # fit to quadratic
    #if np.abs(f0_fit.coef[0]/f0_fit.coef[1]) > BADNESS_THRESHOLD:  # the first reference peak was likely not detected
        #print('fitting linear...')
        #f0_fit = np.poly1d(np.polyfit(f0_peaks[:2], ref_peaks[1:3], 1))
    ###################################################################################

    f0_wvn = f0_fit(np.arange(len(f0_s)))  # calibrated wavenumbers
    f0_s = gnorm(f0_s)
    
    # BACKGROUND FITTING
    ## areas where there are no minima but we know they are at 0
    m1 = (f0_wvn > 1800) & (f0_wvn < 2000)
    m2 = (f0_wvn > 2900) & (f0_wvn < 3100)
    m3 = f0_wvn > 4200
    g = 20  # gap between points 

    ## concatenate minima and empty areas
    f0_min_wvns = [ *(f0_wvn[x] for x in f0_minima), *f0_wvn[m1][::g], *f0_wvn[m2][::g], *f0_wvn[m3][::g] ]
    f0_min_vals = [ *(f0_s[x] for x in f0_minima),   *f0_s[m1][::g],   *f0_s[m2][::g],   *f0_s[m3][::g] ]

    ## 11th degree polynomial fit
    f0_bckg = np.poly1d(np.polyfit(f0_min_wvns, f0_min_vals, 12))
    
    f0_signal = f0_s - f0_bckg(f0_wvn)

    # SAVE AS CSV
    s = 'wvn,raw,bckg,signal'
    for wvn, raw, bckg, signal in zip(f0_wvn, f0_s, f0_bckg(f0_wvn), f0_signal):
        s += f'\n{wvn},{raw},{bckg},{signal}'

    filename = path[:-3] + 'csv'
    with open(filename, 'w') as f:
        f.write(s)

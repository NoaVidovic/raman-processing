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


"""Parse a line from the reference spectrum"""
def readline_ref(line: str) -> tuple[float, float]:
    x, y = line.split('\t')
    return (float(x), float(y))


"""Calibrate from reference spectrum"""
def calibrate_from_reference(f0_s: np.ndarray, ref_path: str) -> np.ndarray:
    # FIND PEAKS (to calibrate) AND MINIMA (to fit background)
    f0_smoothed = gaussian_filter(f0_s, SMOOTHING_FACTOR)
    f0_peaks  = find_peaks( f0_smoothed, prominence=PROMINENCE_MAXIMA)[0]
    if DEBUG_MODE:
        with open('/tmp/peaks', 'w') as f:
            f.write(','.join([str(s) for s in f0_peaks]))


    # GET REFERENCE SPECTRUM OF CORUNDUM (SAPPHIRE GLASS) AND PEAKS
    with open(ref_path, 'r') as f:
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
    return f0_wvn


"""Calibrate from external calibration file"""
def calibrate_from_file(signal_loc: int, path: str) -> np.ndarray:
    return None


"""Find the area of interest of an image and return a spectrum"""
def image_to_spectrum(f0: np.ndarray, signal_loc=None) -> tuple[int, np.ndarray]:
    if signal_loc is None:
        ## first iteration center line
        f0_max1 = np.where(f0.sum(axis=1) == f0.sum(axis=1).max())[0][0]

        center_line = f0[f0_max1, :].astype(np.uint8)
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
    f0 = f0[:, :signal_loc+1]

    ## second iteration max
    f0_max = np.where(f0.sum(axis=1) == f0.sum(axis=1).max())[0][0]

    ## center cut (flip such that signal starts on the left) and spectrum
    f0 = np.flip(f0[f0_max-WIDTH:f0_max+WIDTH, :], axis=1)
    f0_s = f0.sum(axis=0, dtype=np.float64)

    return (signal_loc, f0_s)


"""Read spectrum from file"""
def file_to_spectrum(path: str) -> np.ndarray:
    return None


"""Remove background"""
def remove_bckg(data, halfwidth, sigma, pct):
    maxima = np.where((data[1:-1] > data[2:]) & (data[1:-1] > data[:-2]))[0] + 1
    minima = np.where((data[1:-1] < data[2:]) & (data[1:-1] < data[:-2]))[0] + 1
    extrema = np.concatenate((maxima, minima))

    noise_median = np.empty_like(data)
    for i in np.arange(data.shape[0]):
        m = max(i - halfwidth, 0)
        M = min(i + halfwidth, data.shape[0]-1)
    
        extreme_values = data[extrema[(extrema >= m) & (extrema <= M)]]
        if pct < 100:
            less_extreme_values = extreme_values[extreme_values < np.percentile(extreme_values, pct)]
            noise_median[i] = np.median(less_extreme_values)
        else:
            noise_median[i] = np.median(extreme_values)
    
    return data - gaussian_filter(noise_median, sigma)


# HELP TEXT
HELP_TEXT = """usage: get_spectrum [options]

Removes the background and (optionally) calibrates the spectrum

options:
  -h, --help                       show this help message and exit
  -i IMAGE                         get the spectrum from an image file
  -f FILE                          get the spectrum from a csv/tsv/ascii file
  -c FILE                          use an external calibration file
  -r REFERENCE                     use a reference spectrum to calibrate the input with

An error is raised if one uses both -i and -f options, or if one uses both -c and -r options"""


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
BCKG_HALFWIDTH = 20
BCKG_SIGMA = 10
BCKG_PCT = 80

# MAIN
if __name__ == '__main__':
    if '-h' in argv or len(argv) < 2:
        print(HELP_TEXT)
        exit(0)
    if '-i' in argv and '-f' in argv:
        print('Cannot use both -i and -f options.')
        exit(0)
    if not('-i' in argv or '-f' in argv):
        print('Use -i or -f to provide a spectrum.')
        exit(0)
    if '-c' in argv and '-r' in argv:
        print('Cannot use both -c and -r options.')
        exit(0)
    if not('-c' in argv or '-r' in argv):
        print('Use -c or -r to provide a calibration file or reference spectrum.')
        exit(0)

    if '-i' in argv:
        try:
            i = argv.index('-i') + 1
            path = argv[i]
            f0 = imread_gs(path)
            signal_loc, f0_s = image_to_spectrum(f0)
        except:
            print('File could not be found')
            exit(1)

    if '-f' in argv:
        try:
            i = argv.index('-f') + 1
            path = argv[i]
            f0_s = file_to_spectrum(path)
        except:
            print('File could not be found')
            exit(1)

    # BACKGROUND REMOVAL
    f0_signal = remove_bckg(f0_s, BCKG_HALFWIDTH, BCKG_SIGMA, BCKG_PCT)

    if '-r' in argv:
        i = argv.index('-r') + 1
        path_r = argv[i]
        f0_wvn = calibrate_from_reference(f0_s, path_r)

    if '-c' in argv:
        i = argv.index('-c') + 1
        path_c = argv[i]
        f0_wvn = calibrate_from_file(signal_loc, path_c)
    
    # SAVE AS CSV
    s = 'px,wvn,raw,signal'
    for px, wvn, raw, signal in zip(range(signal_loc, -1, -1), f0_wvn, f0_s, f0_signal):
        s += f'\n{px},{wvn},{raw},{signal}'

    filename = path[:-3] + 'csv'
    with open(filename, 'w') as f:
        f.write(s)

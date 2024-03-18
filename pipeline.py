import cv2
import matplotlib.pyplot as plt
import numpy as np

from inspect import signature
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Any, TypeAlias


### TYPE ALIASES

Image:    TypeAlias = np.ndarray[(Any, Any), np.dtype[np.uint8]]
Region:   TypeAlias = tuple[int, int, int, int]  # x0, x1, y0, y1
Spectrum: TypeAlias = np.ndarray[(Any,), np.dtype[np.float64]]


### IMAGE READING AND DISPLAY

def imread_gs(path: str) -> Image:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def show(img: Image) -> None:
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray')


### IMAGE TO RAW SPECTRUM CONVERSION

def roi(img: Image) -> Region:
    """Finds the Region of Interest (ROI) for a given image"""
    # hardcoded placeholder
    return (0, 1315, 25, 35)


def crop(img: Image, region: Region) -> Image:
    x0, x1, y0, y1 = region
    return img[y0:y1, x0:x1]


def raw_spectrum(img: Image) -> Spectrum:
    return crop(img, roi(img)).sum(axis=0, dtype=np.float64)


### NORMALIZATION

def gnorm(s: Spectrum):
    return (s - np.mean(s))/np.std(s)


### PEAK DETECTION

def smoothe(s: Spectrum) -> Spectrum:
    return gaussian_filter(s, 5)


def peaks(s: Spectrum) -> list[int]:
    return find_peaks(smoothe(gnorm(s)), prominence=0.05)[0]


### PEAK MATCHING
#######################################################
######################### NYI #########################
#######################################################


### BACKGROUND FITTING

def n_fun_pars(f):
    """Number of fittable parameters of a function (not counting x)"""
    return len(signature(f).parameters) - 1

def _fit_fun(x, bckg_fun, peak_means, *args):
    n_gaussians = len(peak_means)
    n_bckg_pars = n_fun_pars(bckg_fun)
    
    c1 = n_bckg_pars                    # first c1 args are pars for the bckg fit function
    c2 = n_bckg_pars + n_gaussians      # next c2-c1 args are amplitudes of the gaussians
    c3 = n_bckg_pars + 2 * n_gaussians  # last c3-c2 args are widths of the gaussians

    bckg_coeff  = args[:c1]
    peak_amps   = args[c1:c2]
    peak_widths = args[c2:c3]

    ret = bckg_fun(x, *bckg_coeff)
    for a, s, m in zip(peak_amps, peak_widths, peak_means):
        ret += a * np.exp(-(x-m)**2 / 2 / s**2)

    return ret


def gen_fit_fun(bckg_fun, peak_means):
    return lambda x, *args: _fit_fun(x, bckg_fun, peak_means, *args)


def init_params(bckg_init, n_peaks):
    return bckg_init + [1]*(2*n_peaks)


def get_bckg_fit(wvns, spec, bckg_fun, bckg_init):
    n_bckg_pars = n_fun_pars(bckg_fun)

    peak_means = [ wvns[x] for x in peaks(spec) ]

    fit_fun = gen_fit_fun(bckg_fun, peak_means)
    p0 = init_params(bckg_init, len(peak_means))

    par, cov = curve_fit(fit_fun, wvns, spec, p0=p0, xtol=1e-3, ftol=1e-3)

    return par, cov


def get_bckg_fit2(wvns, spec, bckg_fun, bckg_init):
    n_bckg_pars = n_fun_pars(bckg_fun)

    peak_means = []

    fit_fun = gen_fit_fun(bckg_fun, peak_means)
    p0 = init_params(bckg_init, len(peak_means))

    par, cov = curve_fit(fit_fun, wvns, spec, p0=p0)

    return par, cov


def remove_bckg(wvns, spec, bckg_fun, bckg_init):
    par, _ = get_bckg_fit(wvns, spec, bckg_fun, bckg_init)
    par = par[:n_fun_pars(bckg_fun)]

    return gnorm(spec - bckg_fun(wvns, *par))


def remove_bckg2(wvns, spec, bckg_fun, bckg_init):
    par, _ = get_bckg_fit2(wvns, spec, bckg_fun, bckg_init)
    par = par[:n_fun_pars(bckg_fun)]

    return gnorm(spec - bckg_fun(wvns, *par))

# Usage

To run the `get_spectrum` script, either create a virtual environment using the provided environment file (`conda env create -p venv -f environment.yml`; `conda activate ./venv`) or simply run it using a modern version of Python with OpenCV, Numpy and Scipy installed.

The script is called as such: `python get_spectrum.py X`, where X is the filename of the image you wish to process.
The script will make a .csv file with the fitted wavenumbers, raw spectrum, background fit and subtracted spectrum (raw minus background).

By default, the reference spectrum is to be located at `data/sapphire_ref_cm++.txt` with respect to the script.
This can be changed by opening the script with a text editor and modifying the `REFERENCE_PATH` variable.

# Introduction
This folder contains a Python codes for detecting flakes within images using HDBSCAN.

# Prerequisites
First make sure you have a version of [Python](https://www.python.org/) < 3.13 installed on your system. (3.13 restriction from https://github.com/numba/numba/issues/9413 - 14/10/2024)

Install the required packages with pip using `py -m pip install -r requirements.txt`.

# Installation
Install this package using: py -m pip install .\dist\flake_detector-x.x.x-py2.py3-none-any.whl
Replacing the x's above with the latest version numbers.

## Alternative fast_hdbscan with caching
A [fork of fast_hdbscan](https://github.com/Daniel-Forbes-HWU/fast_hdbscan_cached) was created, which added caching to the numba jit compilation. This can be installed from the wheel (dist/*.whl) found at the GitHub page or by running 
    `py -m pip install fast_hdbscan_cached-0.1.3-py3-none-any.whl`.

# Generic Flake Detection
The generic flake detection code is available in the `flake_detecor` package.
It can be imported with:
```Python3
from flake_detector import FlakeFindingResult, find_flakes
```

# Example
See the `example_detection.py` script to run flake detection for an included image of a terraced h-BN flake.

# Troubleshooting
At the time of writing, there have been issues compatibility issues with numpy version 2. Specifically Numba might be incompatible. If this is an issue, install and old version of numpy with e.g. `py -m pip install numpy=1.26`
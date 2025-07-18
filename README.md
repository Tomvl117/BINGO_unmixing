# BINGO unmixing.

## About this repo
A Python implementation of the BINGO blind unmixing algorithm.

## Installation
### Notes
This setup has so far only been verified on Windows-based, CUDA-accelerated machines. Testing has only been performed on
    CUDA 12.6. There are no reasons why 11.x should not work (check instructions), but your mileage may vary.
### Conda setup
```bash
conda create -n bingo-unmixing python=3.13
conda activate bingo-unmixing
```
### Local install
Clone the repository, then within the `bingo-unmixing` environment, navigate to the repository directory.
```bash
pip install -e .
pip install scikit-learn tifffile tqdm
```
### GPU acceleration
To use GPU-accelerated code, see the installation instructions on the [PyTorch webpages](https://pytorch.org/get-started/locally/). Run the following command, where
you may need to change the version number to the appropriate CUDA version on your system.
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
## Instructions
### Example code
See `examples/`.

## References
Xinyuan Huang, Xiujuan Gao, & Ling Fu. (2024) BINGO: a blind unmixing algorithm for ultra-multiplexing fluorescence images.
	Bioinformatics, Volume 40, Issue 2, February 2024, btae052, [10.1093/bioinformatics/btae052](https://doi.org/10.1093/bioinformatics/btae052)

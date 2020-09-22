# MultiCorr
*A Relativistic module for the **Multi**poles of 2-point **Corr**elation functions (version 1)*

This is a flexible and easy-to-use Python class to compute the relativistic multipoles of the 2PCF and power spectrum.

It includes all non-vanishing multipoles and **no** inputs are needed.

To set the fiducial cosmology, directly alter the main `.py`  file.

## Quick guide
To use MultiCorr you will need to put the main `MultiCorr.py` file in the folder you are working in. 

### Initializing the module
`import MultiCorr as corrfunc`

`cf = corrfunc.MultiCorr()`

### Getting help
`cf.help()`

### Computing multipoles up to l = 4 (correlation function)
`cf.multipoles(r,redshift,linear_bias1,linear_bias2,evolution_bias1,evolution_bias2,args)`

This returns all 4 non-vanishing multipoles, where `args` can be either a list or an array:

`args = ['linear',s_alpha,s_beta]` (s = magnification biases)

`args = ['linear']`

`True/False` to select *linear* or *non-linear* correlation functions.

## Dependencies 
MultiCorr depends upon `numpy`, `scipy`, and the python wrapper for the Boltzmann solver `Class`: 

https://github.com/lesgourg/class_public

## Primordial non-Gaussianities
Primordial non-Gaussianities are not currently implemented.

This feature will be available with version `v2`.

## License
You can use and alter the code freely.

Just kindly cite this webpage: https://github.com/cmguandalin/MultiCorr

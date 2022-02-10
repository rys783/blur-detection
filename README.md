# blur-detection
This is an optimized implementation of [the official Harr Wavelet based blur detection algorithm](https://github.com/pedrofrodenas/blur-Detection-Haar-Wavelet). The improvements include:
- Added a tunable parameter of kernel size for the max pooling on edge maps of various scales, which helps to reduce the false blur rate in my experiments
- Removed all the for loops in the original implementation and replacing them with the more efficient numpy ndarray operations. My experiments show >2X speedup


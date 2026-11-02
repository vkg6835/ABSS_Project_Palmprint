"""
AI461 Phase II — palmprint verification.

Primary system : RLOC  (Jia, Huang & Zhang, Pattern Recognition 41 (2008)
                        1504-1513, "Palmprint verification based on robust
                        line orientation code")
Baseline       : Han, Cheng, Lin & Fan, Pattern Recognition 36 (2003) 371-381,
                 "Personal authentication using palm-print features"

Dependencies: numpy and Pillow only. No scipy, no OpenCV, no network.
Tested on Python 3.10 and 3.14.
"""

__all__ = ["mfrat", "rloc", "han", "metrics", "protocol", "synthetic"]
__version__ = "0.1.0"

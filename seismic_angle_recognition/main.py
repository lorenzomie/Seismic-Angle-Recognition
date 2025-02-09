import sys
sys.path.append("../ext/axitra/MOMENT_DISP_F90_OPENMP/src") 

import numpy as np
from axitra import *
import matplotlib.pyplot as pt


sources=np.array([[1, 45.100, 2.000, 5000.000],
                  [2, 45.200, 2.000, 5000.000]])
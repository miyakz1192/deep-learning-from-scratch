import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"

from threadpoolctl import threadpool_info
import numpy as np
from pprint import pp
print(np.__version__)
#print(np.__config__.blas_opt_info['libraries'])

pp(threadpool_info())

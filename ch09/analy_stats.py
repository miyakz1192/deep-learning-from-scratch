import pstats
from pstats import SortKey

sts = pstats.Stats('/tmp/res')

#sts.strip_dirs().sort_stats(-1).print_stats()
sts.sort_stats( SortKey.CUMULATIVE)
sts.print_stats()

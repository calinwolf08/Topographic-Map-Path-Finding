#python -m cProfile [-o output_file] [-s sort_order] myscript.py
import pstats, sys
stats = pstats.Stats(sys.argv[1])
stats.sort_stats("tottime")
stats.print_stats(20)

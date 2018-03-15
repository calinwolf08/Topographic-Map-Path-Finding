import pstats, sys
stats = pstats.Stats(sys.argv[1])
stats.sort_stats("tottime")
stats.print_stats(20)

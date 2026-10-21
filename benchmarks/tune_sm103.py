"""Tune B300 attention tiles with precision checks and CUDA-graph timings."""

from benchmarks.tune_sm120 import main

if __name__ == "__main__":
    main((10, 3))

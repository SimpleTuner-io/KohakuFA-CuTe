"""Tune the sm_100 warp-MMA attention path on datacentre Blackwell."""

from benchmarks.experiments.blackwell.tune_sm120 import main

if __name__ == "__main__":
    main((10, 0))

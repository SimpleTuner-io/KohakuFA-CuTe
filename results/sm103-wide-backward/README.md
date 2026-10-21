# B300 wider backward dispatch

Eligible sm_103 attention now uses the 128-column centered backward tile table
already used by sm_100. A separate sm_103 copy allows future full-device tuning
to diverge. This activates both 64-column probability/dS consumer groups and
32-column dP reads on B300. Existing alignment checks, short-sequence crossovers,
explicit-mask fallbacks and centering arithmetic apply as before.

The decision uses measurements on a B300 SXM6 AC MIG 1g.34gb: 18 SMs, sm_103,
32,749,125,632 usable bytes. These measurements establish partition performance
and sm_103 functionality. Full-B300 throughput still needs a full-device run.

## Isolated table comparison

The five-backend [probability-tile reports](../blackwell-probability-tiles/README.md)
rotate original/new 64-column, original/new 128-column and Gluon variants on the
same inputs. Comparing **new-64 against new-128** holds the merged bounds-check
optimization fixed and isolates the table change in this PR. All 18 comparisons
improve by 2.8–50.1%: both input dtypes, S1024/S4096, D64/D128/D256/D512 in dense
attention, plus BF16 S4096/D128 and D512 causal attention.

BF16 S4096, B=1/H=16, forward plus backward including forward, milliseconds:

| D | 64-column after bounds change | Wider default | Gluon | Less CuTe time |
| --- | ---: | ---: | ---: | ---: |
| 64 | 5.5269 | 4.1928 | 1.9761 | 24.1% |
| 128 | 6.1210 | 5.5562 | 3.3707 | 9.2% |
| 256 | 14.2861 | 11.5494 | 7.7511 | 19.2% |
| 512 | 71.9857 | 38.3358 | 24.8099 | 46.7% |

[Derived comparison](configuration-comparison.json) identifies every source row
and its report hash. Bitwise equality is tested separately at each fixed tile
width for the bounds optimization. The table-width change retains the FP32
centering formula and passes the centered FP64 tiny-gradient checks.

Compared with the original default before either improvement, BF16 S4096/D512
falls from 74.3032 to 38.3358 ms, a 48.4% reduction on the partition. Gluon remains
faster at 24.8099 ms. This does not claim fused Q/K/V gradients: query and
key/value passes remain separate and repeat intermediate work.

## Actual production dispatch

The [production benchmark](production-speed.json) runs `benchmarks.speed`
without a configuration override. Its four rounds of 20 graph replays verify
all captured outputs/gradients and select 128-column, twelve-warp backward
configurations. BF16 S4096/H16 gives 5.5604 ms at D128 and 38.3442 ms at D512,
versus Gluon's 3.3691 and 24.8263 ms. These agree with the controlled table
comparison. The report records the actual configurations.

Four [GPU regression methods](sm103-tests.log) pass on the sm_103 partition:
production below/above/below threshold calls preserve nonzero saturated gradients
against FP64; fused tails/GQA/signed and zero scales pass; all four dimensions in
FP16/BF16 pass the constructed tiny-gradient cases; and captured inputs/dO can
change between graph replays. The three existing fused tests now run on both
sm_100 and sm_103. [Validation](validation.json) records 16 graph-verified
production timing samples in addition to the four GPU methods.

[Provenance](provenance.json) records the verified local source/log archive,
its SHA-256, the runtime Git reference and 35 byte-matched runtime files.
The separate preceding archive holds all 729 verified profiling/experiment
files and the 360-sample rotating comparison.

## Reproduction

```bash
python -m benchmarks.speed --dtype bf16 --seq 4096 --dim 128 512 \
  --backends cute kohakufa --iterations 20 --repeats 4 \
  --kohakufa-tuning results/b300-gluon-d128.json results/b300-gluon-d512.json \
  --output artifacts/sm103-production-speed.json
python -m unittest -v \
  tests.test_kernels.CuTeAttentionTests.test_sm103_fused_dispatch_crosses_sequence_threshold \
  tests.test_kernels.CuTeAttentionTests.test_blackwell_fused_probability_and_ds_pipeline \
  tests.test_kernels.CuTeAttentionTests.test_blackwell_fused_pipeline_retains_saturated_gradients \
  tests.test_kernels.CuTeAttentionTests.test_blackwell_fused_pipeline_reads_changed_graph_inputs
```

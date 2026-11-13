# Hopper experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## Explicit H200 studies

Experimental pipeline, masking, persistent scheduling and initialization. [H200 limits](../../../docs/experimental/h200.md), [measured evidence](../../../results/h200.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [compare_hopper_lpt](compare_hopper_lpt.py) | CLI | Compare the causal scheduler intervention in matched rotating rounds. |
| [compare_hopper_masking](compare_hopper_masking.py) | CLI | Compare separate masking against the same causal interleaved policy. |
| [compare_hopper_runner](compare_hopper_runner.py) | CLI | Compare the native Hopper host against the same interleaved Python policy. |
| [configured_hopper](configured_hopper.py) | Imported helper / snapshot; `make_attention` | Explicit Hopper native-host experiment with the causal interleaved policy. |
| [experimental_hopper_lpt](experimental_hopper_lpt.py) | Imported helper / snapshot; `make_attention`, `make_control_attention`, `make_ordinary_attention`, `make_separate_masking_attention` | Unqualified causal head-interleaving experiment; never selected by default. |
| [hopper_ordinary_gradient_stores](hopper_ordinary_gradient_stores.py) | Imported helper / snapshot | Explicit unqualified ordinary-store control for TMA initialization tracking. |
| [initcheck_hopper](initcheck_hopper.py) | CLI | Isolate static Hopper gradient-output tracking with optional NaN allocation. |
| [qualify_hopper](qualify_hopper.py) | CLI | Qualify a Hopper attention factory on directed inputs and changed graphs. |

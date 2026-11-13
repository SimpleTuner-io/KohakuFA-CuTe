# Shared experiment catalog

See the [shared invocation and status guidance](../README.md#source-catalogs).
Evidence links identify each study family, not a measured result for every helper.

## Cross-device and application experiments

Joint backward probes and captured application work. [Retained controls](../results/controls.json), [application evidence with failed logit gate](../results/clef.json).

| Module | Invocation / role | Question or retained purpose |
| --- | --- | --- |
| [clef_forward](clef_forward.py) | CLI; `make_record` | Capture and compare real text-only Clef attention and model forwards. |
| [experimental_joint_backward](experimental_joint_backward.py) | Imported helper / snapshot | Experimental centered prepass and key-owned joint backward for SM89. |
| [probe_joint_backward](probe_joint_backward.py) | CLI | Compare the experimental centered joint backward on a qualified RTX 4090. |
| [warp_inference](warp_inference.py) | CLI | Compare explicit warp inference with the unchanged training entry under NoGrad. |

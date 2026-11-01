"""Hardware and optional-stack bounds for experimental Hopper GPU fixtures."""

import torch


def full_h200_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    properties = torch.cuda.get_device_properties(0)
    if properties.name not in ("NVIDIA H200", "NVIDIA H200 NVL") or properties.multi_processor_count != 132:
        return False
    from kohakufa_cute.fa4_hopper import available

    return available()

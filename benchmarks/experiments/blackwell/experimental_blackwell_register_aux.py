"""Expose the wide backward directly for readout/reference measurements.

Production selects this kernel only for eligible long dense/causal BF16 D512 inputs;
the readout driver can measure it directly outside that measured dispatch scope.
"""

from kohakufa_cute.blackwell_register_aux import JointBackward

WORKSPACE_PLANES = 0

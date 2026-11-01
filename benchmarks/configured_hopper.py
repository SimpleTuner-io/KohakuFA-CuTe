"""Explicit Hopper native-host experiment with the causal interleaved policy.

Run this factory in a fresh process: the native host caches one compiled policy
per family/configuration/tensor metadata. No alternate Hopper runner policy is
installed in the same process. Other backend comparisons use their own APIs.
"""


def make_attention():
    from kohakufa_cute import native
    from kohakufa_cute.fa4_hopper import _native

    original = _native.FlatBackward

    class InterleavedFlatBackward(original):
        def __init__(self, *arguments, **options):
            super().__init__(*arguments, **options)
            self.main.causal_head_interleave = True

    # This is explicit experimental host construction. The GPU method bodies
    # remain the static port; the production package's default is unchanged.
    _native.FlatBackward = InterleavedFlatBackward
    runners = {}

    def attention(q, k, v, *, is_causal=False, scale=None, enable_gqa=False):
        if enable_gqa:
            raise ValueError("The configured Hopper experiment requires equal Q/K/V heads.")
        dimension = q.shape[-1]
        if dimension not in runners:
            config = _native.configuration_for(dimension) | {"maximum_multiple": 128}
            runners[dimension] = native.extension().AttentionRunner("hopper", config)
        return runners[dimension](q, k, v, scale, is_causal)

    return attention

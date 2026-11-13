import sys
import types
import unittest
from unittest.mock import patch

import torch

from benchmarks.neighborhood.backends import make_backend, neighborhood_predicate, spatial_block_lists
from benchmarks.neighborhood.reference import neighborhood_mask


class NeighborhoodBackendTests(unittest.TestCase):
    def test_spatial_blocks_and_predicate_match_independent_small_masks(self):
        for height in range(1, 6):
            for width in range(1, 6):
                length = height * width
                q = torch.arange(length)[:, None]
                k = torch.arange(length)[None, :]
                for window_h in range(1, height + 1):
                    for window_w in range(1, width + 1):
                        spatial, window = (height, width), (window_h, window_w)
                        mask = neighborhood_mask(spatial, window)
                        self.assertTrue(torch.equal(neighborhood_predicate(spatial, window)(0, 0, q, k), mask))
                        for block_size in (2, 8, 128):
                            counts, indices = spatial_block_lists(spatial, window, block_size)
                            for block, (count, row) in enumerate(zip(counts, indices)):
                                actual = row[:count]
                                expected = sorted(
                                    {
                                        int(key) // block_size
                                        for key in mask[block * block_size : (block + 1) * block_size].nonzero(
                                            as_tuple=False
                                        )[:, 1]
                                    }
                                )
                                self.assertEqual(actual, expected)

    def test_blockmask_transpose_preserves_asymmetric_inverse(self):
        from torch.nn.attention.flex_attention import BlockMask

        counts, indices = spatial_block_lists((1, 5), (1, 3), 1)
        mask = BlockMask.from_kv_blocks(
            torch.tensor(counts, dtype=torch.int32)[None, None],
            torch.tensor(indices, dtype=torch.int32)[None, None],
            BLOCK_SIZE=1,
            mask_mod=neighborhood_predicate((1, 5), (1, 3)),
            seq_lengths=(5, 5),
        )
        self.assertEqual(mask.q_num_blocks[0, 0].tolist(), [2, 3, 5, 3, 2])
        self.assertEqual(mask.q_indices[0, 0, 2, :5].tolist(), [0, 1, 2, 3, 4])

    def test_natten_layout_and_gradient_copies_belong_to_call(self):
        seen = []

        def na2d(q, k, v, **kwargs):
            self.assertEqual(q.shape, (2, 3, 5, 2, 64))
            self.assertTrue(all(t.is_contiguous() for t in (q, k, v)))
            seen.append(kwargs)
            return v * 2

        package = types.ModuleType("natten")
        package.na2d = na2d
        backends = types.ModuleType("natten.backends")
        backends.choose_backend = lambda *args, **kwargs: "hopper-fna"
        backends.choose_fmha_backend = lambda *args, **kwargs: "blackwell-fmha"
        with patch.dict(sys.modules, {"natten": package, "natten.backends": backends}):
            adapter = make_backend("natten", (3, 5), (3, 3), device="cpu")
            inputs = [torch.randn(2, 2, 15, 64, requires_grad=True) for _ in range(3)]
            output = adapter.call(*inputs)
            self.assertTrue(output.is_contiguous())
            torch.testing.assert_close(output, inputs[2] * 2)
            gradient = torch.autograd.grad(output.sum(), inputs[2])[0]
            torch.testing.assert_close(gradient, torch.full_like(inputs[2], 2))
            self.assertEqual(adapter.metadata["selected_backend"], "hopper-fna")
            self.assertEqual(seen[0]["backend"], "hopper-fna")
            self.assertEqual(seen[0]["scale"], 0.125)

            full = make_backend("natten", (3, 5), (3, 5), device="cpu")
            full.call(*inputs)
            self.assertEqual(full.metadata["selected_backend"], "blackwell-fmha")
            self.assertEqual(seen[-1]["attention_kwargs"], {"backend": "blackwell-fmha"})

    def test_natten_flex_fallback_is_an_execution_failure(self):
        package = types.ModuleType("natten")
        package.na2d = lambda *args, **kwargs: None
        backends = types.ModuleType("natten.backends")
        backends.choose_backend = lambda *args, **kwargs: "flex-fna"
        backends.choose_fmha_backend = lambda *args, **kwargs: "flex-fmha"
        with patch.dict(sys.modules, {"natten": package, "natten.backends": backends}):
            adapter = make_backend("natten", (3, 5), (3, 3), device="cpu")
            inputs = [torch.zeros(1, 1, 15, 64) for _ in range(3)]
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "independent libnatten"):
                    adapter.call(*inputs)


if __name__ == "__main__":
    unittest.main()

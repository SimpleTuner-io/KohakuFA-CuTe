"""CuTe attention extracted from SimpleTuner, inspired by KohakuFA."""

from .api import attention, automatic_scaled_dot_product_attention, scaled_dot_product_attention

__version__ = "0.1.0"
__all__ = ["attention", "scaled_dot_product_attention", "automatic_scaled_dot_product_attention"]

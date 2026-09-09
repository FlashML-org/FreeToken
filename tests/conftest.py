import pytest


@pytest.fixture(autouse=True)
def _default_quant_backend():
    """Tests that install kernel requests must not leak them into the next test."""
    yield
    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend())

import pytest

from freetoken.core import SamplingParams


@pytest.mark.parametrize(
    ("params", "expected"),
    (
        (SamplingParams(temperature=0.0, top_k=-1, top_p=0.95), True),
        (SamplingParams(temperature=-1.0, top_k=50, top_p=0.1), True),
        (SamplingParams(temperature=0.7, top_k=1, top_p=1.0), True),
        (SamplingParams(temperature=0.7, top_k=1, top_p=0.95), False),
        (SamplingParams(temperature=0.7, top_k=-1, top_p=1.0), False),
    ),
)
def test_is_greedy_contract(params, expected):
    assert params.is_greedy is expected

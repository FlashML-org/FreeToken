"""FreeListAllocator (the DSV4 pool's page-granular window free-list) unit tests."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v4_pool import FreeListAllocator

DEVICE = torch.device("cpu")
P = 128


def test_free_list_conservation_alloc_all_free_all():
    a = FreeListAllocator(capacity=10, device=DEVICE, page_unit=1)
    assert a.available() == 10
    taken = a.alloc(10)
    assert a.available() == 0
    assert sorted(taken.tolist()) == list(range(10))
    a.free(taken)
    assert a.available() == 10


def test_free_list_lifo_recycle_order():
    a = FreeListAllocator(capacity=5, device=DEVICE, page_unit=1)
    # alloc takes the tail slice (ascending within the slice), leaving the head.
    first = a.alloc(2)
    assert first.tolist() == [3, 4]
    assert a.available() == 3  # [0, 1, 2] remain
    # LIFO: the just-freed slots go back on the tail and are popped first.
    a.free(torch.tensor([4, 3], dtype=torch.int64))
    second = a.alloc(2)
    assert second.tolist() == [4, 3]
    # the head slot is only handed out once the recycled tail is exhausted.
    assert a.alloc(1).tolist() == [2]


def test_free_list_page_unit_bases_are_multiples():
    a = FreeListAllocator(capacity=8 * P, device=DEVICE, page_unit=P)
    assert a.capacity == 8 * P
    assert a.available() == 8 * P
    bases = a.alloc(3)
    assert all(int(b) % P == 0 for b in bases)
    assert a.available() == (8 - 3) * P


def test_free_list_overflow_raises():
    a = FreeListAllocator(capacity=4, device=DEVICE, page_unit=1)
    with pytest.raises(RuntimeError):
        a.alloc(5)


def test_free_list_reset():
    a = FreeListAllocator(capacity=4 * P, device=DEVICE, page_unit=P)
    a.alloc(2)
    a.reset()
    assert a.available() == 4 * P


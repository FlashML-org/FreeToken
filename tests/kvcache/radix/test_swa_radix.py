"""SWARadixCache: the behaviour that is irreducibly SWA.

Everything universal (slot conservation, counter drift, prefix closure, LRU victim order, model
agreement) is checked after every op by the harness battery, so this module only writes down what
the *second currency* adds: tombstones, the three insert-side revive branches, windowed match
truncation, the two-tier lock, the two-tier eviction with its cascade, the finish-time soft pin and
``trim_head_swa``.

One module covers all three geometries: ``swa-p1-w4`` (window = 4 pages), ``swa-p4-w8`` (2 pages)
and ``swa-p128-w128`` (1 page -- the DSV4 shape, where one page *is* the window and the windowed
math degenerates).  Scenarios are therefore written in *pages* and in ``_window_pages(spec)``, never
in literal token counts, so no scenario is written twice.

Expectations come from the page-keyed reference model (``session.model``); the implementation is
only ever asked and checked.  Assertions here read the model's own nodes -- which the after-op
battery has already pinned to the implementation node-for-node -- plus closed-form values derived
from the geometry, so a test can fail on either side of the comparison.
"""
from __future__ import annotations

import pytest

from .driver import Session
from .adapters import check_class_integrity, check_counter_model, check_model_agreement
from .model import InvariantViolation

# --------------------------------------------------------------------------- fixtures / helpers


@pytest.fixture
def sess(swa_spec, make_session) -> Session:
    """A Session over one SWA geometry (the ``session`` fixture spans all three classes)."""
    return make_session(swa_spec)


def _pages(spec, indices) -> tuple:
    """Token ids for the given page numbers; distinct pages never share a token."""
    P = spec.page_size
    out: list = []
    for i in indices:
        out.extend([i + 1] * P)
    return tuple(out)


def _seq(spec, n_pages: int, start: int = 0) -> tuple:
    return _pages(spec, range(start, start + n_pages))


def _window_pages(spec) -> int:
    """How many whole pages the sliding window spans (1 for the DSV4 page==window shape)."""
    return -(-spec.window // spec.page_size)


def _ev(s: Session):
    """Reference-model branch counters (see the harness's branch-coverage list)."""
    return s.model.events


def _groups(s: Session) -> list:
    return s.model.trie.groups()


def _lengths(s: Session, tomb: bool) -> list:
    return sorted(g.length for g in _groups(s) if g.tomb is tomb)


def _tombstoned(s: Session) -> set:
    return {g.end for g in _groups(s) if g.tomb}


def _node(s: Session, ids, n_pages: int):
    """The model node owning the page that ends at ``ids[:n_pages * P]``."""
    return s.model.trie.page_group[tuple(ids[: n_pages * s.P])]


def _sizes(s: Session) -> dict:
    return s.model.counters()


def _chain(s: Session, n_pages: int) -> tuple:
    """root -> N0 -> N1 -> ... , one distinct live single-page node each (incremental commits)."""
    ids = _seq(s.spec, n_pages)
    for k in range(1, n_pages + 1):
        s.do_insert(ids[: k * s.P])
    s.check()
    return ids


# --------------------------------------------------------------------------- insert: tombstones
def test_insert_tombstones_the_out_of_window_head(sess):
    """``swa_evicted_seqlen`` splits a fresh commit into a tombstoned head (full KV only) and a
    live tail, on a page boundary."""
    P = sess.P
    sess.do_insert(_seq(sess.spec, 3), swa_evicted=P)
    sess.check()

    assert _ev(sess)["insert.suffix_tombstone"] == 1
    assert _ev(sess)["insert.suffix_live"] == 1
    assert _lengths(sess, True) == [P]
    assert _lengths(sess, False) == [2 * P]
    assert _sizes(sess) == {"full_evictable": 3 * P, "full_protected": 0,
                            "swa_evictable": 2 * P, "swa_protected": 0}


@pytest.mark.parametrize("n_pages", [1, 2, 3])
def test_suffix_clamp_never_creates_a_tombstone_leaf(swa_spec, make_session, n_pages):
    """The suffix boundary is clamped to leave >= one live page, so a leaf is never a tombstone --
    the invariant ``evict_full``'s cascade and the windowed match both lean on."""
    s = make_session(swa_spec)
    P = s.P
    s.do_insert(_seq(swa_spec, n_pages), swa_evicted=(n_pages + 3) * P)   # far past the commit
    s.check()

    assert _lengths(s, True) == ([] if n_pages == 1 else [(n_pages - 1) * P])
    assert _lengths(s, False) == [P]
    leaves = [g for g in _groups(s) if s.model.trie.is_leaf(g)]
    assert leaves and not any(g.tomb for g in leaves)


# --------------------------------------------------------------------------- insert: revive
def test_insert_revives_a_whole_tombstoned_node(sess):
    """Branch 1: the request's swa is live over the node's whole span (``swa_evicted <= start``) ->
    adopt the request's slots, hand the stale tree slots back, clear the tombstone."""
    P, wp = sess.P, _window_pages(sess.spec)
    ids = _chain(sess, wp + 1)
    sess.do_match(ids)                       # stamp decreasing toward root -> N0 is the LRU
    sess.do_evict_second(1)                  # tombstone N0 in place
    assert _tombstoned(sess) == {tuple(ids[:P])}

    stale = list(_node(sess, ids, 1).slots)
    fresh = sess.kv.take(len(ids))
    _, exp = sess.do_insert(ids, slots=fresh, swa_evicted=0)
    sess.check()

    assert _ev(sess)["insert.revive_whole"] == 1
    assert _tombstoned(sess) == set()
    assert _node(sess, ids, 1).slots == fresh[:P]        # adopted the request's live-swa slots
    assert set(stale) <= set(exp.freed)                  # the sentinel-mapped tree slots came back
    assert set(fresh[P:]) <= set(exp.freed)              # dups of the still-live nodes came back
    assert _sizes(sess)["swa_evictable"] == (wp + 1) * P


def test_insert_splits_and_revives_the_live_tail(sess):
    """Branch 2: the request's freed-swa frontier falls strictly inside a multi-page tombstone ->
    split AT THE FRONTIER, keep the out-of-window head tombstoned, revive only the tail past it."""
    P = sess.P
    ids = _seq(sess.spec, 4)
    first = sess.kv.take(len(ids))
    sess.do_insert(ids, slots=first, swa_evicted=3 * P)     # tomb [0,3P) + live [3P,4P)
    sess.check()
    assert _lengths(sess, True) == [3 * P]

    second = sess.kv.take(len(ids))
    _, exp = sess.do_insert(ids, slots=second, swa_evicted=2 * P)   # frontier inside the tombstone
    sess.check()

    assert _ev(sess)["insert.revive_tail"] == 1
    assert _lengths(sess, True) == [2 * P]                 # the head keeps BOTH tombstoned pages
    tail = _node(sess, ids, 3)                             # the revived page is [2P, 3P)
    assert not tail.tomb and tail.slots == second[2 * P: 3 * P]
    assert set(first[2 * P: 3 * P]) <= set(exp.freed)      # the tail's stale tree slots came back
    assert set(second[: 2 * P]) <= set(exp.freed)          # the head's dup came back (still tomb)
    assert _sizes(sess)["swa_evictable"] == 2 * P          # the revived page + the live tail


def test_insert_keeps_a_still_out_of_window_tombstone(sess):
    """Branch 3: the node is wholly below the request's freed-swa frontier -> nothing to revive,
    only the request's duplicate is handed back and the tree's slots are untouched."""
    P, wp = sess.P, _window_pages(sess.spec)
    ids = _chain(sess, wp + 1)
    sess.do_match(ids)
    sess.do_evict_second(1)                                # tombstone N0
    stale = list(_node(sess, ids, 1).slots)

    fresh = sess.kv.take(len(ids))
    _, exp = sess.do_insert(ids, slots=fresh, swa_evicted=P)
    sess.check()

    assert _ev(sess)["insert.keep_tombstone"] == 1
    assert _ev(sess)["insert.revive_whole"] == 0
    head = _node(sess, ids, 1)
    assert head.tomb and head.slots == stale
    assert set(fresh[:P]) <= set(exp.freed)
    assert _sizes(sess)["swa_evictable"] == wp * P


def test_insert_refuses_to_revive_a_full_locked_tombstone(sess):
    """A full-locked reader still gathers the node's CURRENT slots through its own row, so reviving
    under a full lock would hand live KV to the next allocation.  The tombstone must survive the
    insert untouched (Branch-3 shape) and only become revivable once the lock clears."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = wp + 2
    ids = _chain(sess, n)
    held = sess.do_lock(ids)                     # full-pins the path, swa-pins the trailing window
    assert held is not None
    sess.do_evict_second(10 ** 6)                # tombstones the root-ward, swa-unlocked nodes
    sess.check()
    assert _tombstoned(sess) == {tuple(ids[:P]), tuple(ids[: 2 * P])}
    pinned = {g.end: list(g.slots) for g in _groups(sess) if g.tomb}

    fresh = sess.kv.take(len(ids))
    _, exp = sess.do_insert(ids, slots=fresh, swa_evicted=0)
    sess.check()

    assert _ev(sess)["insert.locked_no_revive"] == 2
    assert _ev(sess)["insert.revive_whole"] == 0
    for end, slots in pinned.items():
        assert sess.model.trie.page_group[end].slots == slots
    assert set(exp.freed) == set(fresh)          # ONLY the incoming dup, never the tree's value

    sess.do_unlock(held)
    sess.do_insert(ids, slots=sess.kv.take(len(ids)), swa_evicted=0)
    sess.check()
    assert _ev(sess)["insert.revive_whole"] == 2         # lock cleared -> reviving is safe again
    assert _tombstoned(sess) == set()


def test_insert_within_the_reused_prefix_frees_nothing(sess):
    """``update_kv_after_len`` guard: nodes entirely inside the request's reused prefix hold the
    tree's own slots, so insert must neither free them (double free) nor revive them."""
    ids = _chain(sess, 3)
    held = sess.do_lock(ids)
    m, _ = sess.do_match(ids)
    assert m.cached_len == len(ids)
    before = _ev(sess)["insert.dup_live"]

    got, exp = sess.do_insert(ids, slots=list(m.indices), reused_len=m.cached_len)
    sess.check()

    assert got.matched_len == len(ids)
    assert got.freed == [] and exp.freed == []
    assert _ev(sess)["insert.dup_live"] == before        # nothing was treated as a duplicate
    sess.do_unlock(held)
    sess.check()


def test_a_split_propagates_the_tombstone_to_both_halves(sess):
    """A tombstone covers all of a node's tokens, so a divergent query splitting one must leave
    BOTH halves tombstoned -- a live half would advertise swa KV that has already been freed."""
    P = sess.P
    ids = _seq(sess.spec, 3)
    sess.do_insert(ids, swa_evicted=2 * P)                 # tomb [0,2P) + live [2P,3P)
    other = ids[:P] + _seq(sess.spec, 1, start=9)          # shares page 0, diverges at page 1

    got, _ = sess.do_match(other)
    sess.check()

    assert _ev(sess)["node.split"] == 1
    assert _lengths(sess, True) == [P, P]                  # both halves of the split tombstone
    assert _lengths(sess, False) == [P]
    assert got.cached_len == 0                             # ending at a tombstone is not reusable


def test_a_lock_handle_survives_a_split_at_its_window_boundary(sess):
    """``split_at`` keeps the original object as the SUFFIX and migrates the window handle to the
    new prefix node, so a lock taken before a split still releases exactly its own window."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = max(2, wp)
    ids = _seq(sess.spec, n)
    sess.do_insert(ids)                                    # a single node of n pages
    held = sess.do_lock(ids)
    assert held is not None and held[0].uuid is not None
    assert _sizes(sess)["swa_protected"] == n * P

    other = ids[:P] + _seq(sess.spec, 1, start=9)          # diverges after page 0 -> splits it
    sess.do_match(other)
    sess.check()
    assert _ev(sess)["node.split"] == 1
    assert sorted(g.length for g in _groups(sess)) == sorted([P, (n - 1) * P])

    sess.do_unlock(held)
    sess.check()
    assert _sizes(sess)["swa_protected"] == 0 and _sizes(sess)["full_protected"] == 0


# --------------------------------------------------------------------------- windowed match
def test_match_reuses_a_short_tombstone_free_prefix(sess):
    """A path that reaches the root without a tombstone is reusable however short it is: there is
    no freed swa behind it for the window to fall into."""
    P = sess.P
    ids = _seq(sess.spec, 1)
    sess.do_insert(ids)
    got, _ = sess.do_match(ids)
    sess.check()

    assert got.cached_len == P                            # even though P may be < the window
    assert _ev(sess)["match.windowed_truncation"] == 0


def test_match_truncates_until_the_live_run_covers_the_window(sess):
    """The windowed-reuse boundary accumulates across MULTIPLE nodes: after a tombstone, nothing is
    swa-reusable until the contiguous live run behind the query end reaches the window."""
    P, W = sess.P, sess.spec.window
    wp = _window_pages(sess.spec)
    ids = _seq(sess.spec, wp + 1)

    sess.do_insert(ids[: 2 * P], swa_evicted=P)           # tomb page 0 + live page 1
    for k in range(3, wp + 2):                            # extend one live page at a time
        sess.do_insert(ids[: k * P], swa_evicted=P)       # swa_evicted=P keeps the head tomb
    sess.check()
    assert _lengths(sess, True) == [P]
    assert _lengths(sess, False) == [P] * wp              # the live run really spans wp nodes

    for k in range(1, wp + 2):
        got, _ = sess.do_match(ids[: k * P])
        live_run = (k - 1) * P                            # tokens between the tombstone and the end
        assert got.cached_len == (k * P if live_run >= W else 0)
    sess.check()
    assert _ev(sess)["match.windowed_truncation"] > 0


def test_a_live_run_of_exactly_one_window_between_two_tombstones_is_reusable(sess):
    """The boundary commit at a tombstone is ``>=`` the window, not ``>``: a run of EXACTLY the
    window is covered by it.

    Both tombstone paths are needed to reach this shape, because neither gets there alone.
    ``_stamp_path`` restamps ancestors, so LRU can never age a root-side node past its own
    descendants -- the head tombstone has to come from ``trim_head_swa``. And ``evict_swa``
    unlinks free leaves rather than tombstoning them, so the second one has to be an INTERNAL
    node, which means leaving a live node below it.
    """
    P, wp = sess.P, _window_pages(sess.spec)
    if wp == 1:
        pytest.skip("page == window: the live node below the second tombstone already covers the "
                    "window on its own, so the post-loop commit fires and takes the whole chain -- "
                    "the boundary commit at the tombstone cannot be isolated in this geometry")
    n = wp + 3                                            # tomb | wp live (== W) | tomb | live
    ids = _chain(sess, n)

    sess.do_trim(ids, P)                                  # node 1 -> tombstone
    sess.do_match(ids[: (1 + wp) * P])                    # restamp 1..1+wp -> 2+wp is oldest live
    sess.do_evict_second(1)                               # internal, so tombstoned in place
    sess.check()

    assert _tombstoned(sess) == {tuple(ids[:P]), tuple(ids[: (2 + wp) * P])}
    assert _lengths(sess, False) == [P] * (wp + 1)        # the live run really is wp nodes

    got, _ = sess.do_match(ids)
    # The run between the tombstones is wp * P == W exactly, so the boundary commits there. Under
    # ``>`` nothing commits anywhere: the head tombstone commits at the root, this one is skipped,
    # and the trailing live node is only P < W -- cached_len collapses to 0.
    assert got.cached_len == (1 + wp) * P


# --------------------------------------------------------------------------- two-tier lock
def test_inc_lock_returns_a_uuid_only_when_the_window_is_covered(sess):
    """``inc_lock`` stamps the window boundary and returns its handle; a live path shorter than the
    window has no boundary to stamp, so it returns None while still swa-pinning what it has."""
    P, wp = sess.P, _window_pages(sess.spec)
    ids = _seq(sess.spec, wp)

    if wp > 1:
        short = ids[: (wp - 1) * P]
        sess.do_insert(short)
        held = sess.do_lock(short)
        assert held is not None and held[0].uuid is None
        assert _sizes(sess)["swa_protected"] == (wp - 1) * P
        sess.do_unlock(held)
        sess.check()

    sess.do_insert(ids)
    held = sess.do_lock(ids)
    assert held is not None and held[0].uuid is not None
    assert _sizes(sess)["swa_protected"] >= sess.spec.window
    sess.do_unlock(held)
    sess.check()
    assert _sizes(sess)["swa_protected"] == 0


def test_dec_lock_with_a_uuid_releases_only_its_own_window(sess):
    """Two readers whose windows sit at different depths: releasing the deep one must stop
    decrementing swa refs at its own boundary node and leave the shallow reader's window pinned."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = wp + 2
    ids = _chain(sess, n)

    deep = sess.do_lock(ids)                              # window = the trailing wp nodes
    shallow = sess.do_lock(ids[: wp * P])                 # window = the leading wp nodes
    assert deep is not None and shallow is not None
    assert deep[0].uuid is not None and shallow[0].uuid is not None
    assert deep[0].uuid != shallow[0].uuid                # distinct boundary nodes, distinct ids

    swa_pinned = set(range(n - wp, n)) | set(range(wp))
    assert _sizes(sess)["swa_protected"] == len(swa_pinned) * P
    assert _sizes(sess)["full_protected"] == n * P

    sess.do_unlock(deep)
    sess.check()
    assert _sizes(sess)["swa_protected"] == wp * P        # exactly the shallow reader's window
    assert _sizes(sess)["full_protected"] == wp * P

    sess.do_unlock(shallow)
    sess.check()
    assert _sizes(sess)["swa_protected"] == 0 and _sizes(sess)["full_protected"] == 0


def test_dec_lock_skip_swa_strands_the_window_lock(sess):
    """``skip_swa=True`` has no production caller and this pins why: on its own it drops the full
    ref while keeping the swa ref, so the class's own ``full_ref >= swa_ref`` assertion breaks."""
    P, wp = sess.P, _window_pages(sess.spec)
    ids = _chain(sess, wp + 1)
    held = sess.do_lock(ids)
    assert held is not None

    sess.do_unlock(held, skip_swa=True)

    assert _sizes(sess)["full_protected"] == 0
    assert _sizes(sess)["swa_protected"] == wp * P        # the window tier is stranded
    check_counter_model(sess.ad)                          # bookkeeping itself stays consistent ...
    check_model_agreement(sess.ad, sess.model)
    with pytest.raises(InvariantViolation) as err:        # ... but the class invariant is broken
        check_class_integrity(sess.ad)
    assert err.value.tag == "check_integrity"


# --------------------------------------------------------------------------- eviction: full tier
def test_evict_full_takes_unlocked_leaves_only(sess):
    """The full pass evicts LEAVES only -- an internal node's KV is a prefix dependency -- and never
    touches a locked path, even when the locked/internal nodes are the least recently used."""
    P = sess.P
    ids = _chain(sess, 3)
    held = sess.do_lock(ids)
    assert held is not None
    freed_before = set(sess.kv.free)

    sess.do_evict_full(10 ** 6)                           # the only leaf is locked -> nothing goes
    sess.check()
    assert sess.kv.free == freed_before
    assert len(_groups(sess)) == 3

    sess.do_unlock(held)
    sess.do_match(ids)                                    # N0 (root-most) is now the oldest node
    sess.do_evict_full(P)                                 # ... yet the leaf N2 must be the victim
    sess.check()
    assert {g.end for g in _groups(sess)} == {tuple(ids[:P]), tuple(ids[: 2 * P])}
    assert _sizes(sess) == {"full_evictable": 2 * P, "full_protected": 0,
                            "swa_evictable": 2 * P, "swa_protected": 0}


def test_evict_full_cascades_through_exposed_tombstone_leaves(sess):
    """Combined pressure: the swa pass tombstones the internals, then the full pass pops the leaf
    and must eagerly reclaim every tombstone leaf it exposes upward (a tombstone leaf holds full KV
    that nothing can ever match through) -- without re-pushing a cascade-consumed node."""
    P = sess.P
    ids = _chain(sess, 3)
    sess.do_match(ids)
    sess.do_evict_second(2 * P)                           # tombstone N0 then N1, both internal
    sess.check()
    assert _lengths(sess, True) == [P, P]

    sess.do_evict_full(P)                                 # pop leaf N2 -> cascade N1 -> N0
    sess.check()

    assert _ev(sess)["evict_full.cascade"] == 2
    assert _groups(sess) == []
    assert _sizes(sess)["full_evictable"] == 0 and _sizes(sess)["swa_evictable"] == 0
    assert sess.kv.in_use() == set()                      # nothing leaked on the way out


# --------------------------------------------------------------------------- eviction: swa tier
def test_evict_swa_tombstones_an_internal_node_in_place(sess):
    """The swa pass may take internal nodes: it frees only the swa currency and leaves the full KV
    (and the children) in place, so the prefix stays matchable for the full tier."""
    P, wp = sess.P, _window_pages(sess.spec)
    ids = _chain(sess, wp + 1)
    sess.do_match(ids)
    head_slots = list(_node(sess, ids, 1).slots)
    freed_before = set(sess.kv.free)

    sess.do_evict_second(1)
    sess.check()

    assert _ev(sess)["evict_swa.tombstone_in_place"] == 1
    assert _tombstoned(sess) == {tuple(ids[:P])}
    assert sess.kv.free == freed_before                   # no FULL slot was released
    assert _sizes(sess)["swa_evictable"] == wp * P

    got, _ = sess.do_match(ids)                           # the live tail covers the window ...
    assert got.cached_len == (wp + 1) * P
    assert got.indices[:P] == head_slots                  # ... so the tombstone's full KV is served


def test_evict_swa_frees_a_free_leaf_and_cascades(sess):
    """The swa pass's leaf edge: an unlocked leaf has nothing to keep its full KV alive for, so both
    pools are freed and the exposed tombstone ancestor cascades away with it."""
    P = sess.P
    ids = _chain(sess, 2)
    sess.do_match(ids)

    sess.do_evict_second(2 * P)                           # tombstone N0, then free leaf N1
    sess.check()

    assert _ev(sess)["evict_swa.tombstone_in_place"] == 1
    assert _ev(sess)["evict_swa.leaf_free"] == 1
    assert _ev(sess)["evict_swa.cascade"] == 1
    assert _groups(sess) == []
    assert sess.kv.in_use() == set()


def test_evict_swa_victim_order_follows_match_recency(sess):
    """Victim order is match recency, not insert order: ``match_prefix`` re-stamps the matched path
    with strictly decreasing timestamps toward the root, so the swa pass reclaims the root-most
    STALE node first -- never the freshly matched head."""
    P = sess.P
    ids = _chain(sess, 4)                                 # insert-order recency: N0 oldest
    sess.do_match(ids[: 2 * P])                           # re-stamp N0, N1 as the most recent

    sess.do_evict_second(1)
    sess.check()

    assert _tombstoned(sess) == {tuple(ids[: 3 * P])}     # N2, not N0


def test_locked_window_survives_both_evictors(sess):
    """A held lock is honoured by both passes: the full pass finds no unlocked leaf, and the swa
    pass may only tombstone the root-ward nodes outside the pinned window."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = wp + 2
    ids = _chain(sess, n)
    held = sess.do_lock(ids)
    assert held is not None
    freed_before = set(sess.kv.free)

    sess.do_evict_full(10 ** 6)
    sess.do_evict_second(10 ** 6)
    sess.check()

    assert sess.kv.free == freed_before                   # not one full slot was released
    assert _tombstoned(sess) == {tuple(ids[:P]), tuple(ids[: 2 * P])}
    assert _lengths(sess, False) == [P] * wp              # the pinned window stayed live
    assert _sizes(sess)["swa_protected"] == wp * P

    sess.do_unlock(held)
    sess.check()


def test_swa_pressure_drains_every_slot_exactly_once(sess):
    """Repeated ``evict_swa`` converges (each pass either tombstones or unlinks) and, once the swa
    currency is exhausted, every full slot ever handed out has come back exactly once."""
    P = sess.P
    ids = _chain(sess, 6)
    sess.do_match(ids)

    guard = 0
    while _sizes(sess)["swa_evictable"]:
        sess.do_evict_second(P)
        guard += 1
        assert guard < 40, "evict_swa is not converging"
    while _sizes(sess)["full_evictable"]:
        sess.do_evict_full(P)
        guard += 1
        assert guard < 80, "evict_full is not converging"
    sess.check()

    assert _groups(sess) == []
    assert sess.kv.in_use() == set()                      # no leak, no double free (ledger-checked)


# --------------------------------------------------------------------------- bounded encoder replay
def test_match_prefix_full_serves_the_whole_depth_with_the_windowed_boundary(swa_spec):
    """``match_prefix_full`` (the bounded-encoder-replay match): the full-tier depth and its page
    indices WITHOUT the windowed truncation, plus the windowed-safe boundary the truncating match
    would have returned -- the two agree exactly when the path is fully live."""
    import torch

    from freetoken.kvcache.swa_radix_cache import SWARadixCache

    P, wp, W = swa_spec.page_size, _window_pages(swa_spec), swa_spec.window
    dev = torch.device("cpu")
    ids = torch.tensor(_seq(swa_spec, wp + 1), dtype=torch.int64)
    slots = torch.arange((wp + 1) * P, dtype=torch.int32)

    live = SWARadixCache(dev, P, W)
    live.insert(ids, slots)
    m = live.match_prefix_full(ids)
    assert m.full_len == len(ids) and m.windowed_len == m.full_len
    assert m.kv_indices.tolist() == slots.tolist()
    t = live.match_prefix(ids)
    assert (t.cached_len, t.kv_indices.tolist()) == (m.windowed_len, m.kv_indices.tolist())

    if wp == 1:
        pytest.skip("page == window: a live run shorter than the window is below one whole page, "
                    "which the suffix clamp never leaves -- the DSV4 shape is covered end-to-end "
                    "by the reference-parity replay test")
    evicted = SWARadixCache(dev, P, W)
    evicted.insert(ids, slots, swa_evicted_seqlen=wp * P)   # one live page on top of a tombstoned head
    m = evicted.match_prefix_full(ids)
    assert m.full_len == len(ids) and m.windowed_len == 0    # the run is short -> nothing windowed-safe
    assert m.kv_indices.tolist() == slots.tolist()           # ... yet the FULL indices are all served
    assert evicted.match_prefix(ids).cached_len == m.windowed_len


def test_insert_revives_in_place_when_the_request_reused_the_trees_pages(swa_spec):
    """The bounded-encoder-replay commit: the request's row for the replayed span IS the tree's own
    value (the window KV was rewritten in place on those pages), so insert must revive a tombstone
    there WITHOUT adopting or freeing anything -- freeing would hand the tree's live pages to the
    next allocation, and adopting is a no-op."""
    import torch

    from freetoken.kvcache.swa_radix_cache import SWARadixCache

    P, W = swa_spec.page_size, swa_spec.window
    dev = torch.device("cpu")
    cache = SWARadixCache(dev, P, W)
    ids = torch.tensor(_seq(swa_spec, 3), dtype=torch.int64)
    slots = torch.arange(3 * P, dtype=torch.int32)
    cache.insert(ids, slots, swa_evicted_seqlen=P)          # page 0 tombstone, pages 1-2 live
    assert cache.swa_evictable_size == 2 * P

    # the fresh region starts at the replay start (inside the tombstone); the row == the tree.
    # The flag marks the commit as a replay's (the cache never probes for the alias otherwise).
    matched, freed = cache.insert(ids, slots, swa_evicted_seqlen=0, update_kv_after_len=0,
                                  replay_reused_tree_pages=True)
    assert matched == 3 * P and freed.numel() == 0    # the tree's own slots are never "dups"
    assert cache.swa_evictable_size == 3 * P         # the tombstone revived in place
    cache.check_integrity()
    m = cache.match_prefix(ids)                      # the windowed boundary now reaches the full depth
    assert m.cached_len == 3 * P and m.kv_indices.tolist() == slots.tolist()


# --------------------------------------------------------------------------- retention
def test_finish_time_restamp_soft_pins_the_prompt_window(swa_spec, make_session):
    """Decode never re-matches the prompt, so at finish the prompt window carries the stamp of the
    prefill-boundary commit and is the first swa victim -- losing the prefix a follow-up turn would
    have cut at.  A finish-time re-match soft-pins it and sends the pressure to the idle tail."""
    P, wp = swa_spec.page_size, _window_pages(swa_spec)
    prompt_pages = wp

    def run(restamp: bool) -> int:
        s = make_session(swa_spec)
        ids = _seq(swa_spec, prompt_pages + 2)
        prompt = ids[: prompt_pages * P]
        s.do_insert(prompt)                               # prefill-boundary commit
        s.do_request(ids, prompt_pages)                   # match -> lock -> finish insert -> unlock
        if restamp:
            s.do_match(prompt)                            # the finish-time soft pin
        s.do_evict_second(1)
        s.check()
        got, _ = s.do_match(prompt)
        return got.cached_len

    assert run(restamp=False) == 0
    assert run(restamp=True) == prompt_pages * P


def test_trim_head_swa_reclaims_the_head_and_keeps_the_window(sess):
    """Finish-time retention: only the trailing window has to stay swa-live for a next-turn cut, so
    the head below ``keep_from`` is tombstoned eagerly instead of lingering as LRU litter -- its
    full KV stays and keeps being served."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = wp + 2
    ids = _chain(sess, n)
    head_slots = list(_node(sess, ids, 1).slots) + list(_node(sess, ids, 2).slots)

    sess.do_trim(ids, 2 * P)                              # keep_from = n - wp pages
    sess.check()

    assert _ev(sess)["trim.tombstone"] == 2
    assert _tombstoned(sess) == {tuple(ids[:P]), tuple(ids[: 2 * P])}
    assert _sizes(sess)["swa_evictable"] == wp * P
    assert _sizes(sess)["full_evictable"] == n * P        # not one full token was given up

    got, _ = sess.do_match(ids)                           # the retained window still covers W ...
    assert got.cached_len == n * P
    assert got.indices[: 2 * P] == head_slots             # ... and the trimmed head's KV is served


def test_trim_head_swa_skips_locked_leaf_and_tombstoned_nodes(sess):
    """A concurrent reader still swa-locking the head region must not lose its window, a leaf must
    never become a tombstone, and an already tombstoned node must not be reported freed twice."""
    P, wp = sess.P, _window_pages(sess.spec)
    n = wp + 2
    ids = _chain(sess, n)
    held = sess.do_lock(ids)
    assert held is not None

    sess.do_trim(ids, 0)                                  # no retention boundary -> no-op
    sess.check()
    assert _ev(sess)["trim.tombstone"] == 0

    sess.do_trim(ids, n * P)                              # ask for the whole path
    sess.check()
    assert _ev(sess)["trim.tombstone"] == 2               # only the unlocked internals below it
    assert _tombstoned(sess) == {tuple(ids[:P]), tuple(ids[: 2 * P])}
    assert not any(g.tomb for g in _groups(sess) if sess.model.trie.is_leaf(g))

    sess.do_trim(ids, n * P)                              # idempotent: the tombstones are skipped
    sess.check()
    assert _ev(sess)["trim.tombstone"] == 2

    sess.do_unlock(held)
    sess.check()

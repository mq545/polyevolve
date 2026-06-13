"""Event-grouped coherence normalization tests."""

from __future__ import annotations

from polyevolve.bench.coherence import deoverround, normalize_by_group, normalize_sum_to_one


def test_sum_to_one_fixes_lula_flavio():
    out = normalize_sum_to_one([0.86, 0.99])  # the incoherent bug
    assert abs(sum(out) - 1.0) < 1e-9
    assert out[1] > out[0]  # Flavio was higher, stays higher, but now coherent


def test_deoverround_scales_down_only():
    assert abs(sum(deoverround([0.86, 0.99])) - 1.0) < 1e-9  # over-round removed
    assert deoverround([0.2, 0.3]) == [0.2, 0.3]  # partial group untouched


def test_normalize_by_group_respects_groups_and_order():
    keys = ["E1", "E1", "E2", None]
    ps = [0.8, 0.9, 0.4, 0.95]  # E1 overrounds; E2 fine; lone market untouched
    out = normalize_by_group(keys, ps, mode="deoverround")
    assert abs((out[0] + out[1]) - 1.0) < 1e-9  # E1 de-overrounded
    assert out[2] == 0.4 and out[3] == 0.95  # E2 + lone untouched


def test_lone_market_and_zero_sum_safe():
    assert normalize_by_group(["A"], [0.7]) == [0.7]  # single-market group
    assert normalize_sum_to_one([0.0, 0.0]) == [0.0, 0.0]  # zero-sum guard


def test_brier_improves_on_overconfident_group():
    # one winner (idx 0). genome overconfident on everyone -> normalize sharpens.
    ps = [0.9, 0.8, 0.7]
    out = normalize_sum_to_one(ps)
    y = [1.0, 0.0, 0.0]
    braw = sum((p - t) ** 2 for p, t in zip(ps, y, strict=True)) / 3
    bnorm = sum((p - t) ** 2 for p, t in zip(out, y, strict=True)) / 3
    assert bnorm < braw

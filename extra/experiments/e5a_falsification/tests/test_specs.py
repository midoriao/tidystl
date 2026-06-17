import pytest
import specs


def test_registry_pairs_present():
    for name in ["m1_speed", "m2_mass_spring", "m3_coupled"]:
        b = specs.BENCHMARKS[name]
        assert b.model.name == name
        assert b.hero_threshold in b.thresholds
        specs.assert_principled(b.spec())


def test_ladder_specs_all_principled_and_on_grid():
    for b in specs.BENCHMARKS.values():
        for thr in b.thresholds:
            spec = b.spec(thr)
            specs.assert_principled(spec)
            specs.assert_on_grid(spec, b.model.dt)


def test_rejects_top_level_and():
    with pytest.raises(ValueError, match="top-level"):
        specs.assert_principled("(x >= 0.9) and (x <= 2.0)")


def test_rejects_equality_predicate():
    with pytest.raises(ValueError, match="equality"):
        specs.assert_principled("G[0.0,2.0](x == 0.9)")


def test_rejects_off_grid_bound():
    with pytest.raises(ValueError, match="aligned"):
        specs.assert_on_grid("G[0.3,5.0](x <= 0.4)", 0.5)

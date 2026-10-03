import pytest

from quantum_measurement.parallel.run_profiles import (
    LEGACY_PROFILE,
    TANGENT_GUARDED_PROFILE,
    resolve_tangent_schedule,
)


def test_legacy_profile_preserves_supplied_schedule():
    schedule = resolve_tangent_schedule(
        gamma=4.0,
        J=1.0,
        dt=0.00125,
        n_burnin=400,
        n_samples=3200,
    )

    assert schedule.profile_name == LEGACY_PROFILE
    assert schedule.dt == 0.00125
    assert schedule.n_burnin == 400
    assert schedule.n_samples == 3200
    assert schedule.n_steps == 3600
    assert schedule.decision_state == "baseline"
    assert schedule.parent_fingerprint is None


def test_guarded_profile_refines_future_batch_and_preserves_duration():
    baseline = resolve_tangent_schedule(
        gamma=4.0,
        J=1.0,
        dt=0.00125,
        n_burnin=400,
        n_samples=3200,
        profile_name=TANGENT_GUARDED_PROFILE,
    )
    refined = resolve_tangent_schedule(
        gamma=4.0,
        J=1.0,
        dt=0.00125,
        n_burnin=400,
        n_samples=3200,
        profile_name=TANGENT_GUARDED_PROFILE,
        diagnostics={
            "max_second_tangent_norm": 2000.0,
            "d2q_sem": 12.0,
        },
    )

    assert refined.dt == 0.000625
    assert refined.n_burnin * refined.dt == pytest.approx(0.5)
    assert refined.n_samples * refined.dt == pytest.approx(4.0)
    assert refined.decision_state == "refined_dt"
    assert refined.parent_fingerprint == baseline.fingerprint
    assert refined.fingerprint != baseline.fingerprint


def test_guarded_profile_shortens_severe_horizon():
    refined = resolve_tangent_schedule(
        gamma=4.0,
        J=1.0,
        dt=0.00125,
        n_burnin=400,
        n_samples=3200,
        profile_name=TANGENT_GUARDED_PROFILE,
        diagnostics={"max_second_tangent_norm": 20000.0, "d2q_sem": 0.0},
    )

    assert refined.decision_state == "shortened_horizon"
    assert refined.sampling_time == pytest.approx(2.0)


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_tangent_schedule(
            gamma=1.0,
            J=1.0,
            dt=0.005,
            n_burnin=1,
            n_samples=2,
            profile_name="not-a-profile",
        )

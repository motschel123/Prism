import jax
import jax.numpy as jnp
from jax import random
import x_xy

"""
Calculates an amplifying-function, which can be used to decrease the values of another array with the same length.
An array of shape (time / sampling rate) will be returned, containing values between 0 and 1.
The covariance values are relative to the length of the array that will be created.
"""


def motion_amplifier(
        time,
        sampling_rate,
        key_rigid_phases,
        n_rigid_phases=3,
        rigid_duration_cov=jnp.array([0.02] * 3),
        transition_cov=jnp.array([0.1] * 3)
) -> jnp.ndarray:
    assert rigid_duration_cov.shape == (
        n_rigid_phases,) == transition_cov.shape, "motion_amplifier: There must be a variance for each rigid phase!"
    n_frames = int(time / sampling_rate)
    key_rigid_means, key_rigid_variances, key_slope_down_variances, key_slope_up_variances = random.split(
        key_rigid_phases, 4)

    # Calculate center points of rigid phases
    means = jnp.sort(random.uniform(key_rigid_means, shape=(
        n_rigid_phases, 1), minval=0, maxval=n_frames).T)

    # Calculate durations, which is twice the rigid distance from the center points for each rigid phase.
    rigid_distances = jnp.abs(random.multivariate_normal(key_rigid_variances, mean=jnp.zeros_like(
        means), cov=jnp.diag((rigid_duration_cov * n_frames)**2)))

    # Calculate transition durations
    transition_slowdown_durations = jnp.abs(random.multivariate_normal(
        key_slope_down_variances, mean=jnp.zeros_like(means), cov=jnp.diag((transition_cov * n_frames)**2)))
    transition_speedup_durations = jnp.abs(random.multivariate_normal(
        key_slope_up_variances, mean=jnp.zeros_like(means), cov=jnp.diag((transition_cov * n_frames)**2)))

    # Phase start and end points
    rigid_starts = (means - rigid_distances).astype(int).flatten()
    rigid_ends = (means + rigid_distances).astype(int).flatten()
    starts_slowing = (means - rigid_distances -
                      transition_slowdown_durations).astype(int).flatten()
    ends_moving = (means + rigid_distances +
                   transition_speedup_durations).astype(int).flatten()

    # Create masks
    def create_mask(start, end):
        nonlocal n_frames
        return jnp.where(jnp.arange(n_frames) < start, 1, 0) + jnp.where(jnp.arange(n_frames) >= end, 1, 0)

    mask = jax.vmap(create_mask)
    rigid_mask = jnp.prod(mask(rigid_starts, rigid_ends), axis=0)
    slowdown_masks = mask(starts_slowing, rigid_starts).astype(float)
    speedup_masks = mask(rigid_ends, ends_moving).astype(float)

    def linsp(mask, start, end, begin_val, carry_fun):
        range = end - start
        def true_fun(carry, x): return (carry_fun(carry, range), 1 - carry)
        def false_fun(carry, x): return (carry, x)
        def f(carry, x): return jax.lax.cond(
            x == 0, true_fun, false_fun, *(carry, x))
        return jax.lax.scan(f, begin_val, mask)[1]

    linsp_desc = jax.vmap(lambda m, s1, s2: linsp(
        m, s1, s2, 0.0, lambda carry, range: carry + 1/range))
    slowdown_mask = jnp.prod(linsp_desc(
        slowdown_masks, starts_slowing, rigid_starts), axis=0)

    linsp_asc = jax.vmap(lambda m, s1, s2: linsp(
        m, s1, s2, 1.0, lambda carry, range: carry - 1/range))
    speedup_mask = jnp.prod(
        linsp_asc(speedup_masks, rigid_ends, ends_moving), axis=0)

    return jnp.min(jnp.stack([rigid_mask, slowdown_mask, speedup_mask]), axis=0)


def random_angles_with_rigid_phases_over_time(
    key_t,
    key_ang,
    T,
    Ts,
    key_rigid_phases,
    n_rigid_phases=3,
    rigid_duration_cov=jnp.array([0.02] * 3),
    transition_cov=jnp.array([0.1] * 3),
    ANG_0=0.0,
    dang_min=0.01,
    dang_max=0.05,
    t_min=0.1,
    t_max=0.5,
    randomized_interpolation=False,
    range_of_motion=False,
    range_of_motion_method="uniform"
):
    mask = motion_amplifier(T, Ts, key_rigid_phases,
                            n_rigid_phases, rigid_duration_cov, transition_cov)

    qs = x_xy.algorithms.random_angle_over_time(
        key_t,
        key_ang,
        ANG_0,
        dang_min,
        dang_max,
        t_min,
        t_max,
        T,
        Ts,
        randomized_interpolation,
        range_of_motion,
        range_of_motion_method
    )

    # derivate qs
    qs_diff = jnp.diff(qs, axis=0)

    # mulitply with motion amplifier
    qs_diff = qs_diff * mask[:-1]

    # integrate qs_diff
    qs_rigid_phases = jnp.concatenate((qs[0:1], jnp.cumsum(qs_diff, axis=0)))
    return qs_rigid_phases

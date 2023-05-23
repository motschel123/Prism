import x_xy
from x_xy import render
from x_xy.algorithms import sensors

import jax
from jax import random
import jax.numpy as jnp

# Load system from xml
sys = x_xy.io.load_sys_from_xml("three_seg_chain.xml")

frames = 500


def random_angles_with_rigid_phases_over_time(
    key_t,
    key_ang,
    T,
    Ts,
    key_rigid_phases,
    n_rigid_phases=3,
    rigid_duration_cov=jnp.array([0.2] * 3),
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
    assert rigid_duration_cov.shape == (
        n_rigid_phases,) == transition_cov.shape, "motion_amplifier: There must be a variance for each rigid phase!"
    n_frames = int(T / Ts)
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

    # Create result array
    result = jnp.ones(n_frames)  # todo: maybe call random generator here
    transition_starts_slowing = (
        means - rigid_distances - transition_slowdown_durations).astype(int)
    rigid_starts = (means - rigid_distances).astype(int)
    rigid_ends = (means + rigid_distances).astype(int)
    transition_ends_moving = (
        means + rigid_distances + transition_speedup_durations).astype(int)

    phases = jnp.stack([transition_starts_slowing, rigid_starts,
                       rigid_ends, transition_ends_moving])[:, 0].T

    # Assure that all phases are in boundaries
    phases = jnp.where(phases < 0, 0, phases)
    phases = jnp.where(phases >= n_frames, n_frames-1, phases)

    for i in range(phases.shape[0]):
        result = result.at[phases[i, 0]:phases[i, 1]].set(
            jnp.linspace(1, 0, phases[i, 1] - phases[i, 0] + 2)[1:-1])
        result = result.at[phases[i, 1]:phases[i, 2]].set(0)
        result = result.at[phases[i, 2]:phases[i, 3]].set(
            jnp.linspace(0, 1, phases[i, 3] - phases[i, 2] + 2)[1:-1])

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
    qs_diff = qs_diff * result[:-1]

    # integrate qs_diff
    qs_rigid_phases = jnp.cumsum(qs_diff, axis=0)
    return qs_rigid_phases


def raot(s1, s2, s3):
    return x_xy.algorithms.random_angle_over_time(
        key_t=jax.random.PRNGKey(s1),
        key_ang=jax.random.PRNGKey(s2),
        ANG_0=0.0,
        dang_min=0.01,
        dang_max=0.3,
        t_min=0.1,
        t_max=0.5,
        T=frames/5,
        Ts=0.1,
    )


def rigird_raot(s1, s2, s3):
    return random_angles_with_rigid_phases_over_time(
        key_t=jax.random.PRNGKey(s1),
        key_ang=jax.random.PRNGKey(s2),
        key_rigid_phases=jax.random.PRNGKey(s3),
        ANG_0=0.0,
        dang_min=0.01,
        dang_max=0.3,
        t_min=0.1,
        t_max=0.5,
        T=frames/5,
        Ts=0.1,
    )


# Generate random angles
qs = jnp.array((rigird_raot(0, 1, 2), rigird_raot(
    2, 3, 4), rigird_raot(4, 5, 6))).T

print(qs.shape)
forward_kinematics_transforms_jit = jax.jit(
    lambda qs: x_xy.algorithms.forward_kinematics_transforms(sys, qs)[0])


x, _ = jax.jit(jax.vmap(
    x_xy.algorithms.forward_kinematics_transforms, in_axes=(None, 0)))(sys, qs)


'''
from x_xy.algorithms import dynamics
# Use physics instead of ao
state = x_xy.base.State.create(sys)

# Use jit for faster processing
step_fn = jax.jit(dynamics.step)

xs = []
for _ in range((int)(round(10 / sys.dt))):
    state = step_fn(sys, state, jnp.zeros_like(state.qd))
    xs.append(state.x)

x = xs[0].batch(*xs[1:])
'''

scene = render.VispyScene(sys.geoms)
render.animate("animation", scene, x, sys.dt, fmt="mp4")

imu = sensors.imu(x, sys.gravity, sys.dt)

idx_s = sys.name_to_idx('upper'), sys.name_to_idx(
    'middle'), sys.name_to_idx('lower')

# print(imu['gyr'][:20])

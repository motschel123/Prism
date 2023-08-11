import x_xy
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass


# Define System XMLs
three_seg_rigid = r"""
<x_xy model="three_seg_rigid">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <geom type="box" mass="10" pos="0.1 0 0" dim="0.2 0.05 0.05" vispy_color="red"/>
            <body name="seg1" joint="rsry" pos="0 0 0">
                <geom type="box" mass="10" pos="-0.1 0 0" dim="-0.2 0.05 0.05" vispy_color="yellow"/>
                <body name="imu1" pos="-0.1 0.0 0.03" joint="frozen">
                    <geom type="box" mass="2" pos="0 0 0" dim="0.05 0.01 0.01" vispy_color="lightgreen"/>
                </body>
            </body>
            <body name="seg3" joint="rsrz" pos="0.2 0 0">
                <geom type="box" mass="10" pos="0.1 0 0" dim="0.2 0.05 0.05" vispy_color="blue"/>
                <body name="imu2" joint="frozen" pos="0.1 0.0 0.03">
                    <geom type="box" mass="2" pos="0 0 0" dim="0.05 0.01 0.01" vispy_color="lightgreen"/>
                </body>
            </body>
        </body>
    </worldbody>
    <defaults>
        <geom vispy_edge_color="black" vispy_color="1 0.8 0.7 1"/>
    </defaults>
</x_xy>
"""

dustin_exp_xml_seg1 = r"""
<x_xy model="dustin_exp">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg1" joint="free">
            <geom type="box" mass="10" pos="1 0 0" dim="1 0.25 0.2"/>
            <body name="seg2" joint="ry">
                <geom type="box" mass="10" pos="1 0 0" dim="1 0.25 0.2"/>
                <body name="seg3" joint="rz"></body>
                    <geom type="box" mass="10" pos="1 0 0" dim="1 0.25 0.2"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""


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
    assert rigid_duration_cov.shape == (n_rigid_phases,) == transition_cov.shape, "motion_amplifier: There must be a variance for each rigid phase!"
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
        key_t=key_t,
        key_ang=key_ang,
        ANG_0=ANG_0,
        dang_min=dang_min,
        dang_max=dang_max,
        delta_ang_min=0.0,
        delta_ang_max=2 * jnp.pi,
        t_min=t_min,
        t_max=t_max,
        T=T,
        Ts=Ts,
        randomized_interpolation=randomized_interpolation,
        range_of_motion=range_of_motion,
        range_of_motion_method=range_of_motion_method
    )

    # derivate qs
    qs_diff = jnp.diff(qs, axis=0)

    # mulitply with motion amplifier
    qs_diff = qs_diff * mask[:-1]

    # integrate qs_diff
    qs_rigid_phases = jnp.concatenate((qs[0:1], jnp.cumsum(qs_diff, axis=0)))
    return qs_rigid_phases


# Declatarion of Extended Config dataclass used for storing additional parameters
@dataclass
class ExtendedConfig(x_xy.algorithms.RCMG_Config):
    n_rigid_phases : int = 3
    cov_rigid_durations : jax.Array = jnp.array([0.02] * n_rigid_phases)
    cov_transitions : jax.Array = jnp.array([0.1] * n_rigid_phases)
    
    def __post_init__(self):
        assert self.cov_rigid_durations.shape == self.cov_transitions.shape



def define_joints():

    def _draw_sometimes_rigid(
            config: ExtendedConfig, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
    ) -> jax.Array:
        key_t, key_rigid_phases = jax.random.split(key_t)
        return random_angles_with_rigid_phases_over_time(
            key_t=key_t,
            key_ang=key_value,
            T=config.T,
            Ts=config.Ts,
            key_rigid_phases=key_rigid_phases,
            n_rigid_phases=config.n_rigid_phases,
            rigid_duration_cov=config.cov_rigid_durations,
            transition_cov=config.cov_transitions,
            ANG_0=0,
            dang_min=config.dang_min,
            dang_max=config.dang_max,
            t_min=config.t_min,
            t_max=config.t_max,
            randomized_interpolation=config.randomized_interpolation_angle, # ???????????????????????
            range_of_motion=config.range_of_motion_hinge,
            range_of_motion_method=config.range_of_motion_hinge_method
        )

    def _rxyz_transform(q, _, axis):
        q = jnp.squeeze(q)
        rot = x_xy.maths.quat_rot_axis(axis, q)
        return x_xy.base.Transform.create(rot=rot)

    rsrx_joint = x_xy.algorithms.JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])), [None], rcmg_draw_fn=_draw_sometimes_rigid
    )
    rsry_joint = x_xy.algorithms.JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0, 1.0, 0])), [None], rcmg_draw_fn=_draw_sometimes_rigid
    )
    rsrz_joint = x_xy.algorithms.JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0, 0, 1.0])), [None], rcmg_draw_fn=_draw_sometimes_rigid
    )
    x_xy.algorithms.register_new_joint_type("rsrx", rsrx_joint, 1)
    x_xy.algorithms.register_new_joint_type("rsry", rsry_joint, 1)
    x_xy.algorithms.register_new_joint_type("rsrz", rsrz_joint, 1)


def create_sys():
    try:
        define_joints()
    except AssertionError:
        pass  # Joints already seem to be defined 
    return x_xy.io.load_sys_from_str(three_seg_rigid)


def get_dustin_sys():
    return x_xy.io.load_sys_from_str(dustin_exp_xml_seg1)


def finalize_fn_imu_data(key, q, x, sys):
    imu_seg_attachment = {"imu1": "seg1", "imu2": "seg3"}

    X = {}
    for imu, seg in imu_seg_attachment.items():
        key, consume = jax.random.split(key)
        X[seg] = x_xy.algorithms.imu(
            x.take(sys.name_to_idx(imu), 1), sys.gravity, sys.dt, consume, True
        )
    return X


def finalize_fn_rel_pose_data(key, _, x, sys):
    # Defines what the network should predict
    dustin_sys = x_xy.io.load_sys_from_str(dustin_exp_xml_seg1)
    y = x_xy.algorithms.rel_pose(sys_scan=dustin_sys, xs=x, sys_xs=sys)
    return y


def finalize_fn(key, q, x, sys):
    X = finalize_fn_imu_data(key, q, x, sys)
    y = finalize_fn_rel_pose_data(key, q, x, sys)
    return X, y, x


def generate_data(sys, config : ExtendedConfig):
    generator = x_xy.algorithms.build_generator(sys, config, finalize_fn=finalize_fn)
    # we can even batch together multiple generators
    # generator = algorithms.batch_generator([generator, generator], [32, 16])
    seed = jax.random.PRNGKey(1,)
    X, y, xs = generator(seed)
    return X, y, xs
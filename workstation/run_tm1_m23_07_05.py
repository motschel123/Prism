import jax
import jax.numpy as jnp
import x_xy
from x_xy import maths, base
from random_angle import random_angles_with_rigid_phases_over_time
from dataclasses import dataclass
from neural_networks.logging import NeptuneLogger
from neural_networks.rnno import dustin_exp_xml, rnno_v2, train

three_seg_seg2 = r"""
<x_xy model="three_seg_seg2">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <body name="seg1" joint="rsry">
                <body name="imu1" joint="frozen"/>
            </body>
            <body name="seg3" joint="rsrz">
                <body name="imu2" joint="frozen"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""

def _rxyz_transform(q, _, axis):
    q = jnp.squeeze(q)
    rot = maths.quat_rot_axis(axis, q)
    return base.Transform.create(rot=rot)


@dataclass
class ExtendedConfig(x_xy.algorithms.RCMG_Config):
    n_rigid_phases : int = 3
    cov_rigid_durations : jax.Array = jnp.array([0.02] * n_rigid_phases)
    cov_transitions : jax.Array = jnp.array([0.1] * n_rigid_phases)
    
    def __post_init__(self):
        assert self.cov_rigid_durations.shape == self.cov_transitions.shape


def define_joints():
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
        randomized_interpolation=config.randomized_interpolation,
        range_of_motion=config.range_of_motion_hinge,
        range_of_motion_method=config.range_of_motion_hinge_method
    )


def draw_pos_uniform(key, pos_min, pos_max):
    key, c1, c2, c3 = jax.random.split(key, num=4)
    pos = jnp.array(
        [
            jax.random.uniform(c1, minval=pos_min[0], maxval=pos_max[0]),
            jax.random.uniform(c2, minval=pos_min[1], maxval=pos_max[1]),
            jax.random.uniform(c3, minval=pos_min[2], maxval=pos_max[2]),
        ]
    )
    return key, pos


def setup_fn_seg2(key, sys: x_xy.base.System) -> x_xy.base.System:
    def replace_pos(transforms, new_pos, name: str):
        i = sys.name_to_idx(name)
        return transforms.index_set(i, transforms[i].replace(pos=new_pos))

    ts = sys.links.transform1

    # seg1 relative to seg2
    key, pos = draw_pos_uniform(key, [-0.3, -0.02, -0.02], [-0.05, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg1")

    # imu1 relative to seg1
    key, pos = draw_pos_uniform(
        key, [-0.25, -0.05, -0.05], [-0.05, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu1")

    # seg3 relative to seg2
    key, pos = draw_pos_uniform(key, [0.05, -0.02, -0.02], [0.3, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg3")

    # imu2 relative to seg2
    key, pos = draw_pos_uniform(key, [0.05, -0.05, -0.05], [0.25, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu2")

    return sys.replace(links=sys.links.replace(transform1=ts))


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
    dustin_sys = x_xy.io.load_sys_from_str(dustin_exp_xml)
    y = x_xy.algorithms.rel_pose(dustin_sys, x, sys)
    return y


def finalize_fn(key, q, x, sys):
    X = finalize_fn_imu_data(key, q, x, sys)
    y = finalize_fn_rel_pose_data(key, q, x, sys)
    return X, y


def main():
    define_joints()
    sys = x_xy.io.load_sys_from_str(three_seg_seg2)
    configs = [(i, jnp.array([0.02] * i), jnp.array([0.1] * i)) for i in range(1, 4)] + [(i, jnp.array([0.01] * i), jnp.array([0.05] * i)) for i in range(1, 4)]
    for n_rigid_phases, cov_rigid_durations, cov_transitions in configs:
        config = ExtendedConfig(
        n_rigid_phases=n_rigid_phases, cov_rigid_durations=cov_rigid_durations, cov_transitions=cov_transitions
        )
        
        gen = x_xy.algorithms.build_generator(
            sys, config, setup_fn_seg2, finalize_fn)
        gen = x_xy.algorithms.batch_generator(gen, 80)

        rnno = rnno_v2(x_xy.io.load_sys_from_str(dustin_exp_xml))
        print(config)
        try:
            train(gen, 1500, rnno, loggers=[NeptuneLogger(name=f"Phases={n_rigid_phases}, DurationCov={cov_rigid_durations}")])
        except:
            pass


if __name__ == "__main__":
    main()

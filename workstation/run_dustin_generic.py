import jax
import jax.numpy as jnp
import x_xy
from x_xy import maths, base
from random_angle import random_angles_with_rigid_phases_over_time
from dataclasses import dataclass
from neural_networks.logging import NeptuneLogger
from neural_networks.rnno import SaveParamsTrainingLoopCallback
from neural_networks.rnno import dustin_exp_xml, rnno_v2, train
from sys import argv

three_seg_rigid = r"""
<x_xy model="three_seg_rigid">
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

three_seg_seg2 = r"""
<x_xy model="three_seg_seg2">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <body name="seg1" joint="ry">
                <body name="imu1" joint="frozen"/>
            </body>
            <body name="seg3" joint="rz">
                <body name="imu2" joint="frozen" pos="0 0 0"/>
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


def usage(): 
    print(f"""Usage: python {argv[0]} <rigid> <n_rigid_phases> <rigid_durarion_cov> <transition_cov> <name>
          \tpython {argv[0]} <normal> <name>""")
    exit()


def main():
    # Check arguments
    if len(argv) != 3:
        if len(argv) != 6:
            usage()
        if argv[1] != "rigid":
            usage()
        try:
            n_rigid_phases = int(argv[2])
            cov_rigid_durations = float(argv[3])
            cov_transitions = float(argv[4])
        except:
            usage()
        
        cov_rigid_durations = jnp.array([cov_rigid_durations] * n_rigid_phases)
        cov_transitions = jnp.array([cov_transitions] * n_rigid_phases)
        config = ExtendedConfig(
        n_rigid_phases=n_rigid_phases, cov_rigid_durations=cov_rigid_durations, cov_transitions=cov_transitions
        )
        name = argv[5]
    else:
        if (argv[1] != "normal"):
            usage()
        config = x_xy.algorithms.RCMG_Config()
        name = argv[2]
    
    # Set up system
    define_joints()
    if argv[1] == "normal":
        sys = x_xy.io.load_sys_from_str(three_seg_seg2)
    else:
        sys = x_xy.io.load_sys_from_str(three_seg_rigid)
    

    
    gen = x_xy.algorithms.build_generator(
        sys, config, setup_fn_seg2, finalize_fn)
    gen = x_xy.algorithms.batch_generator(gen, 80)

    rnno = rnno_v2(x_xy.io.load_sys_from_str(dustin_exp_xml))
    save_params = SaveParamsTrainingLoopCallback(f"/data/"<idm>"/prism_params/{name}")
    # Start training
    print(f"Starting run with config:\n{config}\n")
    
    train(gen, 1500, rnno, loggers=[NeptuneLogger(name=name)], callbacks=[save_params])



if __name__ == "__main__":
    main()

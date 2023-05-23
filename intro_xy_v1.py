from neural_networks.rnno import train, rnno_network, rnno_network_local
from x_xy.rcmg.rcmg_old_3Seg import rcmg_3Seg
from jax import random
import jax.numpy as jnp

#
# this produces a generating function that generates random motion of a three-segment chain
batchsize = 32
# check out the source code of this function; it's quite intuitive
generator = rcmg_3Seg(
    batchsize,
    randomized_interpolation=False,
    randomized_anchors=True,
    range_of_motion=True,
    range_of_motion_method="uniform",
    Ts=0.01,  # seconds
    T=60,  # seconds
    t_min=0.15,  # min time between two generated angles
    t_max=0.75,  # max time ...
    dang_min=jnp.deg2rad(0),  # minimum angular velocity in deg/s
    dang_max=jnp.deg2rad(120),  # maximum angular velocity in deg/s
    dang_min_global=jnp.deg2rad(0),
    dang_max_global=jnp.deg2rad(60),
    dpos_min=0.001,  # speed of translation
    dpos_max=0.1,
    pos_min=-2.5,
    pos_max=+2.5,
    param_ident=None,
)

seed = 1
data = generator(random.PRNGKey(seed))

X, y = data["X"], data["y"]

# where `X` and `y` have a leading batchsize of 32


# rnno_network would work too
# but let's go with RNNO_v2
network = rnno_network_local(length_of_chain=3)

generator = rcmg_3Seg(batchsize=32)

# start training
n_episodes = 1

train(generator, n_episodes, network, loggers=[])

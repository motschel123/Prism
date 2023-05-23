import jax
import x_xy
from x_xy import algorithms, render, utils
import time

def main(do_render: bool = False):
    """Tested on x_xy_v2 == 0.2.2"""
    # disable warnings
    utils.disable_jit_warn()

    sys = x_xy.io.load_example("double_pendulum")
    config = algorithms.RCMG_Config(T=20.0)
    generator = algorithms.build_generator(sys, config)
    # we can even batch together multiple generators
    generator = algorithms.batch_generator([generator, generator], [32, 16])
    seed = jax.random.PRNGKey(1,)
    qs, xs = generator(seed)

    # batchsize, timesteps, two hinge joints
    assert qs.shape == (48, 2000, 2)

    if do_render:
        # take the first trajectory from the 48
        x = xs[0]
        scene = render.VispyScene(sys.geoms)
        render.animate("rcmg_double_pendulum", scene, x, 0.01, fmt="mp4")

    # Let's try some timings

    def time_me(f, n=5):
        # don't count time for jitting
        f()
        t0 = time.time()
        for _ in range(n):
            # technically we should block
            f()
        t1 = time.time()
        print(f"Execution time: {round((t1 - t0) / n, 4)}")

    # ~3s
    # slowish even with vectorization using jax.vmap
    # without it would take even longer
    time_me(lambda: generator(seed))
    # ~10ms
    # crazy fast even on CPU
    time_me(lambda: jax.jit(generator)(seed))
    
if __name__ == "__main__":
    main(True)
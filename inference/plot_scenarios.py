import base_network

import jax.random as random
import jax.numpy as jnp

import numpy as np

import x_xy
from x_xy.subpkgs import pipeline
from neural_networks.rnno import rnno_v2

import matplotlib.pyplot as plt

from dataclasses import dataclass
from math import ceil
from sys import argv
from os import environ
from os.path import isdir, exists


# Class declarations
@dataclass
class Problem:
    n : int
    rigid_cov : jnp.array
    transition_cov : jnp.array


# Constant declarations
USER = environ["USER"]
PROBLEMS = {
    "best_run" : Problem(1, jnp.array([0.02]), jnp.array([0.1])),
    "long_rigid_phase" : Problem(1, jnp.array([0.25]), jnp.array([0.1])),
    "many_tiny_stops" : Problem(30, jnp.array([0.001] * 30), jnp.array([0.0001] * 30)),
    "some_short_stops" : Problem(5, jnp.array([0.02] * 5), jnp.array([0.005] * 5)),
    "some_rigid_phases" : Problem(3, jnp.array([0.05] * 3), jnp.array([0.01] * 3)),
    "baseline": Problem(0, jnp.array([]), jnp.array([]))
}
problem_path = f"/data/{USER}/prism_params/params"
output_path = f"/data/{USER}/prism_output"
sys = base_network.create_sys()
dustin_sys = base_network.get_dustin_sys()


# Functions
def load_pickle_params(problem_key, prngkey_num = 0):
    from neural_networks import io_params
    pickle_file = f"{problem_path}/{problem_key}_{prngkey_num}.pickle"
    try:
        params = io_params.load(pickle_file)
    except ValueError as error:
        print(error.args)
        exit()
    return params


def config_from_problem(problem : Problem):
    return base_network.ExtendedConfig(
        n_rigid_phases=problem.n,
        cov_rigid_durations=problem.rigid_cov,
        cov_transitions=problem.transition_cov
        )
    

def infer(params, X, y, xs, generate_mp4 = False, name = None):
    if generate_mp4 and name is None:
        print("infer: Missing name for the mp4-file")
        exit(1)
    
    # Run prediction
    yhat, _ = pipeline.predict(
        sys=dustin_sys,
        rnno_fn=rnno_v2,
        X=X,
        y=y,
        xs=xs,
        sys_xs=sys,
        params=params,
        plot=False,
        render=generate_mp4,
        render_prediction=generate_mp4,
        render_path=f"{output_path}/{name}_prediction.mp4"
    )
    
    if (generate_mp4):
        # Render actual data
        x_xy.render.animate(
            path=f"{output_path}/{name}_actual.mp4",
            sys=dustin_sys,
            x=xs,
            fps=25,
            show_pbar=True,
            verbose=False
        )
    return yhat

"""
Plots the prediction compared to the true values, for a set of data.
The data must be in a dict as follows:
{
    "problem_name" : (y, yhat)
}
Where y is the sequence of actual angles, whereas yhat is the prediction.

data_key will be the title of the entire plot.
"""
def create_fig_for_problems(data_key : str, data : dict):
    # Generate plot
    NCOLS = 4
    NROWS = len(data)
    fig1, ax1 = plt.subplots(NROWS, NCOLS, figsize=(16, 4.5 * NROWS))
        
    fig1.suptitle(data_key)
    means = {}
    
    for row, (problem_name, (y, yhat)) in enumerate(data.items()):
        ax1[row, 0].text(-0.5, 0.3, problem_name.replace('_', ' '), transform=ax1[row, 0].transAxes, fontweight='bold', rotation=45)
        means[problem_name] = {}
        # loop over seg2, seg3
        for j, link_name in enumerate(yhat.keys()):
            euler_angles_hat = jnp.rad2deg(x_xy.maths.quat_to_euler(yhat[link_name]))
            euler_angles = jnp.rad2deg(x_xy.maths.quat_to_euler(y[link_name]))

            # get changing axis from 'segX' string 
            n = int(link_name[3]) - 1

            # Plot angles
            elemAngles = ax1[row, j * 2]
            elemAngles.plot(euler_angles_hat[:, n], label='ŷ')
            elemAngles.plot(euler_angles[:, n], linestyle='-.', label='y')
            elemAngles.legend()
            elemAngles.set_title(f"{problem_name} {link_name}")

            # Plot delta
            elemDeltas = ax1[row, j * 2 + 1]
            ang_err = jnp.abs(euler_angles[:,n] - euler_angles_hat[:,n])
            elemDeltas.plot(ang_err, label=f"{link_name}")
            elemDeltas.plot([jnp.average(ang_err)] * len(ang_err), color='red', label=f"Avg {link_name}")
            # plot moving average of delta
            elemDeltas.plot(jnp.convolve(ang_err, np.ones(100)/100, mode='valid'), color='orange', linestyle='-.', label="MovAvg")
            elemDeltas.legend()
            elemDeltas.set_title(f"deltas {problem_name} {link_name}")
            # Save mean
            means[problem_name][link_name] = float(jnp.average(ang_err))

    # Save plot
    plt.savefig(f"{output_path}/{data_key}_tmp.pdf")
    plt.close(fig1)

    # BARGRAPH
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 4.5))

    bar_width = 0.3

    for i, (topic, subtopics) in enumerate(means.items()):
        ax2.bar([p * bar_width + i for p in range(len(subtopics.values()))], subtopics.values(), width=bar_width, label=topic)

    ax2.set_xticks([i + j * 0.3 for i in range(len(means.keys())) for j in range(2)])

    ax2.set_xticklabels([item for sublist in [list(means[p].keys()) for p in means] for item in sublist])
        # add a legend
    ax2.legend()
    plt.savefig(f"{output_path}/{data_key}_bar.pdf")
    plt.close(fig2)
    
    # Combine pdf files into one
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    merger.append(f"{output_path}/{data_key}_bar.pdf")
    merger.append(f"{output_path}/{data_key}_tmp.pdf")
    merger.write(f"{output_path}/{data_key}.pdf")
    merger.close()
    print(f"{output_path}/{data_key}.pdf")


def main():
    params = load_pickle_params("long_rigid_phase")
    problems_dict = {}

    for problem_key, problem in PROBLEMS.items():
        X, y, xs = base_network.generate_data(sys, config_from_problem(problem))
        yhat = infer(params, X, y, xs)
        problems_dict[problem_key] = (y, yhat)
    
    # Plot
    create_fig_for_problems("long_rigid_phase", problems_dict)
    
    
    

if __name__ == "__main__":
    # Read parameters
    if len(argv) > 2:
        print(f"Usage: python {argv[0]} [path to parameters]")
        exit(1)
    elif len(argv) == 2:
        problem_path = argv[1]
    # Check path
    if not isdir(problem_path):
        print(f"Error: {problem_path}: No such directory")
        print(f"Usage: python {argv[0]} [path to parameters]")
        exit(1)
    main()
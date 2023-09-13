import base_network

import jax.numpy as jnp
import jax.tree_util as jtu

import numpy as np

import pickle

import tree_utils

import x_xy
from x_xy.subpkgs import pipeline
from neural_networks.rnno import rnno_v2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import blended_transform_factory

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
        render_path=f"{output_path}/{name}_prediction.mp4",
        verbose=False
    )
    
    if (generate_mp4):
        # Render actual data
        x_xy.render.animate(
            path=f"{output_path}/{name}_actual.mp4",
            sys=sys,
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
    fig1, ax1 = plt.subplots(NROWS, NCOLS, figsize=(16, 3.5 * NROWS))
        
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
    # plt.savefig(f"{output_path}/{data_key}_tmp.pdf")
    # plt.close(fig1)

    # BARGRAPH
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 4.5))
    bar_width = 0.3

    for i, (topic, subtopics) in enumerate(means.items()):
        ax2.bar([p * bar_width + i for p in range(len(subtopics.values()))], subtopics.values(), width=bar_width, label=topic)

    ax2.set_xticks([i + j * 0.3 for i in range(len(means.keys())) for j in range(2)])
    ax2.set_xticklabels([item for sublist in [list(means[p].keys()) for p in means] for item in sublist])
    ax2.legend()
    # plt.savefig(f"{output_path}/{data_key}_bar.pdf")
    #plt.close(fig2)
    
    filename = f"{output_path}/{data_key}.pdf"
    fig2.suptitle(data_key)
    with PdfPages(filename) as pdf:
        pdf.savefig(fig2)
        pdf.savefig(fig1)
    
    return means, filename
    
    
def plot_results(results : dict[str, dict[str, float]]):
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    bar_width = 1/(len(results) * 3)

    xticks = []
    xticklabels = []
    
    transform = blended_transform_factory(axs.transData, axs.transAxes)

    axs.set_xticks([])
    for i, (key, values) in enumerate(results.items()):
        # print(range(len(values)))
        positions_a = [j + i * bar_width * 2.5 for j in range(len(values))]
        positions_b = [j + bar_width for j in positions_a]
        positions = [x for pair in zip(positions_a, positions_b) for x in pair]
        keys = [path[1].key for path, _ in jtu.tree_flatten_with_path(results[key])[0]]
        print(key, positions)
        rects = axs.bar(positions, jtu.tree_flatten(results[key])[0], bar_width, label=key)
        #print(rects.get_children()[0].get_center())
        xticks += positions
        xticklabels += keys
        axs.text(i + 0.25, -0.1, key, ha='center', transform=transform)

    axs.set_xticks(xticks)   
    axs.set_xticklabels(xticklabels)
    axs.tick_params(labelsize=7, direction="out")
    axs.legend()
    plt.savefig(f"{output_path}/results_bar.pdf")
    return f"{output_path}/results_bar.pdf"


def merge_pdfs(files : list):
    print(f"Trying to merge: {files}")
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for file in files:
        merger.append(file)
    merger.write(f"{output_path}/results_combined.pdf")
    merger.close()
    print(f"{output_path}/results_combined.pdf")


def get_inferred_data(n_batch):
    cached_infered_data_path = f"{output_path}/.cache/inferred_data_batched_{n_batch}.pickle"
    inferred_data = {}
    if exists(cached_infered_data_path):
        with open(cached_infered_data_path, 'rb') as file:
            print("Using cached inferred data...")
            return pickle.load(file)
    
    i, j, k = 0, 0, 0
    maxI, maxJ, maxK = len(PROBLEMS), len(PROBLEMS), n_batch
    for topic, _ in PROBLEMS.items():
        i += 1
        j, k = 0, 0
        params = load_pickle_params(topic)
        for problem_key, problem in PROBLEMS.items():
            j += 1
            k = 0
            name = f"{topic}_{problem_key}_{n_batch}"
            cache_file = f"{output_path}/.cache/{name}.pickle"
            if exists(cache_file):
                with open(cache_file, 'rb') as file:
                    data = pickle.load(file)
                    X, y, xs = data
                    print("Using cached data...")
            else:
                X, y, xs = base_network.generate_data(sys, config_from_problem(problem), n_batch)
                with open(cache_file, 'wb') as file:
                    pickle.dump((X, y, xs), file)
                
            """
            X: {'seg1': {
                'acc': Array(5, 6000, 3), 
                'gyr': Array(5, 6000, 3)
            }}
            """

            for generatorIndex in range(n_batch):
                k += 1
                # Un-batch data
                X_i = tree_utils.tree_slice(tree_utils.tree_indices(X, jnp.array([generatorIndex])),1)
                y_i = tree_utils.tree_slice(tree_utils.tree_indices(y, jnp.array([generatorIndex])),1)
                xs_i = tree_utils.tree_slice(tree_utils.tree_indices(xs, jnp.array([generatorIndex])),1)
                
                yhat_i = infer(params, X_i, y_i, xs_i, False, f"{topic}-{problem_key}")
                
                inferred_data[f"{topic}_{problem_key}_{generatorIndex}"] = (y_i, yhat_i)
                
                print(f"Finished Topic {i}/{maxI}, Problem {j}/{maxJ}, Data-row {k}/{maxK}")

    # Cache inferred data
    with open(cached_infered_data_path, 'wb') as file:
        pickle.dump(inferred_data, file)
     
    return inferred_data

def main():
    N_BATCH = 5
    results = {}
    filenames = []

    inferred_data = get_inferred_data(N_BATCH)

    # Plot
    for topic, _ in PROBLEMS.keys():
        
        pass
        ##sub_res, filename = create_fig_for_problems(topic, problems_dict)
    ##filenames.append(filename)
    
    ##results[topic] = sub_res
    ##print(results)

    ##res_file = plot_results(results)
    ##merge_pdfs([res_file] + filenames)

    
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
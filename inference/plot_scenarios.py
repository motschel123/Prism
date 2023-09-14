import base_network

import jax.numpy as jnp
import jax.tree_util as jtu

import numpy as np

import pickle

import tree_utils

import x_xy
from x_xy.subpkgs import pipeline
from neural_networks.rnno import rnno_v2
import neural_networks.rnno.dustin_exp.dustin_exp as dustin_exp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import blended_transform_factory

from dataclasses import dataclass
from math import ceil
from sys import argv
from os import environ
from os.path import isdir, exists

from collections import defaultdict

# Class declarations
@dataclass
class Problem:
    n : int
    rigid_cov : jnp.array
    transition_cov : jnp.array


# Constant declarations
USER = environ["USER"]
COLOR_PALETTE = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
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


def create_fig_for_problems(topic : str, data : dict, n_batch: int, problem_keys = list(PROBLEMS.keys())):
    # Generate plot
    NCOLS = 4
    NROWS = len(problem_keys)
    figs = []
        
    means = {}

    for batch_index in range(n_batch):
        print(f"Plotting batch {batch_index}/{n_batch}")
        fig1, ax1 = plt.subplots(NROWS, NCOLS, figsize=(16, 3.5 * NROWS))
        for ax in ax1.flat:
            ax.xaxis.set_label_coords(1.065, 0.011)
            ax.set_xlabel("t", horizontalalignment='right')
            #xlabel.set_position((1.065, 1.5))
            ax.set_ylabel("ang [deg]", labelpad=-4.1)
            ax.grid(True)
        figs.append(fig1)
        
        def get_ax1(row, col):
            if NROWS == 1:
                return ax1[col]
            else:
                return ax1[row, col]
        
        for row, problem_name in enumerate(problem_keys):
            dict_key = f"{topic}_{problem_name}_{batch_index}"
            (y, yhat) = data[dict_key]
            
            get_ax1(row, 0).text(-0.52, 0.3, problem_name.replace('_', ' '), transform=get_ax1(row, 0).transAxes, fontweight='bold', rotation=45)
            if batch_index == 0:
                means[problem_name] = defaultdict(lambda: 0, {})
            # loop over seg2, seg3
            for j, link_name in enumerate(yhat.keys()):
                euler_angles_hat = jnp.rad2deg(x_xy.maths.quat_to_euler(yhat[link_name]))
                euler_angles = jnp.rad2deg(x_xy.maths.quat_to_euler(y[link_name]))

                # get changing axis from 'segX' string 
                n = int(link_name[3]) - 1

                # Plot angles
                elemAngles = get_ax1(row, j * 2)
                elemAngles.plot(euler_angles_hat[:, n], label='ŷ')
                elemAngles.plot(euler_angles[:, n], linestyle='-.', label='y')
                elemAngles.legend()
                elemAngles.set_title(f"{problem_name} {link_name}")

                # Plot delta
                elemDeltas = get_ax1(row, j * 2 + 1)
                ang_err = jnp.abs(euler_angles[:,n] - euler_angles_hat[:,n])
                elemDeltas.plot(ang_err, label=f"{link_name}")
                elemDeltas.plot([jnp.average(ang_err)] * len(ang_err), color='red', label=f"Avg {link_name}")
                # plot moving average of delta
                elemDeltas.plot(jnp.convolve(ang_err, np.ones(100)/100, mode='valid'), color='orange', linestyle='-.', label="MovAvg")
                elemDeltas.legend()
                elemDeltas.set_title(f"deltas {problem_name} {link_name}")
                # Save mean
                # TODO: Check if mean exists else set to 0
                
                means[problem_name][link_name] += float(jnp.average(ang_err)) / n_batch


    # Save plot
    # plt.savefig(f"{output_path}/{data_key}_tmp.pdf")
    # plt.close(fig1)

    # BARGRAPH
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 4.5))
    ax2.set_ylabel("mae [deg]")
    bar_width = 0.3

    for i, (mean_topic, mean_subtopics) in enumerate(means.items()):
        ax2.bar([p * bar_width + i for p in range(len(mean_subtopics.values()))], mean_subtopics.values(), width=bar_width, label=mean_topic)

    ax2.set_xticks([i + j * 0.3 for i in range(len(means.keys())) for j in range(2)])
    ax2.set_xticklabels([item for sublist in [list(means[p].keys()) for p in means] for item in sublist])
    ax2.legend(loc='center right', bbox_to_anchor=(1.11, 0.5))
    # plt.savefig(f"{output_path}/{data_key}_bar.pdf")
    #plt.close(fig2)
    
    print("Saving pdf...")
    filename = f"{output_path}/{topic}.pdf"
    fig2.suptitle(topic)
    with PdfPages(filename) as pdf:
        pdf.savefig(fig2)
        for fig in figs:
            pdf.savefig(fig)
    
    plt.close('all')
    print(f"Created plot: \'{filename}\'")
    return means, filename
    
    
def plot_results(results : dict[str, dict[str, float]]):
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    bar_width = 1/(len(results) * 3)
    axs.set_ylabel("mae [deg]")

    xticks = []
    xticklabels = []
    
    transform = blended_transform_factory(axs.transData, axs.transAxes)

    axs.set_xticks([])
    for i, (key, values) in enumerate(results.items()):
        # print(range(len(values)))
        positions_a = [j + i * bar_width * 2.5 for j in range(len(values))]
        positions_b = [j + bar_width for j in positions_a]
        positions = [x for pair in zip(positions_a, positions_b) for x in pair]
        tick_positions = [(positions_a[i] + positions_b[i]) / 2 for i in range(len(positions_a))]
        
        #keys = [path[1].key for path, _ in jtu.tree_flatten_with_path(results[key])[0]]
        #keys = [s for s in PROBLEMS.keys()]
        print(key, positions)
        for j, (position, height) in enumerate(zip(positions, list(jtu.tree_flatten(results[key])[0]))):
            rects = axs.bar(position, height, bar_width, color=f'C{j//2}')
            # Display exact value above the graph # axs.text(position, 1.05, f"{height:.2f}", ha='center', transform=transform, color=f'C{i//2}') 
        #print(rects.get_children()[0].get_center())
        xticks += tick_positions
        xticklabels += [f"{i}"] * len(tick_positions)
        axs.text(i + 0.25, -0.1, key, ha='center', transform=transform)

    axs.set_xticks(xticks)   
    axs.set_xticklabels(xticklabels)
    axs.tick_params(labelsize=9, direction="out")
    # Create custom Legend
    legend_elements = []
    for i, topic in enumerate(PROBLEMS.keys()):
        legend_elements.append(mpatches.Patch(label=f"{i}: {topic}", color='none'))
    axs.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.13, 0.5), borderpad=0.2, handlelength=0, handletextpad=0, markerscale=0, prop={'size': 10})
    # Save and return
    plt.savefig(f"{output_path}/results_bar.pdf")
    return f"{output_path}/results_bar.pdf"


def merge_pdfs(files : list, name = 'results_combined.pdf'):
    print(f"Trying to merge: {files}")
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for file in files:
        merger.append(file)
    merger.write(f"{output_path}/{name}")
    merger.close()
    print(f"{output_path}/{name}")


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


def get_inferred_data_dustin_exp():
    X, y = get_dustin_exp_data()
    N_BATCH_EXP_DATA = tree_utils.tree_shape(X)
    
    cached_infered_data_path = f"{output_path}/.cache/inferred_data_dustin.pickle"
    inferred_data_dustin_exp = {}
    if exists(cached_infered_data_path):
        with open(cached_infered_data_path, 'rb') as file:
            print("Using cached inferred data dustin exp...")
            return pickle.load(file)
    
    i,j = 0,0
    maxI, maxJ = len(PROBLEMS.keys()), N_BATCH_EXP_DATA
    for topic in PROBLEMS.keys():
        i += 1
        j = 0
        params = load_pickle_params(topic)
        for batch_index in range(N_BATCH_EXP_DATA):
            j += 1
            X_i = tree_utils.tree_slice(tree_utils.tree_indices(X, jnp.array([batch_index])),1)
            y_i = tree_utils.tree_slice(tree_utils.tree_indices(y, jnp.array([batch_index])),1)
            yhat_i = infer(params, X_i, y_i, None, generate_mp4 = False, name = f"{topic}-dustin")
            inferred_data_dustin_exp[f"{topic}_dustin_{batch_index}"] = (y_i, yhat_i)
            print(f"Finished Topic {i}/{maxI}, Data-row {j}/{maxJ}")

    # Cache inferred data
    with open(cached_infered_data_path, 'wb') as file:
        pickle.dump(inferred_data_dustin_exp, file)
     
    return inferred_data_dustin_exp


def get_dustin_exp_data():
    # shape (8, 6000, 3)
    X, y = dustin_exp.dustin_exp_Xy()
    return X, y

def main():
    N_BATCH = 5
    N_BATCH_EXP_DATA = 8
    means = {}
    filenames = []

    inferred_data = get_inferred_data(N_BATCH)

    i = 0
    # Plot randomly generated data
    for topic in PROBLEMS.keys():
        i += 1
        print(f"Plotting {topic} ({i}/{len(PROBLEMS)})")
        sub_res, filename = create_fig_for_problems(topic, inferred_data, N_BATCH)
        filenames.append(filename)
        means[topic] = sub_res
        
    
    print(means)

    res_file = plot_results(means)
    
    # Plot dustin data
    inferred_data_dustin_exp = get_inferred_data_dustin_exp()
    ## Un-Batch data
    means_dustin_exp = {}
    dustin_filenames = []
    for topic in PROBLEMS.keys():
        sub_res, filename = create_fig_for_problems(topic, inferred_data_dustin_exp, N_BATCH_EXP_DATA, ['dustin'])
        dustin_filenames.append(filename)
        print(f"Dustin exp data: {filename}")
        means_dustin_exp[topic] = sub_res
    res_file_dustin_exp = plot_results(means_dustin_exp)


    
    #merge_pdfs([res_file] + filenames)
    merge_pdfs([res_file_dustin_exp] + dustin_filenames, name='results_dustin_exp_combined.pdf')

    
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

using Revise
#import PyPlot: plt
import DelimitedFiles: readdlm
import Random
ENV["JULIA_LOAD_PATH"]="/home/user/git/CoTETE.jl/src/"

using CoTETE


using OnlinePCA
using OnlinePCA: readcsv, writecsv
using Distributions
using DelimitedFiles
import NeuroAnalysis.plotspiketrain
using Pkg
ENV["PYTHON"] = "/home/user/miniconda3/bin/python"           # example for *nix
Pkg.build("PyCall")
using PyCall
import PPSeq
const seq = PPSeq

#using Plots
#open("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf") do file
    # do stuff with the open file
#end
#using PyCall



@time spike_mat = readdlm("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf", '\t', Any, '\n')
@time spike_mat = spike_mat[:,1:2]
@time spike_mat = spike_mat[1:1000000]
@show(spike_mat[1])

whole_mat = reduce(vcat, spike_mat)

tmp = mktempdir()
writecsv(joinpath(tmp, "Data.csv"), spike_mat)
# Binarization
csv2bin(csvfile=joinpath(tmp, "Data.csv"), binfile=joinpath(tmp, "Data.zst"))
# Summary of data
output = sumr(binfile=joinpath(tmp, "Data.zst"), outdir=tmp)

@time plotspiketrain(whole_mat)|>display
input=joinpath(tmp, "Data.zst")
@time plotspiketrain(input)|>display

#using DataFrames
#using PlotlyJS
# Other Imports

# Songbird metadata
#num_neurons = Float64(sizeof(whole_mat))

# Load spikes.
spikes = seq.Spike[]
num_neurons = 750#sizeof(spikes)
_p = Random.randperm(num_neurons)
#global max_time = 0

function read_file(num_neurons)
    local max_time = 0.0

    for (n, t) in eachrow(readdlm("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf", '\t', Any, '\n'))
        if t > max_time
            max_time = t
        end
        if n<num_neurons
            push!(spikes, seq.Spike(_p[Int(n)], t))
            #@show(estimate_TE_from_event_times(spike_mat[:,1], spike_mat[:,2], 1, 1))

            #@show(spikes)
        end
    end
    return (spikes,max_time)
end
(spikes,max_time) = read_file(num_neurons)
@show(max_time)
#spikes = spikes[1:20]
#max_time = Float64(maximum(spikes))

#num_neurons = 600
# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)


fig = seq.plot_raster(spikes; color="k") # returns matplotlib Figure
fig.set_size_inches([7, 3]);
config = Dict(

    # Model hyperparameters
    :num_sequence_types =>  10,
    :seq_type_conc_param => 1.0,
    :seq_event_rate => 0.5,

    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,

    :neuron_response_conc_param => 0.1,
    :neuron_offset_pseudo_obs => 1.0,
    :neuron_width_pseudo_obs => 1.0,
    :neuron_width_prior => 0.5,

    :num_warp_values => 1,
    :max_warp => 1.0,
    :warp_variance => 1.0,

    :mean_bkgd_spike_rate => 30.0,
    :var_bkgd_spike_rate => 30.0,
    :bkgd_spikes_conc_param => 0.3,
    :max_sequence_length => Inf,

    # MCMC Sampling parameters.
    :num_anneals => 15,
    :samples_per_anneal => 100,
    :max_temperature => 40.0,
    :save_every_during_anneal => 10,
    :samples_after_anneal => 2000,
    :save_every_after_anneal => 10,
    :split_merge_moves_during_anneal => 10,
    :split_merge_moves_after_anneal => 10,
    :split_merge_window => 1.0,

);
#=
# Initialize all spikes to background process.
init_assignments = fill(-1, length(spikes))

# Construct model struct (PPSeq instance).
model = seq.construct_model(config, max_time, num_neurons)

# Run Gibbs sampling with an initial annealing period.
results = seq.easy_sample!(model, spikes, init_assignments, config);


# Grab the final MCMC sample
final_globals = results[:globals_hist][end]
final_events = results[:latent_event_hist][end]
final_assignments = results[:assignment_hist][:, end]

# Helpful utility function that sorts the neurons to reveal sequences.
neuron_ordering = seq.sortperm_neurons(final_globals)

# Plot model-annotated raster.
fig = seq.plot_raster(
    spikes,
    final_events,
    final_assignments,
    neuron_ordering;
    color_cycle=["red", "blue", "yellow", "green","pink","brown"] # colors for each sequence type can be modified.
)
fig.set_size_inches([7, 3]);
=#
using PyCall
py"""
import pyspike
from pyspike import SpikeTrain
import numpy as np
#from quantities import ms, s, Hz
#from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
#spike_train = SpikeTrain(np.array([0.1, 0.3, 0.45, 0.6, 0.9], [0.0, 1.0]))
#print(spike_train)
def wrangle_spikes_trains(spike_trains):
    for sp in spike_trains:
        #import pdb
        #pdb.set_trace()
        print(sp)
    return spike_trains
"""
#spike_trains = py"wrangle_spikes_trains"(spike_mat)

py"""
import pyspike
#from quantities import ms, s, Hz
#from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
#spike_train = SpikeTrain(np.array([0.1, 0.3, 0.45, 0.6, 0.9], [0.0, 1.0]))

def get_spikes_trains(spike_trains):
    #spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt", 4000)

    plt.figure()
    isi_distance = spk.isi_distance_matrix(spike_trains)
    plt.imshow(isi_distance, interpolation='none')
    plt.title("ISI-distance")

    plt.figure()
    spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0,1000))
    plt.imshow(spike_distance, interpolation='none')
    plt.title("SPIKE-distance")

    plt.figure()
    spike_sync = spk.spike_sync_matrix(spike_trains, interval=(2000,4000))
    plt.imshow(spike_sync, interpolation='none')
    plt.title("SPIKE-Sync")

    plt.show()

"""

#plotspiketrain(spike_mat)|>display

#spike_mat = spike_mat[:,1::]
#@show(size(spike_mat))
#@show(sizeof(spike_mat))
#for col in eachcol(spike_mat)
#       println(col)
#end

#m = Matrix{Int}(undef, 1012741, 2)
#for j in 1:2
#m = hcat([col for col in eachcol(spike_mat[:,1:2])])
#@show(m[1])
    #m[:, j] = column(spike_mat)
#end

#for col in eachcol(spike_mat[1:5,:])
#       println(col[:])
#end

#for col in eachrow(spike_mat[1:5,:])
#       println(col[:])
#end

#spike_mat[1,:]
#spike_mat[2,:]
#for col in
#=
totaltime, totallines = open("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf") do f
    #linecounter = 0
    #timetaken = @elapsed
    for l in eachline(f)

        #@show(l)
        #linecounter += 1
    end
    #(timetaken, linecounter)
end
=#

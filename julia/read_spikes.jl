
using Revise
import DelimitedFiles: readdlm
import Random
using ProgressMeter
using OnlinePCA
using OnlinePCA: readcsv, writecsv
using Distributions
using DelimitedFiles
using SpikeSynchrony
using DataFrames
import PPSeq
const seq = PPSeq

import NeuroAnalysis.plotspiketrain

fuction dontcallthis()
    """
    wrapped code, not meant for normal execution.
    """
    using Pkg
    ENV["PYTHON"] = "/home/user/miniconda3/bin/python"           # example for *nix
    Pkg.build("PyCall")
    using PyCall
end

function spike_train_diff(spkd0, spkd_found)
    if !isempty(spkd0) && !isempty(spkd_found)
        maxt1 = findmax(spkd0)[1]
        maxt2 = findmax(spkd_found)[1]
        maxt = findmax([maxt1, maxt2])[1]
        if maxt1 > 0.0 && maxt2 > 0.0
            t, S = SpikeSynchrony.SPIKE_distance_profile(
                spkd0,
                spkd_found;
                t0 = 0.0,
                tf = maxt,
            )
            spkd = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1])
        end
    end
    spkd # this is the same as return spkd
end



spike_mat = readdlm("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf", '\t', Any, '\n')
spike_mat_ = spike_mat[:,1:2]
#@time spike_mat = spike_mat_[1:1000000]
spike_mat = spike_mat_[1:1975]

temp = [i for (indi,i) in enumerate(spike_mat_)]
#mat0 = spike_train_diff.(temp', temp)
#mat1 = spike_train_diff.(temp, temp')

# for plotting vertically concatone spike_matrix
# into whole matrix.
whole_mat = reduce(vcat, spike_mat)

# Load spikes.

num_neurons = size(spike_mat)[1]


function read_file(num_neurons)
    """
    take a global spike train variable

    and convert it into a list of weird struct types:
    that
    populated dictionary of
    keys: spike id index, values: a list of spike times.

    """
    spikes = seq.Spike[]
    _p = Random.randperm(num_neurons)

    spike_dict = Dict()
    local max_time = 0.0
    for (n, t) in eachrow(readdlm("../data/7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf", '\t', Any, '\n'))
        if t > max_time
            max_time = t
        end
        if !(n in keys(spike_dict))
            spike_dict[n] = []
        end
        if n<num_neurons
            append!(spike_dict[n],t)
            push!(spikes, seq.Spike(_p[Int(n)], t))
        end
    end
    return (spikes,max_time,spike_dict)
end
(spikes,max_time) = read_file(num_neurons)
#@show(max_time)
ab = []

function get_pca(spike_dict)
    using OnlineStats
    iter_list = collect(values(Vector(Float64,spike_dict)))
    iter_req = size(iter_list)[1]
    o = OnlineStats.CCIPCA(100,iter_req)
    for x in iter_list
        print(typeof(x))
        #fit!(o,x)
        #OnlineStats.transform(o, x)
        OnlineStats.fittransform!(o, x) # Fit u4 and then project u4 into the space
        sort!(o)                         # Sort from high to low eigenvalues
        @show(o[1])                             # Get primary (1st) eigenvector
        @show(OnlineStats.variation(o))         # Get the variation (explained) "by" each eigenvector
    end
end

function applied_spike_diff(spike_dict)
    iter_list = collect(values(spike_dict))
    iter_req = size(iter_list)[1]^2
    p = Progress(iter_req, 1)
    @inbounds @showprogress for i in iter_list
        @inbounds for j in iter_list
            push!(ab,spike_train_difference(i, j))
            next!(p)
        end
    end
    ab
end
@time ab = applied_spike_diff(spike_dict)

ab = []
@inline for (indi,i) in spike_dict
    @inline for (indj,j) in spike_dict
        if indi!=indj
            if size(spike_dict[indi])[1]>1
                if size(spike_dict[indj])[1]>1
                    push!(ab,raster_difference(spike_dict[indi], spike_dict[indj]))
                    @show(last(ab))
                end
            end
        end
    end
end
@show(ab)
@show(spike_mat[1])

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

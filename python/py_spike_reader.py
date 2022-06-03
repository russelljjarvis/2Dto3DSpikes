from multiprocessing import Process
import numpy as np
import pyspike as spk
import quantities as pq
from GA import GPFA
from pyspike import SpikeTrain as SpikeTrainPy
import pickle
from pandas import read_csv
df = read_csv("7d933154b27ff70fc2df9ed8bec39480-spikes-00001-0.gdf", names=["id", "time"], skiprows=0, usecols=[0,1], delimiter="\t",
                dtype={"id":np.uint64, "time":np.float64}, engine="c")
#print(df)
data2 = np.vstack((df["time"], df["id"]))
print(len(np.unique(df["id"])))
spike_ids = np.unique(df["id"])
spike_times = df#["time"]

sts = []
sts2 = []
DURATION_MS = np.max(df["time"])
from neo import SpikeTrain as SpikeTrainN
import vlgp
trials = []
for j in range(0,max(spike_ids)):
    if j<49:
        spike_times_ = list(df.where(df["id"]==j))
        print(spike_times_)
        st = SpikeTrainN(spike_times_ * pq.ms, t_stop=DURATION_MS * pq.ms)
        sts.append([st])
        trials.append({"y": spike_times_, "id": j})
        #sts2.append([spike_times_)
#trials = [{'ID': i, 'y': y} for i, y in enumerate(sample['y'])]  # make trials

fit = vlgp.fit(
    trials,
    n_factors=3,  # dimensionality of latent process
    max_iter=20,  # maximum number of iterations
    min_iter=10  # minimum number of iterations
)
trials = fit['trials']  # extract trials
mu = trials[0]['mu']  # extract posterior latent
W = np.linalg.lstsq(mu, x[0, ...], rcond=None)[0]
mu = mu @ W
# Plot posterior latent
plt.figure(figsize=(20, 5))
plt.plot(x[0, ...] + 2 * np.arange(3), color="b")
plt.plot(mu + 2 * np.arange(3), color="r")
plt.axis("off")
plt.show()
plt.close()
with open("spike_df","wb") as f:
    pickle.dump(sts,f)
    #sts.append(SpikeTrainPy(spike_times_,edges=(0.0,DURATION_MS)))

#import pdb;
#pdb.set_trace()


bin_size = 20 * pq.ms
latent_dimensionality = 2

gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)
gpfa_2dim.fit(sts)
print(gpfa_2dim.params_estimated.keys())
trajectories = gpfa_2dim.transform(sts)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

linewidth_single_trial = 0.5
color_single_trial = 'C0'
alpha_single_trial = 0.5

linewidth_trial_average = 2
color_trial_average = 'C1'
ax2.set_title('Latent dynamics extracted by GPFA')
ax2.set_xlabel('Dim 1')
ax2.set_ylabel('Dim 2')
ax2.set_aspect(1)
# single trial trajectories
for single_trial_trajectory in trajectories:
    ax2.plot(single_trial_trajectory[0], single_trial_trajectory[1], '-', lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)
# trial averaged trajectory
average_trajectory = np.mean(trajectories, axis=0)
ax2.plot(average_trajectory[0], average_trajectory[1], '-', lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')
ax2.legend()

plt.tight_layout()
plt.show()

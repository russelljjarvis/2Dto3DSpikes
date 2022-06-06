# 2Dto3D spike re-organization and transformation.

This is code does not do anything useful yet, its a collection of loosely related large scale spike train analysis techinques applied in Julia and Python.

The original intention was to apply existing dimensionality reduction techniques to reduce the size of >1million cell spike trains down to something useful. The code here does not do that yet, and may not do it ever.

To get a Julia JLD file of V1 spiking data, run: ```https://cloudstor.aarnet.edu.au/plus/s/sbpQ2vlihvsNOUh/download```

TODO: give instructions to load this data into Julia.

Others goals of this repository:
* 1 compute the Kreutz spike distance.
* **2 Map 2D spike events (biological spike raster plots) to 3D raster plots that look like 3D event based camera spike events.**

## Getting Started

```
julia
```
At the prompt hit ] to enter julia's package mode:
```
] add https://github.com/lindermanlab/PPSeq.jl
```
This is short hand for
```
using Pkg
Pkg.add(url="https://github.com/lindermanlab/PPSeq.jl")
```

All other missing packages can be installed in a similar manner
Most often missing packages are registered.

For example

```
] add Plots.jl
```
Installs the registered Julia package plots.

Ultimately this project will need a Project.toml file this is analogous to a
python requirements.txt file

```
cd julia
julia
include("read_spikes.jl")
```

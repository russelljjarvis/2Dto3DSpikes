![1millionspikes.png](1millionspikes.png)
This is a repository of code that does not do anything useful yet.

The intention was to apply existing dimensionality reduction techniques to reduce the size of >1million cell spike trains down to something useful. This does not do that yet.

Another goal of this repository was to compute the Kreutz spike distance.

# Getting started. If Julia is installed.

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

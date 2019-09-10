On the interplay between Acceleration and Identification
===

Numerical experiments and figures can be produced by running top-level scripts, in Julia v1.1 or higher.

## Activating the environment
The first time, launch julia from the root of this repo, and execute the following:
```julia
]activate .
]instantiate
```

This needs to be done only once, afterward a simple `]activate .` is enough.

## Running the scripts
Scripts can be run as follow:
```julia
include("plot_1_lasso2d.jl")
include("plot_2_distball1.3.jl")
include("plot_3_distball2.6.jl")
include("script_expenums_l12.jl")
include("script_expenums_l1.jl")
include("script_expenums_lnuclear.jl")
```

By default, figures will be saved in the folder `./figs`.
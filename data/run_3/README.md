# Run 3

## Parameters

* $L\in\{20,40,60,\dotsc,200\}$
* $c=16$
* $T=L$
* $p\in\{0.01,0.02,0.03,\dotsc,0.05\}$
* $\eta=0.1$
* $10000$ shots
* 4 hours, 8 CPUs per task

## Data format

Filename: `run_3_{L}_{p}.npy`
Contains shape $(2,)$ `np.ndarray` containing fail rate and anyon density.

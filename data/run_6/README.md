# Run 6

## Parameters

* $L\in\{20,40,60,\dotsc,100\}$
* $c=16$
* $T=L$
* $p\in\{0.001,0.002,0.003,\dotsc,0.010\}$
* $\eta=0.1$
* $100000$ shots
* 4 hours, 8 CPUs per task

## Data format

Filename: `run_6_{L}_{1000*p}.npy`
Contains shape $(2,)$ ragged `np.ndarray` containing fail rate and anyon density history.

## Runtimes (min)

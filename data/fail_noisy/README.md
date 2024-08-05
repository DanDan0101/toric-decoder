# Run 2

## Parameters

* $L\in\{20,40,60,\dotsc,200\}$
* $c\in\{2,4,8,\dotsc,64\}$
* $T=L$
* $p=0.05$
* $\eta=0.1$
* $10000$ shots
* 4 hours, 8 CPUs per task

## Data format

Filename: `fail_noisy_{L}_{c}.npy`
Contains shape $(2,)$ `np.ndarray` containing fail rate and anyon density.

## Timeouts

* $L=120$, $c=64$
* $L\geq140$

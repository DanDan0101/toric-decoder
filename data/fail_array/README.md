# Run 1

## Parameters

* $L\in\{20,40,60,80,100\}$
* $c\in\{2,4,8,16,32\}$
* $T=L$
* $p_{\text{initial}}=0.05$
* $p_{\text{intermediate}}=0$
* $\eta=0.1$
* $10000$ shots
* 1 hour, 4 CPUs per task

## Data format

Filename: `fail_array_{L}_{c}.npy`
Contains shape $(3,)$ `np.ndarray` containing fail rates of MWPM, 2D, and 2D + MWPM.

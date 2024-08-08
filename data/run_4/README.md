# Run 4

## Parameters

* $L\in\{20,40,60,\dotsc,100\}$
* $c=16$
* $T=L$
* $p\in2^{-\{6,7,8,\dotsc,10\}}$
* $\eta=0.1$
* $10000$ shots
* 1 hour, 8 CPUs per task

## Data format

Filename: `run_4_{L}_{p}.npy`
Contains shape $(2,)$ ragged `np.ndarray` containing fail rate and anyon density history.

## Runtimes (min)

| $L$ | Low | High |
| --- | --- | ---- |
| 20  | 1:11 | 2:03 |
| 40  | 1:39 | 2:31 |
| 60  | 2:34 | 5:18 |
| 80  | 5:54 | 8:22 |
| 100 | 7:40 | 26:54|

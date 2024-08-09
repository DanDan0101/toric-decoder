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

| $1000p\setminus L$ | 20 | 40 | 60 | 80 | 100 |
| ------------------ | -- | -- | -- | -- | --- |
| 1                  | 2  | 10 | 18 | 57 | 71  |
| 2                  | 6  | 8  | 20 | 61 | 78  |
| 3                  | 3  | 9  | 20 | 65 | 59  |
| 4                  | 3  | 7  | 22 | 60 | 65  |
| 5                  | 3  | 7  | 30 | 63 | 74  |
| 6                  | 7  | 9  | 32 | 84 | 81  |
| 7                  | 7  | 9  | 32 | 73 | 178 |
| 8                  | 7  | 9  | 36 | 53 | 97  |
| 9                  | 4  | 11 | 36 | 65 | 117 |
| 10                 | 7  | 7  | 37 | 61 | 122 |

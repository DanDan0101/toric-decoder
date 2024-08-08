# Run 5

## Parameters

* $L\in\{100,200,300,\dotsc,500\}$
* $c=16$
* $T=L$
* $p\in2^{-\{1,2,3,\dotsc,10\}}$
* $\eta=0.1$
* $10000$ shots
* 4 hours, 8 CPUs per task

## Data format

Filename: `run_5_{L}_{p}.npy`
Contains a singular float of the final anyon density.

## Runtimes (min)

| $1/p\setminus L$ | 100      | 200      | 300      | 400      | 500      |
| ---------------- | -------- | -------- | -------- | -------- | -------- |
| 1024             | 00:07:53 | 01:37:42 | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 512              | 00:14:40 | 02:07:17 | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 256              | 00:17:23 | 02:20:07 | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 128              | 00:21:43 | 04:00:22 | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 64               | 00:29:44 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 32               | 00:41:22 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 16               | 00:47:57 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 8                | 00:58:28 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 4                | 00:45:16 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |
| 2                | 00:40:33 | TIMEOUT  | TIMEOUT  | TIMEOUT  | TIMEOUT  |

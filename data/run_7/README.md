# Run 7

## Parameters

* $L\in\{20,40,60,\dotsc,100\}$
* $c=16$
* $T=L$
* $p=0.004$
* $\eta=0.1$
* $100000$ shots x 10
* 2 hours, 8 CPUs per task

## Data format

Filename: `run_7_{L}_{ID}.npy`
Contains single float of fail rate.

## Runtime

| $L$ | Time     | Std      |
| --- | -------- | -------- |
| 20  | 00:04:28 | 00:00:42 |
| 40  | 00:07:58 | 00:01:28 |
| 60  | 00:19:45 | 00:01:26 |
| 80  | 00:45:49 | 00:01:22 |
| 100 | 01:40:53 | 00:04:35 |

### Timeouts

The last two tasks (48 and 49) of $L=100$ timed out.

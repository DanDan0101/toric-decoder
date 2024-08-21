# Run 10

## Parameters

* $L\in\{100,200,300,\dotsc,500\}$
* $c=16$
* $T=L$
* $p\in\{0.0035,0.0036,0.0037,\dotsc,0.0045\}$
* $\eta=0.1$
* $960000/(L/100)^3$ shots
* 1.5 hours, 1 GPU per task

## Data format

Filename: `run_10_{n}.npy`, where $L=100\left(1+\left\lfloor\frac{n}{11}\right\rfloor\right)$ and $p_{\mathrm{error}}=(35+n\mod11)\cdot10^{-4}$.

Contains (2,) np.ndarray of fail rate and number of samples.

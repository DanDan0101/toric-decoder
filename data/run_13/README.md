# Run 13

## Parameters

* $L\in\{50,100,150,\dotsc,250\}$
* $c=16$
* $T=L$
* $p\in\{0.00400,0.00401,0.00402,\dotsc,0.00419\}$
* $\eta=0.1$
* 12 hours, 1 GPU per task, x 10

## Data format

Filename: `run_13_{n}.npy`, where $L=50\left(1+\left\lfloor\frac{n\mod100}{20}\right\rfloor\right)$ and $p_{\mathrm{error}}=(400+n\mod20)\cdot10^{-5}$.

Contains (2,) np.ndarray of fail rate and number of samples.

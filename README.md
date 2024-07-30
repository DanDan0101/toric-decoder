# toric-decoder

Local decoders for the toric code, based on [Cellular-automaton decoders for topological quantum memories](https://arxiv.org/abs/1406.2338) (Herold *et al.*).

We assume that only $X$ errors may occur, independently Bernoulli distributed on each spin in the lattice. Hence, we focus on $m$ (plaquette) anyons.

## Dependencies

* `numpy`
* `scipy`
* `matplotlib`
* `seaborn`
* `tqdm`
* `networkx`
* `pymatching`
* `multiprocess`

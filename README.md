# dl_kcat
Machine learning tools for predicting rate constants from enzyme dynamics.

The scripts in `dl_kcat` train deep learning models to predict an enzyme mutant's specific activity based on its pre-reaction structural dynamics. The structural dynamics are captured by a set of manually engineered features (interatomic distances, angles, and torsions) that collectively describe the structure of the active site over some time interval. Therefore, the data are a multivariate time series comprised of 70 features and typically 30+ time points. These time series are drawn from molecular dynamics simulations of attempted reactions, which were sampled using [transition interface sampling (TIS)](https://pubs.aip.org/aip/jcp/article/118/17/7762/185320/A-novel-path-sampling-method-for-the-calculation).

The reaction that was simulated was a methyl transfer:

<img src="demo/figs/rxn.png" width="400">

Note that $C_{5}$ is originally bound to $C_{4}$ in the reactant, but it is transfered during the reaction and bound to $C_{7}$ in the product.

We use TIS to collect examples of both successful, as well as failed, attempted reactions by the enzyme.

### Here is a successful reaction:
Substrate is shown in light orange, and its orientation corresponds with the above Lewis structure. NADPH is light purple. Mg ions are shown as white spheres with their coordinating waters in stick representation. The enzyme is shown in light gray. The migrating methyl, $C_{5}$, appears to be "floating" near the center of the gif. The graphics showing its bonds to either $C_{4}$ or $C_{7}$ were erased for visual clarity.

<img src="demo/figs/r1-opt.gif" width="300">

### Here is a failed reaction:

<img src="demo/figs/nr2-opt.gif" width="300">

Note that $C_{5}$ makes significant progress toward the left, before hitching and ultimately returning to its starting point bound to $C_{4}$. 

This repo provides scripts for handling data from these kinds of simulations and using them to train transformer- or LSTM-based models for various learning tasks. The purposes for these models and analyses are to (i) understand and identify structural drivers of catalysis and (ii) predict a mutant enzyme's catalytic activity from limited data. 

The main working scripts are `./scripts/transformer_1.py` and `./scripts/lstm_1.py`, which both source functions primarily from `./sctipts/pred_kcat.py`.

For a more in depth demonstration, check out the `./demo/` folder.

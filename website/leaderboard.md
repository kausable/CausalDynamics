# Leaderboard

We showcase results of existing state-of-the-art algorithms evaluated on our preprocessed dataset. 

Evaluation is done after running e.g., `python eval.py --data_dir data/coupled --causal_model pcmciplus` script. The default setup for simple case refers to ODE systems ($\delta = 0$), and no unobserved confounder (585 graphs). In the coupled system, the default setup refers to dynamics with $n=10$ coupled ODEs ($\delta = 0$), no confounder, no time-lag ($\tau = 0$), and no internal standardization (4745 graphs). Each experiment type varies the mentioned configuration while keeping other unchanged.

**Table 1: Baseline AUROC (↑) / AUPRC (↑) scores across different experiments in the hierarchy of increasingly complex dynamical systems. The scores are averaged over all generated graphs within each experiment.**


| Experiments     | PCMCI+    | FPCMCI    | Varlingam | DYNOTEARS | NGC       | TSCI      | CUTS+     |
| --------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **— Simple —**   |           |           |           |           |           |           |           |
| Default         | .47 / .68 | .51 / .70 | .50 / .69 | .43 / .67 | .50 / .68 | .46 / .68 | .48 / .68 |
| Noise           | .50 / .69 | .52 / .70 | .53 / .69 | .48 / .68 | .50 / .68 | .49 / .68 | .50 / .68 |
| Confounder      | .48 / .59 | .50 / .59 | .48 / .57 | .52 / .63 | .50 / .57 | .53 / .65 | .52 / .59 |
| **— Coupled —**  |           |           |           |           |           |           |           |
| Default         | .68 / .27 | .67 / .24 | .57 / .18 | .66 / .32 | .50 / .16 | .69 / .36 | .50 / .16 |
| Noise           | .66 / .30 | .62 / .25 | .58 / .19 | .62 / .29 | .50 / .16 | .52 / .18 | .50 / .16 |
| Confounder      | .56 / .20 | .57 / .19 | .51 / .17 | .49 / .17 | .50 / .17 | .49 / .18 | .50 / .17 |
| Lag             | .55 / .23 | .55 / .23 | .52 / .21 | .52 / .22 | .50 / .20 | .53 / .23 | .50 / .20 |
| Standardize     | .69 / .27 | .67 / .23 | .57 / .18 | .68 / .34 | .50 / .15 | .69 / .35 | .51 / .16 |
| **— Climate —**  |           |           |           |           |           |           |           |
| Atmos-ocean     | .69 / .88 | .50 / .81 | .50 / .81 | .62 / .86 | .49 / .81 | .58 / .84 | .50 / .81 |
| ENSO modes      | .50 / .81 | .51 / .81 | .51 / .81 | .50 / .81 | .50 / .81 | .50 / .81 | .49 / .81 |

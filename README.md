# Impact of Imputation Strategies on Fairness in Machine Learning
This is the code repo for the Journal of Artifical Research (JAIR) paper: add doi link here later

## Abstract
Research on Fairness and Bias Mitigation in Machine Learning often uses a set of reference datasets for the design and evaluation of novel approaches or definitions. While these datasets are well structured and useful for the comparison of various approaches, they do not reflect that datasets commonly used in real-world applications can have missing values. When such missing values are encountered, the use of imputation strategies is commonplace. However, as imputation strategies potentially alter the distribution of data they can also affect the performance, and potentially the fairness, of the resulting predictions, a topic not yet well understood in the fairness literature. In this article, we investigate the impact of different imputation strategies on classical performance and fairness in classification settings. We find that the selected imputation strategy, along with other factors including the type of classification algorithm, can significantly affect performance and fairness outcomes. The results of our experiments indicate that the choice of imputation strategy is an important factor when considering fairness in Machine Learning. We also provide some insights and guidance for researchers to help navigate imputation approaches for fairness.

## Main Results

Performance and fairness metrics vs. Machine Learning Model and Imputation Strategy: not all ML models and metrics respond in a similar manner to the imputation strategies applied.
![Figure 1](../assets/ImputationPerformanceByMetric.png?raw=true)
![Figure 2](../assets/ImputationFairnessByMetric.png?raw=true)

Following on from this observation, a robust three factor ANOVA reveals similar observations.
![ANOVA Results](../assets/ANOVA.png?raw=true)

To try and capture different canonical preference structures, a Friedman ranking was used to illustrate the fairness vs. performance (accuracy etc.) trade-off.
![Friedman Rankings](../assets/rankings.png)

## Notebooks

One notebook per dataset (Adult, German Credit, and COMPAS) is available to recreate the results used in the paper. 

## Citation
```
@article{caton2022imputing,
	title={{Impact of Imputation Strategies on Fairness in Machine Learning}},
	author={Caton, S. and Malisetty, S. and Haas, C.},
	journal={Journal of Artificial Intelligence Research (JAIR)},
	volume={},
	pages={},
	year={2022}
}
```

## Disclaimer

These software distributions are open source, licensed under the GNU General Public License (v3 or later). Note that this is the full GPL, which allows many free uses, but does not allow its incorporation (even in part or in translation) into any type of proprietary software which you distribute. Commercial licensing is also available; please contact us if you are interested.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.



# Измерение масс скоплений галактик по данным ACT+Planck с помощью Deep Learning

##  Описание проекта

Проект посвящен измерению масс скоплений галактик с использованием методов глубокого обучения на основе комбинированных данных Атакамского космологического телескопа (ACT) и Planck.

##  Научный контекст

Скопления галактик - наиболее массивные гравитационно-связанные системы во Вселенной, состоящие из:
- 85% темной материи
- 12% горячего барионного газа  
- 3% звезд

Точные измерения масс скоплений играют ключевую роль в космологических исследованиях, включая изучение темной материи и определение фундаментальных космологических параметров.

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A Machine Learning Approach for Dynamical Mass Measurements of Galaxy Clusters* | SDM | https://arxiv.org/abs/1410.0686, https://arxiv.org/abs/1509.05409 |
| *A Deep Learning Approach to Galaxy Cluster X-ray Masses* | CNN | https://arxiv.org/abs/1810.07703 |
| *An application of machine learning techniques to galaxy cluster mass estimation using the MACSIS simulations* | OLR, RR, BRR, KRR, SVR, DT, BDT, ADA, kNN | https://arxiv.org/abs/1810.08430 |
| *A Robust and Efficient Deep Learning Method for Dynamical Mass Measurements of Galaxy Clusters* | CNN | https://arxiv.org/abs/1902.05950 |
| *Multiwavelength cluster mass estimates and machine learning* | GB, RF | https://arxiv.org/abs/1905.09920 |
| *Using X-Ray Morphological Parameters to Strengthen Galaxy Cluster Mass Estimates via Machine Learning* | RF | https://arxiv.org/abs/1908.02765 |
| *Large-scale structures in the LCDM Universe: network analysis and machine learning* | XGBoost | https://arxiv.org/abs/1910.07868 |
| *Dynamical mass inference of galaxy clusters with neural flows* | NF (MADE) | https://arxiv.org/abs/2003.05951 |
| *Mass Estimation of Galaxy Clusters with Deep Learning I: Sunyaev-Zel'dovich Effect* | U-Net | https://arxiv.org/abs/2003.06135 |
| *Galaxy cluster mass estimation with deep learning and hydrodynamical simulations* | CNN | https://arxiv.org/abs/2005.11819 |
| *Mass Estimation of Galaxy Clusters with Deep Learning II: CMB Cluster Lensing* | U-NET | https://arxiv.org/abs/2005.13985 |
| *Multiwavelength classification of X-ray selected galaxy cluster candidates using convolutional neural networks* | CNN | https://arxiv.org/abs/2006.05998 |
| *Approximate Bayesian Uncertainties on Deep Learning Dynamical Mass Estimates of Galaxy Clusters* | BNN | https://arxiv.org/abs/2006.13231 |
| *Simulation-based inference of dynamical galaxy cluster masses with 3D convolutional neural networks* | CNN | https://arxiv.org/abs/2009.03340 |
| *DeepSZ: Identification of Sunyaev-Zel'dovich Galaxy Clusters using Deep Learning* | CNN | https://arxiv.org/abs/2102.13123 |
| *DeepMerge II: Building Robust Deep Learning Algorithms for Merging Galaxy Identification Across Domains* | CNN | https://arxiv.org/abs/2103.01373 |
| *Mass Estimation of Planck Galaxy Clusters using Deep Learning* | CNN | https://arxiv.org/abs/2111.01933 |
| *Estimating Cluster Masses from SDSS Multi-band Images with Transfer Learning* | CNN | https://arxiv.org/abs/2203.06288 |
| *Machine Learning methods to estimate observational properties of galaxy clusters in large volume cosmological N-body simulations* | RF, XGBoost, NN | https://arxiv.org/abs/2204.10751 |

[from here](https://github.com/georgestein/ml-in-cosmology)

## Каталоги скоплений для обучения
- [ACTDR6](https://extragalactic.phys.wits.ac.za/act-dr6-clusters/v0.10/) (main)

Дополнительные:
- ACTDR5
- PSZ2
- PSZ1
- SPT-SZ
- SPTDEEP
- и тд


## Задача
Построить архиитектуру модели, которая на разных красных смещениях будет показывать хорошее качество предсказания массы 

> [!NOTE]
> Красное смещение - индикатор расстояния во Вселенной, который завивисит от нескольких факторов, главные из которых:
> - убегание галактик
> - расширение Вселенной 


По сути вы сравниваетесь с алгоритмом [nemo](https://github.com/simonsobs/nemo)
Пример измерения массы этим алгоритмом:
.. code-block::

   nemoMass DR6ClusterSearch.yml -c erass1cl_primary_v3.2.fits -F -z z
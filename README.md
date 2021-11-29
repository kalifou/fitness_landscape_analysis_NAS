# Fitness Landscape Analysis for Neural Architecture Search:

## Step 1: *Install Nasbench-101 submodule*

## Generate descriptive data of NASbench-101:
  * **cd nasbench/**
  * **python nasbench/nasalg/landscape.py  > experiments/full-landscape.json**

## Analyze the Fitness Landscape of NASBench-101    
  * **Statistics and FDC:** nasbench/experiments/fitness-landscape.Rmd (R-sudio)
  * **Local optima estimation:** card_optima/local_search_for_card_optima.py
  * **Local optima analysis:** notebooks/Card_Optima_-_CIFAR_-_10.ipynb (notebook)
  * **Random Walk Analysis:** notebooks/Random_Walks_Analysis_-_CIFAR-10.ipynb'(notebook)
  * **Persistence:** notebooks/Persistence_-_CIFAR-10.ipynb'(notebook)
            

##Project using this code:
**Fitness Landscape Footprint: A Framework to Compare Neural Architecture Search Problems**,
*arXiv:2111.01584, 
 Kalifou René Traoré, Andrés Camero, Xiao Xiang Zhu.*


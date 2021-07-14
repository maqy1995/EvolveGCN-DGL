# Implement EvolveGCN with DGL
official code: [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)  
another implement by [pyG_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py)  

## Dependency:
> dgl 0.6.1
> pandas
> numpy

## How to Run
* download Elliptic dataset from [kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set)
* unzip the dataset into a raw directory, such as /home/Elliptic/
* make a new dir to save processed data, such as /home/Elliptic/processed/
* run train.py by:
```bash
python train.py --raw-dir /home/Elliptic/ --processed-dir /home/Elliptic/processed/
```

## Attention:  
only used elliptic dataset now.

## TODO:
* refactor code.
* add EvolveGCN-H model
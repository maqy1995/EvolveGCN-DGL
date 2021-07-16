# Implement EvolveGCN with DGL
official code: [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)  
another implement by [pyG_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py)  

## Dependency:
* dgl 0.6.1
* pandas
* numpy

## Run
* donwload Elliptic dataset from [kaggle](kaggle.com/ellipticco/elliptic-data-set)
* unzip the dataset into a raw directory, such as /home/Elliptic/
* make a new dir to save processed data, such as /home/Elliptic/processed/
* run train.py by:
```bash
python train.py --raw-dir /home/Elliptic/ --processed-dir /home/Elliptic/processed
```

## Attention:  
* only used Elliptic dataset now.
* EvolveGCN-H is not solid in Elliptic dataset, the official code is the same.  
Official Code Reuslt:  
1. set seed to 1234, finally result is :
> TEST epoch 189: TEST measures for class 1 - precision 0.3875 - recall 0.5714 - f1 0.4618  
2. not set seed manually, run the same code three times:
> TEST epoch 168: TEST measures for class 1 - precision 0.3189 - recall 0.0680 - f1 0.1121  
> TEST epoch 270: TEST measures for class 1 - precision 0.3517 - recall 0.3018 - f1 0.3249  
> TEST epoch 455: TEST measures for class 1 - precision 0.2271 - recall 0.2995 - f1 0.2583  

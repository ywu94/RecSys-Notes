# RecSys-Notes

Classic papers and resources on recommendation system, along with python implementation (focusing on PyTorch).

## Covered Model & Performance

Model | Key Idea | Criteo Test AUC | Implementation
--- | --- | --- | ---
Factorization Machine | Use embedding and dot product to learn interaction between features | `0.792564` after one epoch | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Factorization%20Machine.pdf)<br/>[BiClf PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/FM_BinClf_Torch.py)
Field-aware Factorization Machine | Expand embedding matrix to capture interaction between different fields separately | | | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Field-aware%20Factorization%20Machine.pdf)
Deep Factorization Machine | Combine `DNN` and `FM` | `0.801416` after two epoches | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/DeepFM-%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction.pdf)<br/>[BiClf PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DeepFM_BinClf_Torch.py)
Deep Cross Network | Use `Cross Net` to capture higher-degree interaction | | `0.804345` after three epoches | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Deep%20%26%20Cross%20Network%20for%20Ads%20Click%20Prediction.pdf)<br/>[BiClfPyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DCN_BinClf_Torch.py)
Extreme Deep Factorization Machine | Introduce `Compressed Interaction Network` to enhance Cross Net, capture feature interaction at vector level instead of bit level | | | 

## Data Preparation

### Criteo Data

Criteo data can be downloaded at [Kaggle Displaying Ads Dataset](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/), to prepare the data, do the following steps.

* Git clone this repo to your local environment and change directory to your local repo

* Create directory `mkdir ./Data/crieto/criteo_raw_artifact`

* Unzip the criteo data `dac.tar.gz` and move `train.txt` and `test.txt` to `./Data/crieto/criteo_raw_artifact`

* Run the following command in shell

   ```bat
   cd ./Data/crieto
   python3 split.py
   python3 prepare.py
   ```
   
   Note that the current implementation of `prepare.py` will have all the prepared data stored in memory which may not be feasible for machines with small memory. A work around would be to store the prepared data in partition.


# RecSys-Notes

Classic papers and resources on recommendation system, along with python implementation (focusing on PyTorch).

## Implemented Model & Performance

Model | Criteo Train AUC | Criteo Test AUC | Implementation | Note
--- | --- | --- | --- | ---
Factorization Machine | 0.805224 | 0.792564 | [PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/FM_BinClf_Torch.py) | 
Deep Factorization Machine | 0.819064 | 0.801416 | [PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DeepFM_BinClf_Torch.py) | 
Deep Cross Network | | | [PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DCN_BinClf_Torch.py) |  

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


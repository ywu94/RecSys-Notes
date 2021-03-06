# RecSys-Notes

Classic papers and resources on recommendation system, along with python implementation (focusing on PyTorch).

## What should we consider about recommendation - [From Netflix](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a): 
> We want our recommendations to be accurate in that they are relevant to the tastes of our members, but they also need to be diverse so that we can address the spectrum of a member’s interests versus only focusing on one. We want to be able to highlight the depth in the catalog we have in those interests and also the breadth we have across other areas to help our members explore and even find new interests. We want our recommendations to be fresh and responsive to the actions a member takes, such as watching a show, adding to their list, or rating; but we also want some stability so that people are familiar with their homepage and can easily find videos they’ve been recommended in the recent past.

## Covered Model & Performance

Model | Key Idea | Recommended Hyperparameter | Criteo Test AUC | Implementation
--- | --- | --- | --- | ---
Factorization Machine | Use embedding and dot product to model low-level interaction explicitly | | `0.792564` after one epoch | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Factorization%20Machine.pdf)<br/>[PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/FM_BinClf_Torch.py)
Field-aware Factorization Machine | Model interactions between different fields differently | | | | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Field-aware%20Factorization%20Machine.pdf)
Deep Factorization Machine | Use `FM` to model low-level interaction explicitly and `DNN` to model high-level interaction implicitly | DNN: `3 * 400`| `0.801416` after two epoches | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/DeepFM-%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction.pdf)<br/>[PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DeepFM_BinClf_Torch.py)
Deep Cross Network | Use `Cross Net` to model bit-level interaction between feature embedding explicitly and `DNN` to model high-level interaction implicitly | Cross: `6`<br/>DNN: `2*1024`| `0.801345` after three epoches | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/Deep%20%26%20Cross%20Network%20for%20Ads%20Click%20Prediction.pdf)<br/>[PyTorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/DCN_BinClf_Torch.py)
Extreme Deep Factorization Machine | Introduce `Compressed Interaction Network` to enhance Cross Net, capture feature interaction at vector level instead of bit level | CIN: `3*200`<br/>DNN: `4*400`| `0.804545` after two epoches | [Paper](https://github.com/ywu94/RecSys-Notes/blob/master/Papers/xDeepFM.pdf)<br/>[Pytorch](https://github.com/ywu94/RecSys-Notes/blob/master/Implementations/xDeepFM_BinClf_Torch.py)

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


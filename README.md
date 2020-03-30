# RecSys-Notes

Classic papers and resources on recommendation system, along with python implementation (focusing on PyTorch).

## Data

### Criteo Data

Criteo data can be downloaded at [Kaggle Displaying Ads Dataset](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/), to prepare the data, do the following steps.

* Git clone this repo to your local environment `{local repo}`

* Create directory `{local repo}/Data/crieto/criteo_raw_artifact`

* Unzip the criteo data `dac.tar.gz` and move `train.txt` and `test.txt` to `{local repo}/Data/crieto/criteo_raw_artifact`

* Run the following command in shell

   ```bat
   cd {local repo}/Data/crieto
   python3 split.py
   ```


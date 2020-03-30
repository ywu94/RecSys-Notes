# Recommendation-Papers
 Classic papers and resources on recommendation, along with python implementation (focusing on PyTorch).

## Data

### Criteo Data

Criteo data can be downloaded at [Kaggle Displaying Ads Dataset](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/), to prepare the data, do the following steps.

* Git clone this repo to your local environment

* Create directory `{Local Repo}/Data/crieto/criteo_raw_artifact`

* Unzip the criteo data `dac.tar.gz` and move the `train.txt` and `test.txt` to `{Local Repo}/Data/crieto/criteo_raw_artifact`

* Run the following command in shell

   ```bat
   cd {Local Repo}/Data/crieto
   python3 split.py
   ```


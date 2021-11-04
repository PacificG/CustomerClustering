# CustomerClustering

> To reproduce the works here. clone the repo. and run the following command:
```bash
git clone https://www.github.com/pacificg/CustomerClustering.git
```

make sure you have virtualenv installed.

```bash
virtualenv --version
```
if not installed, run the following command:
```bash
pip install virtualenv
```

then run the following command to create a virtualenv:
```bash 
virtualenv venv
```
activate the virtualenv(for ubuntu):
```bash
source venv/bin/activate
```
activate the virtualenv(for windows):
```bash
venv\Scripts\activate
```

then run the following command to install the dependencies:
```bash 
pip install -r requirements.txt
```

then run the following command to run do the preprocessing:

```bash
python preprocess.py \n
    --pathToCSV data/marketing_campaign.csv \n
    --pathToSaveProcessedCSV data/processed_marketing_campaign.csv
```
fireup the jypyter notebook:    
```bash 
jupyter notebook
```

check the KMeans notebook:    
```bash
jupyter notebook EDAClustering.ipynb
```
# Task 1: Customer Clustering with K-Means 

Now we will use the K-Means algorithm to cluster our customers into groups. I created some clusters for you to see how the algorithm works. The algorithm is very simple:  it takes a set of points and then assigns each point to the cluster with the closest centroid. 

Run the EDAClustering notebook:
```bash
jupyter notebook EDAClustering.ipynb
```

# Task 2: Binomial Classificaiton with XGBoost

Run the BinomialClassification notebook:
```bash
jupyter notebook CustomerPersonalityAnalysis.ipynb
```

Thank you for your time!




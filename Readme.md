# Affect recognition in muscular response signals
The repository belongs to paper that investigates the potential of recognising arousal in motor activity. 
We formulate arousal detection as a statistical problem of separating two sets - motor activity under emotional arousal and motor activity without arousal. 
We establish and apply a hypothesis testing regime for machine learning classifiers to distinguish between the two sets of motor activity. Our proposed test
procedure assumes that the two groups can be significantly distinguished if the classifiers perform better than random guessing. 
We apply repeated corrected cross-validation t-test to verify that the models perform better than random guessing.
The classification models are evaluated based on accuracy and Matthew’s correlation coefficient. The repeated corrected cross-validation t-test tests whether accuracy is
significantly greater than 50%.

The repository contains scripts and functions for the preprocessing of the data and the computational experiments conducted. 
Our aim is to keep our research reproducable and transparent. 

## Data Access 
The data are accessible in the Open Science Framework (OSF) via the following [link](https://osf.io/txnqp/). 
All the files should be stored in the folder **data**, so that the provided scripts have the correctly assigned paths. 

## Installation
The projects uses **poetry** for dependency management. It is recommended to install poetry when using this repository. 
Instructions on how to install poetry can be found on their [webpage](https://python-poetry.org/docs/#installation). 

When poetry is successfully installed, it can be used to install the repository. 
```
cd emotion-extraction
poetry init
```



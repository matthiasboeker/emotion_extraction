# Affect recognition in muscular response signals
The repository belongs to paper that investigates the potential of recognising arousal in motor activity. 
We formulate arousal detection as a statistical problem of separating two sets - motor activity under emotional arousal and motor activity without arousal. 

The repository contains scripts and functions for the preprocessing of the data and the machine learning experiments conducted. 
Our aim is to keep our research reproducable and transparent. 

## Data Access 
The data are accessible in the Open Science Framework (OSF) via the following [link](https://osf.io/txnqp/). 
All the files should be stored in the folder **data**, so that the provided scripts have the correctly assigned paths. 

## Installation
The projects uses **poetry** for dependency management. Instructions on how to install poetry can be found on their [webpage](https://python-poetry.org/docs/#installation). 

When poetry is successfully installed, it can be used to install the repository. 
```
cd emotion-extraction
poetry init
```

## Usage 
The data is loaded, structured as objects and pickled with the script *emotion_extraction/pickle_objects.py*
The machine learning experiments can be run by the script  *emotion_extraction/ml_classification.py.py*




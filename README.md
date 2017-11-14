# Predicting Sales of Chain Stores 

> This repository contains all code, saved models and plots used in the CZ4032 Data Analytics and Mining course at NTU.

The Rossmann Store Sales problem is a [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales). The challenge requires participants to forecast sales of Rossmann over a period of 6 weeks given historical data of 1,115 stores located across Germany.

## Setup
All code for this project is written in [Python 3](https://www.python.org/downloads/). The list of dependencies can be found in `requirements.txt`. To set up your development environment, enter the following on a terminal.

```
$ pip install -r requirements.txt
```

The `data/` directory contains all data files downloaded from Kaggle. The `plots/` directory contains all plots generated in execution of the programs. All output files are recorded in the `predictions/` directory.

## Running

### Data Analysis
The first step is to analyze the datasets for relationships and trends. To generate all plots we used for data analysis, run the following:

```
$ python DataAnalysis/generate_plots.py
```

To generate more plots, please extend the code in `DataAnalysis/generate_plots.py`.

### Models
For each type of model, the codes can be found in the respective directories. The private leaderboard RMSPE score of each model is reported in the doc string of the python file. Description of the model and assumptions made can also be found in the python file. To run the codes, run

```
$ python ModelType/model-name.py
```

from the root of the repository.

The code will automatically generate any relevant post processing plots in the `plots/` directory and generate the output file in `predictions/`. 

The generated prediction csv files can be uploaded to Kaggle for RMSPE score on the public and private leaderboard. 

## Team

1. Suyash Lakhotia ([SuyashLakhotia](https://github.com/SuyashLakhotia))
2. Virat Chopra ([chopravirat](https://github.com/chopravirat))
3. Nikhil Venkatesh ([nikv96](https://github.com/nikv96))
4. Priyanshu Singh ([singhpriyanshu5](https://github.com/singhpriyanshu5))
5. Shantanu Kamath ([ShantanuKamath](https://github.com/ShantanuKamath))

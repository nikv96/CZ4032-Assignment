# Predicting Sales of Chain Stores 

> This repository contains all code, saved models and plots used in the CZ 4032 Data Analytics and Mining course at NTU.

The Rossmann Store Sales problem is a [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales). The challenge requires participants to forecast sales of Rossmann over a period of 6 weeks given historical data of 1,115 stores located across Germany.

## Team

CZ 4032 Group 7, AY 17/18 Semester 1

1. Nikhil Venkatesh ([nikv96](https://github.com/nikv96))
2. Suyash Lakhotia ([SuyashLakhotia](https://github.com/SuyashLakhotia))
3. Virat Chopra ([chopravirat](https://github.com/chopravirat))
4. Priyanshu Singh ([singhpriyanshu5](https://github.com/singhpriyanshu5))
5. Shantanu Kamath ([ShantanuKamath](https://github.com/ShantanuKamath))

## Setup

All code for this project is written in [Python 3](https://www.python.org/downloads/). The list of dependencies can be found in `requirements.txt`. To set up your development environment, navigate to this repository and run the following on a terminal:

```
$ pip install -r requirements.txt
```

The `data/` directory contains all data files downloaded from Kaggle. The `plots/` directory contains all plots generated and all output files are recorded in the `predictions/` directory.

## Running

### Data Analysis

The first step is to analyze the datasets for relationships and trends. To generate all plots we used for data analysis, run the following:

```
$ python DataAnalysis/generate_plots.py
```

To generate more plots, please extend the code in `DataAnalysis/generate_plots.py`.

### Models

For each type of model, the codes can be found in the respective directories. The private leaderboard RMSPE score of each model is reported in the docstring of the Python file. Description of the model and assumptions made can also be found in the Python file. To run the code, run the following from the root of the repository:

```
$ python ModelType/model-name.py
```

The code will automatically generate the output predictions in `predictions/`. The generated `.csv` file can be uploaded to Kaggle to get the private & public RMSPE scores.

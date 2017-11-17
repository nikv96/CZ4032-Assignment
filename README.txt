=============================================
|                                           |
|  CZ 4032 AY 17/18 Semester 1 - Group 07   |
|                                           |
=============================================

> This .zip file contains all the code, datasets and plots used for the CZ 4032 Data Analytics & Mining project as well as our group's report (Project_Report_Group_07.pdf) and presentation slides (Project_Slides_Group_07.pdf).


Team
=====================================

1. Suyash Lakhotia (U1423096J)
2. Nikhil Venkatesh (U1423078G) 
3. Virat Chopra (U1422252H)
4. Priyanshu Singh (U1422744C)
5. Shantanu Kamath (U1422577F)


Setup
=====================================

All the code for this project is written in Python 3 (https://www.python.org/downloads/). The list of dependencies can be found in `requirements.txt`. To set up your development environment, navigate to this folder and run the following on a terminal:

```
$ pip install -r requirements.txt
```

The `data/` directory contains all the data files downloaded from Kaggle. The `plots/` directory contains all plots generated and all output files are recorded in the `predictions/` directory.


Running
=====================================

Data Analysis
--------------------

To generate all the plots we used for data analysis, run the following:

```
$ python DataAnalysis/generate_plots.py
```

This will save the generated plots in the `plots/` directory with appropriate filenames that reflect the content of the plot.

Models
--------------------

For each type of model, the codes can be found in the respective directories. The private leaderboard RMSPE score of each model is reported in the doc string of the python file. Description of the model and assumptions made can also be found in the python file. To run the code, run the following from the root of the repository:

```
$ python ModelType/model-name.py
```

The code will automatically generate the output predictions in `predictions/`. The generated .csv file can be uploaded to Kaggle to get the private & public RMSPE scores.

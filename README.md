# Classification_problem_ML
binary classification problem given dirty data

This repo contains a machine learning model that aims to solve a binary classification problem given a "dirty" dataset.

I used excel in the beginning to merge the data and then separate each field using the semicolon as the delimiter, 
obtaining a cleaner version of the dataset. the new datasets are included in this repo as well.

The new dataset were then cleaned even more using python by deleting all rows that had no label value.
Some of these entries actually had a label value; however, the some of the data in the middle was missing,
which led the label value to be pushed to the left a cell or two, maybe more. Probably the solution to this problem would be
somehow pushing the label value a couple of cells to its original place and then substitute for the cell that caused the problem,
but there was no apparent way to do this. The label 'yes' was mapped for value 1 and 'no' for value 0. I also tried to upscale the 
poorly represented label ('no') to have a more balanced data set, but that did not change results much in this case.

In addition; I transformed any other number given as text to its equivalent value.
To deal with the rest of the categorical data, I used one hot encoding so as to represent all values and not give any certain value
higher priority.

I also deleted one of the columns to the right as it had a huge number of NaN values. an improvement in the results came about this
decision.

the model used is the multi-layer nueral network model with 2 hidden layers, 10 neurons each. The choice of the size of the layers
is arbitrary, however; it showed the best result for different metrics after serveral trials. I used the stochastic gradient descent solver
with max iterations of 200. The choice of this model was mainly based on results as it showed higher results in accuracy, precision, recall
and f1-score metrics.

# Machine Learning 20: Practical work 3

> Simon Mirkovitch, Tiago Povoa Quinteiro 

## Questions: point 1 et 2

We answered the question in the notebook `Answers to questions.ipynb`

## Goal

> You will be provided with a database of vowels spoken by men, women and children (of 3, 5 and 7 years old). The task will be to train artificial neural networks to recognize the speaker having produced the given sounds and evaluate its performance (e.g., by crossvalidation). 

## Part 1

> 1. Man vs Woman. Use only the natural voices of men and women to train a neural network that recognizes the gender of the speaker. 

### In brief

* All the data is in a dataset (np.hstack)
* Labeled Female=1 and Male=-1
* Two features: median and std. Probably we could use only one
* Normalized with MinMaxScaler from sklearn.preprocessing (between -1 and 1)
* Used the hyperbolic tangent as the activation function

### Find the correct configuration

In this section, we'll discuss how we found the correct number of epochs, learning rate and momentum.

#### Features

By testing both with one and two features, we didn't saw much difference. It could signify that we are overfitting. We'll see in the following sections that it doesn't look so bad though. We'd need to test deeper to compare.

#### Normalization

As we started, we had some issues because of misconfigurations.

We forgot to use normalization at first. Without it, we observed that the problem was way harder than it's supposed to be. It was really impressive how much difference it made by correctly providing our features to the training process.

#### Epochs

By running with default parameters `Learning_rate=0.1, Momentum=0.5`, it was clear that the problem didn't needed much epochs. From 200, we went way to to 120, then 80 and finally 65 as we tuned better and better our parameters.

#### Learning rate and Momentum

![png](./mkdown/1/output_21_1.png)

As we can see in the graphs above, we managed to bring the mean squared error pretty low. 

Our approach was to test various values of learning rate by steps of 0.001 and various momentum values by steps of 0.1. 

Here we can see the result of an execution with:

```python
LEARNING_RATE = 0.012 
MOMENTUM = 0.5 
```

#### Hidden neurons

In this section, we tested with the **k fold cross validation** how much neurons we should put in the hidden layer.

![png](./mkdown/1/output_34_0.png)

The more convincing was the configuration with 10 hidden neurons. 

![png](./mkdown/1/output_35_0.png)

In the graph above, we see that the **MSE** is also very low for 20 hidden neurons. 

### Results

#### Confusion Matrix

In our first experiments, we obtained the following confusion matrix:

```
[[35.  1.]
 [ 7. 29.]]
```

And with the final tuning we arrived at:

```
[[36.  0.]
 [ 2. 34.]]
```

- True positif 36.
- False negative 0.
- False positive 3.
- True negative 33.

Over 72 files read

#### F-Score

TODO: F1-Socre

## Part 2



## Part 3

### In brief

Since we have 3 categories, we couldn't use Female=1 and Male=-1.

This time we have: Female=[**1**, -1, -1], Male=[-1, **1**, -1], Kid=[-1, -1, **1**]

* We only used the natural voices.

We tried with two features and it was clearly not enough. Here is the confusion matrix as an example:

```
[[ 35.  31.  34.]
 [ 27.  36.  18.]
 [ 79.  26. 108.]]
```

* Four features: Median, std, minimum and maximum

* Normalized with MinMaxScaler from sklearn.preprocessing (between -1 and 1)
* Used the hyperbolic tangent as the activation function

### Find the correct configuration

#### Number of Epochs

This time, we went all the way to TODO 200 Epochs. It started to look good only after 150.

#### Learning rate and Momentum

TODO

#### Hidden neurons

TODO

### Results

#### Confusion Matrix

TODO

#### F-Score
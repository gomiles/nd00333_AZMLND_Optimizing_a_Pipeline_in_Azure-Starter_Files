# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The provided dataset (https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) contains information about bank costumers. The  column ```y``` indicates if the costumers agreed to a term deposit.

The best performing model was Voting Ensemble with an accuracy of ```91.8%``` this solution was found by the automl algorithm. The provided model (logistic regression) archived an accuracy of ```91.6%``` with the hyperparameter ```C``` tuned to ```0.037```.

## Scikit-learn Pipeline
In the Scikit-learn pipeline there is only one specific model used. This is specified in a standalone python script that does the training and validation. Also it gets hold of the data directly from the internet and does not need a dataset from Azure as an input. The train.py script logs the accuracy.
The algorithm used is a logistic regression. ```C``` was the hyperparameter that was tuned. To tune it, a random sampling was used. It's advantage is that is is faster than a whole grid search wile generating similar results. To speed the hyperparameter tuning even further a ealy termination policy was used. If the result from an experiment with a specific hyperparameter is performing 20% worse than the best result, training is stopped.

## AutoML
In an AutoML run multiple models with various parameters are evalutated and compared. To tailor the search to this specific problem the following AutoML config was used:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=dataset,
    label_column_name='y',
    n_cross_validations=5,
    compute_target=compute)
```
Experiment timeout limits the use of resources. Matching our problem, the task is set to classification and it is evaluated by the metric accuracy. The previously created dataset is given as training data with 'y' as column for the labels. For validation a 5 fold cross validation is chosen. For the excecution the previously created compute cluster is used.

The best model from AutoML is a Voting Ensemble containing nine different models whose outputs are combined. 
There is also data transformation applied:
![image](https://user-images.githubusercontent.com/56161454/157039037-ba4c9740-0316-4f50-9a3d-57aa9a6767d6.png)
One example model used in the Voting Ensamble is the XGBoost Classifier with an ensable weight of 0.13:
```
{
    "class_name": "XGBoostClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "booster": "gbtree",
        "colsample_bytree": 1,
        "eta": 0.05,
        "gamma": 0,
        "max_depth": 6,
        "max_leaves": 0,
        "n_estimators": 200,
        "objective": "reg:logistic",
        "reg_alpha": 0.625,
        "reg_lambda": 0.8333333333333334,
        "subsample": 0.8,
        "tree_method": "auto"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
```
The parameters shown above specify the used model. "booster" specifies that a tree based classifier is used. Basic parameters are e.g. "max_depth" which specifies how complex the model is allowed to become.
Explanation for the other parameters can be found here: https://xgboost.readthedocs.io/en/stable/parameter.html

## Pipeline comparison
The performance of the model from the AutoML pipeline is ```0.2%``` better then the Scikit-learn pipeline. In most cases this can be expected since the AutoML compares multiple models and given enough resources should be able to come up with a better solution than a hand tuned model.
From the architecture perspective one difference is, that the data is loaded centrally and needs to be distributed to the different automl experiments. This requires the data to be published as a dataset.

## Future work
To have a better comparison of different models it would be benefitial to also look at other metrics and curves. Especially the performance of a model from the AutoML can be further improved by giving more resources. This can be done by increasing the number of ```iterations``` and ```experiment_timeout_hours```.

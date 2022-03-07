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
The provided dataset (https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) contains information about employees. There is one column (y) that can be predicted from the other information.

The best performing model was Voting Ensemble with an accuracy of 0.918 this solution was found by the automl algorithm. The provided model (logistic regression) archived a accuracy of 0.916 with the hyperparameter C tuned to 0.037.

## Scikit-learn Pipeline
In the Scikit-learn pipeline there is only one specific model used. This is specified in a standalone python script that does the training and validation. Also it gets hold of the data directly from the internet and does not need a dataset from Azure as an input. The train.py script logs the accuracy.
The algorithm used is a logistic regression. "C" was the hyperparameter that was tuned. To tune it, a random sampling was used. It's advantage is that is is faster than a whole grid search wile generating similar results. To speed the hyperparameter tuning even further a ealy termination policy was used. If the result from an experiment with a specific hyperparameter is performing 20% worse than the best result, training is stopped.

## AutoML
The best model from AutoML is a Voting Ensemble. Using the XGBoost Classifier with an ensable weight of 0.13:
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

## Pipeline comparison
From the architecture perspective one difference is, that the data is loaded centrally and needs to be distributed to the different automl experiments. This requires the data to be published as a dataset. In AutoML there is no model or hyperparameter specified, multiple algorithms are tested that fit the type of problem (classification in this case).

## Future work
To have a better comparison of different models it would be benefitial to also look at other metrics.

## Proof of cluster clean up
![image](https://user-images.githubusercontent.com/56161454/157013491-57cdc02a-a14f-434e-b68a-2e88c1e86f68.png)

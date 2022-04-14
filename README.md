# Evaluating Membership Inference Through Adversarial Robustness

This repository is the official implementation of [Evaluating Membership Inference Through Adversarial Robustness]. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Victim model

To train victim model:

```
Get_Test_Model.py
```

> 📋  you could change dataset and victim model by commenting directly on the code.

## Inference strategy I_{dd}

To get directional distance with different T and \lambda:

```
robust_sphere.py
```

To evaluate the victim model though I_{dd}:

```
eval_sphere.py
```

## PGD-AT trained model and membership inference:

To get PGD-AT trained model and evaluate its privacy though difference traditional metric based membership inference methods:

```
AT_train_test.py
```

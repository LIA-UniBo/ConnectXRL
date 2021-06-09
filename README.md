# ConnectXRL

The problem is taken from [this Kaggle competition](https://www.kaggle.com/c/connectx). The goal of the competition is to create an agent able to play at the famous game connect-4, here generalized to an arbitrary number X. We will use this environment to study and discuss some related interested topics.

## Setup

Easily, creating a Python 3 virtual environment, activating it and running the command

```
pip install -r requirements.txt
```

to install all the dependencies.

## Constraints in DRL

We are interested in exploring how constraints can be used in Deep Learning, in particular, in Deep Reinforcement Learning.

### Related code

./
├── connectxrl_constraints.ipynb (notebook presenting the constraints)
├── models (model weights)
└── src
    ├── connectx
    │   ├── constraints.py (constraints logic)
    │   ├── dqn.py (training logic)
    │   ├── environment.py (environment related code)
    │   ├── evaluate.py (code to test the agents)
    │   ├── __init__.py
    │   ├── opponents.py (some custom opponents)
    │   ├── plots.py (code to plot metrics)
    │   └── policy.py (neural networks and policy interfaces)
    ├── test
    │   ├── dqn.py (train an agent using DQN)
    │   ├── __init__.py
    │   └── play.py (play against a trained agent)
    └── __init__.py


### Bibliography

Deep Constrained Q-learning
Gabriel Kalweit and Maria Huegle and Moritz Werling and Joschka Boedecker.
2020
https://arxiv.org/abs/2003.09398


## Explainability in DRL

We are interested in exploring some recent advancements in the field of explainability and interpretability in Machine Learning and apply them in the context of Deep Reinforcement Learning models.

### Bibliography

Interpretable machine learning. A Guide for Making Black Box Models Explainable
Molnar Christoph
2019
https://christophm.github.io/interpretable-ml-book

Reconstructing Actions To Explain Deep Reinforcement Learning
Xuan Chen and Zifan Wang and Yucai Fan and Bonan Jin and Piotr Mardziel and Carlee Joe-Wong and Anupam Datta
2021
https://arxiv.org/abs/2009.08507

Explainable Reinforcement Learning: A Survey
Erika Puiutta and Eric MSP Veith
2020
https://arxiv.org/abs/2005.06247

Explain Your Move: Understanding Agent Actions Using Specific and Relevant Feature Attribution
Nikaash Puri and Sukriti Verma and Piyush Gupta and Dhruv Kayastha and Shripad Deshmukh and Balaji Krishnamurthy and Sameer Singh
2020
https://arxiv.org/abs/1912.12191

Visualizing and understanding atari agents
Sam Greydanus and Anurag Koul and Jonathan Dodge and Alan Fern
2017
https://arxiv.org/pdf/1711.00138

from typing import Optional, List

import shap
import torch
from shap import Explainer
from torch import Tensor

from src.connectx.policy import Policy

"""
A Unified Approach to Interpreting Model Predictions
https://arxiv.org/abs/1705.07874

SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction.
Usually is slower than LIME, but it's not a local approximation as LIME and support more models. LIME is a subset of
SHAP.

It aims to unify multiple model explanations methods. It is based on game theory, a shapley value is the average 
contribution of features which are predicting in different situation. In this scenario the game is assigning a credit
reward to each contributor (feature) of the model.

Shapley values are:
- additive (their sum is the final game result)
- consistent (the more a feature is important the bigger is its Shapley value)

Traditional Shapely values can be represented as money to be fairly split among some players, the way they are computed 
for a specific player is playing the game with every possible combinations of the other players and observe the money
difference between these subsets with the specific player and without it. The average of all these differences are the
SHAP values. In ML the game is the model, the players the features and the features belong to a specific instance being
explained.

- How to run every possible subset (exponential)?
- How to run a model where part of the features are missing?


TreeExplainer: The most powerful which computes exact valeus. Can be used for tree-based solutions such as XGBoost, 
CatBoost, LightGBM, RandomForests, ... All subsets can be computed in polynomial time.
DeepExplainer: For neural network models by combining DeepLIFT and Shapley values. Computes SHAP values for small 
portions of the network than the DeepLIFT is used.
KernelExplainer: It is connected with LIME. LIME linearly approximate in a specific area of the distribution using a 
kernel to describe the locality which is an hyperparameter, if it is set to the SHAP kernel LIME computed shap values.
It is model agnostic and gives approximation because it does not compute every combination but it runs for a limited 
time. The missing features are replaced from a background dataset. Choosing the background dataset is not easy.
GradientExplainer: For neural network models-

SHAP pinpoints which features are most impactful for a each sample, providing a quantified impact on the same scale of 
the target.
"""


def explain(policy: Policy,
            states_training: Optional[List[Tensor]],
            states_test: Optional[List[Tensor]],
            explainer: Optional[Explainer] = None) -> Explainer:
    """

    :param policy: the policy involved
    :param states_training: a list with tensors containing the sequence of states seen in each match used to train the
    explainer. If not passed only test is performed.
    :param states_test: a list with with tensors containing the sequence of states seen in each match used to test the
    explainer. If not passed only train is performed.
    :param explainer: an already prepared SHAP explainer. If train is not passed this must be passed.
    :return: the shapley values associated to each action for each state seen in the matches
    """

    if explainer is None and states_training is None:
        raise ValueError('You must either pass an explainer or a recording to create a new one!')

    # If the input is a list, the states are concatenated
    if states_training is not None and type(states_training) == list:
        states_training = torch.cat(states_training)
    if states_test is not None and type(states_test) == list:
        states_test = torch.cat(states_test)

    # Create explainer
    if states_training is not None:
        explainer = shap.DeepExplainer(policy, states_training)

    if states_test is not None:
        # Get a list with the length of possible actions, containing the shapley values as Numpy arrays for each
        # encountered state.
        shap_values = explainer.shap_values(states_test)

        # The Numpy arrays are transformed to PyTorch's tensors.
        # shap_values = list(map(lambda x: torch.from_numpy(x), shap_values))

        shap.image_plot([sv.transpose([0, 2, 3, 1]) for sv in shap_values],
                        states_test.data.numpy().transpose([0, 2, 3, 1]))

    return explainer

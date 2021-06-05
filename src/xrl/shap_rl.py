from typing import List

import shap
import torch
from src.connectx.policy import Policy


def explain(policy: Policy,
            state_recording: list,
            action_recording: list) -> List[torch.Tensor]:
    """

    :param policy: the policy involved
    :param state_recording: a list with tensors containing the states seen during each match
    :param action_recording: TODO
    :return: the shapley values associated to each action for each state seen during the matches
    """

    # If the input is a list, the states are concatenated
    if type(state_recording) == type(action_recording) == list:
        state_recording = torch.cat(state_recording)

    # Create explainer
    explainer = shap.DeepExplainer(policy, state_recording)
    shap_values = explainer.shap_values(state_recording)

    # A list with the size of possible actions containing as Numpy arrays the shapley values for each state. Here the
    # Numpy array is transformed to torch.
    shap_values = list(map(lambda x: torch.from_numpy(x), shap_values))

    return shap_values

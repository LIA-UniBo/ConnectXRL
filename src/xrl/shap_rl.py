from typing import List

import shap
import torch
from src.connectx.policy import Policy


def explain(policy: Policy,
            state_recording: list,
            state_test: list) -> List[torch.Tensor]:
    """

    :param policy: the policy involved
    :param state_recording: a list with tensors containing the sequence of states seen in each match used to train the
    explainer
    :param state_test: a list with with tensors containing the sequence of states seen in each match used to test the
    explainer
    TODO :param action_recording: a list with tensors containing the sequence of actions performed in each match
    :return: the shapley values associated to each action for each state seen in the matches
    """

    # If the input is a list, the states are concatenated
    if type(state_recording) == type(state_test) == list:
        state_recording = torch.cat(state_recording)
        state_test = torch.cat(state_test)

    # Create explainer
    explainer = shap.DeepExplainer(policy, state_recording)

    # Get a list with the length of possible actions, containing the shapley values as Numpy arrays for each encountered
    # state.
    shap_values = explainer.shap_values(state_test)

    # The Numpy arrays are transformed to PyTorch's tensors.
    # shap_values = list(map(lambda x: torch.from_numpy(x), shap_values))

    shap.image_plot([sv.transpose([0, 2, 3, 1]) for sv in shap_values],
                    state_test.data.numpy().transpose([0, 2, 3, 1]))

    return shap_values

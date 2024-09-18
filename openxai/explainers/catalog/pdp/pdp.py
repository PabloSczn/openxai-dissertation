import numpy as np
import torch
from ...api import BaseExplainer
from openxai.experiment_utils import convert_to_numpy

class PDP(BaseExplainer):
    """
    Partial Dependence Plot (PDP) Explainer.
    Computes feature importance by averaging model predictions over a grid of feature values.
    """

    def __init__(self, model, inputs: torch.FloatTensor, grid_resolution=100) -> None:
        """
        Initialize the PDP explainer.

        :param model: The model to explain.
        :param inputs: The input data as a torch.FloatTensor.
        :param grid_resolution: Number of points to evaluate for each feature.
        """
        super(PDP, self).__init__(model.predict)
        self.model = model
        self.inputs = inputs
        self.grid_resolution = grid_resolution

        # Compute feature value grids for each feature
        self.feature_grids = self._compute_feature_grids(inputs)

    def _compute_feature_grids(self, inputs):
        """
        Compute the grids of feature values for PDP.

        :param inputs: The input data.
        :return: A list of tensors, one for each feature.
        """
        feature_grids = []
        for i in range(inputs.shape[1]):
            min_val = inputs[:, i].min().item()
            max_val = inputs[:, i].max().item()
            grid = torch.linspace(min_val, max_val, steps=self.grid_resolution)
            feature_grids.append(grid)
        return feature_grids

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        """
        Compute the PDP explanations for the input x and summarize them into feature importance scores.
        
        :param x: The input data (not used in PDP).
        :param label: Not used in PDP.
        :return: A torch.FloatTensor containing the feature importance scores.
        """
        n_features = self.inputs.shape[1]
        importance_scores = []
        
        # Compute PDP for each feature
        for feature_idx in range(n_features):
            grid = self.feature_grids[feature_idx]
            pdp = self._compute_pdp_for_feature(feature_idx, grid)
            # Summarize PDP into a single importance score per feature
            importance = pdp.max() - pdp.min()  # Alternatively, use pdp.std() or pdp.var()
            importance_scores.append(importance.item())
        
        # Return the importance scores for each instance in x
        importance_scores = torch.FloatTensor(importance_scores)
        explanations = importance_scores.unsqueeze(0).repeat(x.size(0), 1)
        return explanations

    def _compute_pdp_for_feature(self, feature_idx, grid):
        """
        Compute PDP for a single feature.

        :param feature_idx: Index of the feature.
        :param grid: Grid of values for the feature.
        :return: PDP values for the feature.
        """
        # Prepare the dataset for predictions
        pdp_values = []
        inputs_copy = self.inputs.clone()
        for val in grid:
            inputs_copy[:, feature_idx] = val
            outputs = self.model(inputs_copy.float())
            # Assuming binary classification, we take the probability of the positive class
            preds = outputs[:, 1]
            pdp_values.append(preds.mean().item())
        return torch.tensor(pdp_values)
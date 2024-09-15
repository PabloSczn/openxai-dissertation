import numpy as np
import torch
from ...api import BaseExplainer
from openxai.experiment_utils import convert_to_numpy

class PFI(BaseExplainer):
    """
    Permutation Feature Importance Explainer.
    Computes feature importance by measuring the decrease in model performance when shuffling each feature.
    """

    def __init__(self, model, inputs: torch.FloatTensor, labels: torch.Tensor,
                 metric='accuracy', n_repeats=5, seed=None) -> None:
        """
        Initialize the PFI explainer.

        :param model: The model to explain.
        :param inputs: The input data as a torch.FloatTensor.
        :param labels: The true labels for the input data.
        :param metric: The performance metric to use (e.g., 'accuracy').
        :param n_repeats: Number of times to repeat the permutation for each feature.
        :param seed: Random seed for reproducibility.
        """
        super(PFI, self).__init__(model.predict)
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.metric = metric
        self.n_repeats = n_repeats
        self.seed = seed

        # Compute the baseline performance
        self.baseline_performance = self._compute_performance(inputs, labels)

    def _compute_performance(self, inputs, labels):
        """
        Compute the performance of the model on the given inputs and labels.

        :param inputs: The input data.
        :param labels: The true labels.
        :return: The performance score.
        """
        outputs = self.model(inputs.float())
        preds = outputs.argmax(dim=1)
        labels = labels.to(preds.device)
        if self.metric == 'accuracy':
            correct = preds.eq(labels).sum().item()
            total = labels.size(0)
            return correct / total
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented in PFI.")

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        """
        Compute the PFI explanations for the input x.

        :param x: The input data.
        :param label: Not used in PFI.
        :return: A torch.FloatTensor containing the feature importance scores.
        """
        # Since PFI is a global method, we will return the same importance scores for each instance in x
        n_features = self.inputs.shape[1]
        importance_scores = np.zeros(n_features)
        rng = np.random.default_rng(self.seed)

        for feature_idx in range(n_features):
            performance_decreases = []
            for _ in range(self.n_repeats):
                permuted_inputs = self.inputs.clone()
                permuted_feature = permuted_inputs[:, feature_idx]
                permuted_feature = permuted_feature[rng.permutation(permuted_feature.size(0))]
                permuted_inputs[:, feature_idx] = permuted_feature
                performance = self._compute_performance(permuted_inputs, self.labels)
                performance_decrease = self.baseline_performance - performance
                performance_decreases.append(performance_decrease)
            importance_scores[feature_idx] = np.mean(performance_decreases)

        # Return the importance scores for each instance in x
        importance_scores = torch.FloatTensor(importance_scores)
        explanations = importance_scores.unsqueeze(0).repeat(x.size(0), 1)
        return explanations

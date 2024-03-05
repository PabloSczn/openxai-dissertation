import numpy as np
import torch
from ...api import BaseExplainer
from openxai.experiment_utils import convert_to_numpy

# import lime
from .lime_package import lime_tabular
from .lime_package import lime_image


class LIME(BaseExplainer):
    """
    This class gets explanations for tabular data. The explanations are generated according to Ribeiro et al's tabular
    sampling algorithm.

    model : model.predict function
    data : np array
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data: torch.FloatTensor, std: float = 0.1,
                 n_samples: int = 1000, kernel_width: float = 0.75,
                 sample_around_instance: bool = True, mode: str = "tabular",
                 discretize_continuous: bool = False, seed=None) -> None:

        self.output_dim = 2
        self.data = data.numpy()
        self.mode = mode
        self.model = model.predict
        self.n_samples = n_samples
        self.discretize_continuous = discretize_continuous
        self.sample_around_instance = sample_around_instance
        self.seed = seed

        if self.mode == "tabular":
            self.explainer = lime_tabular.LimeTabularExplainer(
                self.data,
                mode="classification",
                sample_around_instance=self.sample_around_instance,
                discretize_continuous=self.discretize_continuous,
                kernel_width=kernel_width * np.sqrt(self.data.shape[1]),
                std=std
            )
        else:
            self.explainer = lime_image.LimeImageExplainer()

        super(LIME, self).__init__(model.predict)

    def get_explanation(self, all_data: torch.FloatTensor, label) -> torch.FloatTensor:
        label = convert_to_numpy(label)
        label = np.repeat(label, all_data.shape[0]) if label.shape == () else label

        if self.seed is not None:
            torch.manual_seed(self.seed); np.random.seed(self.seed)
        if self.mode == "tabular":
            all_data = all_data.numpy()
            num_features = all_data.shape[1]
            attribution_scores = np.zeros(all_data.shape)
            for i in range(all_data.shape[0]):
                exp = self.explainer.explain_instance(all_data[i, :], self.model,
                                                      num_samples=self.n_samples,
                                                      num_features=num_features)
                # bring explanations into data order (since LIME automatically orders according to highest importance)
                for feature_idx, feature_attribution in exp.local_exp[1]:
                    attribution_scores[i, feature_idx] = feature_attribution * (2*label[i]-1)
            return torch.FloatTensor(attribution_scores)
        else:
            attribution_scores = []
            for i in range(all_data.shape[0]):
                img = all_data  # .detach().numpy()
                # img = np.transpose(img, (1, 2, 0)).astype('double')

                # lime requires an image input size of (height, width, channels)
                # the classification wrapper within the cnn method needs to be adjusted accordingly
                # in this style: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

                exp = self.explainer.explain_instance(img,
                                                      self.model,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=self.n_samples)
                attribution_scores.append(exp)

            return torch.FloatTensor(attribution_scores)

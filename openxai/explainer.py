# Explainers
from openxai.explainers import Gradient, IntegratedGradients,\
    InputTimesGradient, SmoothGrad, LIME, SHAPExplainerC, RandomBaseline, PFI, PDP

explainers_dict = {
    'grad': Gradient,
    'sg': SmoothGrad,
    'itg': InputTimesGradient,
    'ig': IntegratedGradients,
    'shap': SHAPExplainerC,
    'lime': LIME,
    'control': RandomBaseline,
    'pfi': PFI,
    'pdp': PDP
}

def Explainer(method, model, param_dict={}):
    """
    Returns an explainer object for the given method
    :param method: str, name of the method
    :param model: PyTorch model or function
    :param param_dict: dict, __init__ parameters dictionary for the explainer
    :return: explainer object
    """
    if method not in explainers_dict.keys():
        raise NotImplementedError("This method has not been implemented, yet.")
    
    if method in ['lime', 'ig', 'pdp'] and param_dict == {}:
        raise ValueError(f"Please provide training data for {method} using param_dict = openxai.experiment_utils.fill_param_dict('{method}'" + ", {}, X_train)")
    
    explainer = explainers_dict[method](model, **param_dict)
    return explainer
# Utils
import os
import time
import numpy as np
import torch
import warnings; warnings.filterwarnings("ignore")
import openxai.experiment_utils as utils

# Models, Data, and Explainers
from openxai.model import LoadModel
from openxai.dataloader import ReturnTrainTestX, ReturnLoaders  # Make sure to import ReturnLoaders
from openxai.explainer import Explainer

if __name__ == '__main__':
    # Parameters
    config = utils.load_config('experiment_config.json')
    methods, n_test_samples = config['methods'], config['n_test_samples']
    param_strs = {method: utils.construct_param_string(config['explainers'][method]) for method in methods}

    # Generate explanations
    start_time = time.time()
    for data_name in config['data_names']:
        for model_name in config['model_names']:
            # Determine if pre-trained model is available
            pretrained = True
            if data_name == 'synthetic':
                pretrained = False  # Train from scratch for synthetic dataset

            # Make directory for explanations
            folder_name = f'explanations/{model_name}_{data_name}'
            utils.make_directory(folder_name)
            print(f"Data: {data_name}, Model: {model_name}")

            X_train, X_test, y_train, y_test = ReturnTrainTestX(data_name, n_test=n_test_samples, float_tensor=True, return_labels=True)
            model = LoadModel(data_name, model_name, pretrained=pretrained)
            predictions = model(X_test).argmax(dim=-1)

            # Load test labels
            trainloader, testloader = ReturnLoaders(data_name, download=False)
            test_labels = testloader.dataset.targets.to_numpy()
            test_labels = test_labels[:n_test_samples]

            # Loop over explanation methods
            for method in methods:
                print(f'Computing explanations for {method} (elapsed time: {time.time() - start_time:.2f}s)')
                param_dict = utils.fill_param_dict(method, config['explainers'][method], X_train)
                
                if method == 'pfi':
                    # PFI needs inputs and labels
                    param_dict['inputs'] = X_test
                    param_dict['labels'] = torch.tensor(y_test)
                
                explainer = Explainer(method, model, param_dict)
                explanations = explainer.get_explanations(X_test, predictions).detach().numpy()
                # Save explanations
                filename = f'explanations/{model_name}_{data_name}/{method}_{n_test_samples}{param_strs[method]}.npy'
                np.save(filename, explanations)

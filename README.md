# Ease.ml/Snoopy

[Ease.ml/Snoopy](https://www.ease.ml/snoopy) is a library to estimate the _best_ accuracy _any_ machine learning model can achieve on the given data distribution.
The core of [ease.ml/Snoopy](https://www.ease.ml/snoopy) is a powerfull, computationally efficient Bayes error rate (BER) estimator relying on pre-trained feature transformations available online.

### Supported Functionality

The library uses pre-trained embeddings and simple BER lower bound estimators to _guess_ the best possible accuracy any ML model can achieve. The system will return a binary signal by comparing the target error with the BER lower bound estimate (_feasible_ if above, _infeasible_ otherwise). Beeing aware of different failure modes (mainly false negatives), we also provide the convergence plots for additional insights. If the best transformation did not converge and the curve is likely to fall below the target accuracy with more data, users might want to collect more data and re-run to evaluation. On the other hand, if the curve did converge, users might want to try other pre-trained embeddings to bypass the possible bias introduced by using a transformation in the first place. An illustrative example is given in the jupyter notebook example below.

### Requirements and Installation

Run the script ```bash install.sh``` to create the conda environment with GPU support, and the script ```bash install-cpu.sh``` to install without GPU support. The dependencies will be automatically installed via ```conda```.

After successfully having created the conda environments, active either the environment ```snoopy``` (with GPU support) or ```snoopy-cpu``` (without GPU support), using ```conda activate``` and the corresponding environment name as additional argument.

## Quick Start

The [ease.ml/Snoopy](https://www.ease.ml/snoopy) library is containted in the directory ```snoopy```. We provide two examples on how to use the library: (1) inside a jupyter notebook, and (2) in a web application using a REST API.

### Example 1: Jupyter Notebook

The jupyter notebook ```ExampleUsage.ipynb``` contains a minimal example on how to use our library and how to interpret the results. The jupyter notebook needs to be started after activating the conda environment.

### Example 2: WebApp and REST API

The second example offers illustrates how to use [ease.ml/Snoopy](https://www.ease.ml/snoopy) in a simple web application connecting to a REST API via javascript.

- The REST API is using the [FastAPI](https://fastapi.tiangolo.com/) library. This library along with the [Uvicorn](https://www.uvicorn.org/) library need to be installed before starting the service. The service can be started running the script ```bash start_service.sh```.
- The service will use the functionality in the python file ```service.py```, whereby the functionality is very similar to example 1, where instaed of having everything chronolagically in a single python function, the functionality is spread across different functions.
- The website is contained in the html file ```snoopy.html```. The javascript for accessing the REST API are visible in the file ```assets/js/material-bootstrap-wizard.js```.

## Citations

```bibtex
% System 
@article{renggli2020ease,
  title={Ease.ml/snoopy: Towards Automatic Feasibility Study for Machine Learning Applications},
  author={Renggli, Cedric and Rimanic, Luka and Kolar, Luka and Hollenstein, Nora and Wu, Wentao and Zhang, Ce},
  journal={arXiv preprint arXiv:2010.08410},
  year={2020}
}

% Theory on BER Estimator Convergence
@article{rimanic2020convergence, 
  title={On Convergence of Nearest Neighbor Classifiers over Feature Transformations}, 
  author={Rimanic, Luka and Renggli, Cedric and Li, Bo and Zhang, Ce}, 
  journal={Advances in Neural Information Processing Systems}, 
  volume={33}, 
  year={2020} 
}

% Framework for BER Estimator Comparison
@article{renggli2021evaluating, 
  title={Evaluating Bayes Error Estimators on Real-World Datasets with FeeBee},
  author={Renggli, Cedric and Rimanic, Luka and Hollenstein, Nora and Zhang, Ce},
  journal={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  volume={34},
  year={2021}
}

% Demo
@article{renggli2020ease,
  title={Ease.ml/snoopy in action: Towards automatic feasibility analysis for machine learning application development}, 
  author={Renggli, Cedric and Rimanic, Luka and Kolar, Luka and Wu, Wentao and Zhang, Ce}, 
  journal={Proceedings of the VLDB Endowment}, 
  volume={13}, 
  year={2020}
}
```

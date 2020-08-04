# Probabilistic Inference for Learning Control (PILCO)
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

This folder contains an implementation of the paper `Extended Radial Basis Function Controller for Reinforcement Learning` submitted to the 4th Conference on Robot Learning (CoRL 2020).

The implementation is based on the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) framework (as per Section 2 Method in the paper) and written in `Python 3`. 
`TensorFlow v2` and [`GPflow v2`](https://github.com/GPflow/GPflow) have also been used for optimisation and Gaussian Process Regression respectively.

The rest of this document brings the reader through on how to set up the implementation and reproduce some of the results in the paper.

## Installation
1.Install venv
```bash
virtualenv -p python3 venv
source venv/bin/activate
```
2.Install requirements
```bash
pip install -r requirements.txt
python setup.py develop
```
3.You might also need to install openai gym
```bash
pip install gym
```

## Example of usage
Once these dependecies have been installed, one can run the code to reproduce the results presented in the paper. 

#### Main experiment
The main experiment of the paper trains an extended RBF controller that comprises of an engineered linear controller and a series of RBF controllers.
This could be done by running the python scripts corresponding to the three environments, under the `examples` folder. 

For instance, to obtain a trained controller and a plot similar to that of `Figure 4` in the paper, one could run
```
python examples/swing_up.py
```
Analogous commands could produce trained controllers in the `cartpole` or `mountain_car` environments.

#### Interaction time
Interaction time could be adjusted by toggling the timesteps `T` and number of epochs `N` parameters in each of the example scripts.

Interaction times shown in `Figure 5` are based on the extended RBF controller and RBF controller (currently commented out) run in the three different environments. 

#### Gain and Phase Margins
The gain and phase margins presented in `Table 2` are calculated based on the models set up in 
```
examples/matlab/run_margins.m
```

#### Further stability analysis
The data from the stability analysis in `Figure 6` could be obtained by running the following script
```
python tests/test_robust_analysis.py 
```
Please do note however that the script is currently set up to test an untrained controller. The user might want to test a trained controller by loading it using the function `load_controller_from_obj`.

## Credits:

This implementation is forked off an existing [PILCO repo](https://github.com/nrontsis/PILCO).

Credits also go out to OpenAI for and its [gym environments](https://github.com/openai/gym/tree/master/gym/envs/classic_control) for making testing possible.
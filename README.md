# Locally Enforced Optimal Controller (LEOC)

This repo contains an implementation of the paper `LEOC: A Principled Method in Integrating Reinforcement Learning and Classical Control Theory` submitted to the 3rd Annual Learning for Dynamics & Control Conference (L4DC 2021).

The implementation is based off the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) and [DDPG](https://arxiv.org/abs/1509.02971) frameworks and written in `Python 3`.
`TensorFlow v2`, [`Tensorflow Agents`]() and [`GPflow v2`](https://github.com/GPflow/GPflow) packages have also been used for optimisation and learning.

The rest of this document brings the reader through on how to set up the implementation and reproduce some of the results in the paper.

## Installation
1. Install venv
```bash
virtualenv -p python3 venv
source venv/bin/activate
```
2. Install requirements
```bash
pip install -r requirements.txt
python setup.py develop
```
<!-- 3. You might also need to install openai gym
```bash
pip install gym
``` -->

## File directory
Before we move on, we take a tour of the file directory.
The main training loop is called in `run.py`, configured by a config file in `data` and defined in `dao/trainer.py`.
In this loop, objects associated with the policies are imported from `DDPG` and `pilco`.
The main training loop would also save the trained models in `controllers`, and training rewards as byte stream in `pickle`.
Scripts to perform miscellaneous experiments on the trained policies are kept in `plotting`.

```
LEOC
│   README.md
│   LICENSE
│   requirements.txt
│   setup.py
│   run.py
│
└───DDPG  # files for implementing the DDPG network
│   └───...
│   
└───pilco  # files for implementing the PILCO framework
│   └───...
│   
└───dao  # util files for dependency injection
│   │   trainer.py
│   │   ...
│   │
│   └───envs  # environment files
│       │   cartpole_env.py
│       └───...
│       
└───data  # .gin configuration files
│   │   Cartpole_DDPG_Baseline.gin
│   └───...
│   
└───controllers  # saved trained controllers
│   └───...
│   
└───pickle  # saved training rewards
│   └───...
│   
└───plotting  # scripts for plotting graphs in the paper
│   │   plotter.py
│   └───...
│
└───resources  # resources for README.md
│   └───...
│
└───...
```

## Example of usage
Once the dependecies have been installed, one can run the code to train the policies and reproduce the results in the paper.

#### Training of policies
The experimental sections 5 & 6 of the paper require the trained baseline PILCO and DDPG controllers as well as their hybrid counterparts for subsequent analysis.

Since there are three environments, each of which sees a baseline and a hybrid policy in the PILCO and DDPG frameworks, in addition to a linear controller, there are therefore altogether `3 x (2 x 2 + 1) = 15` policies. Each of these policies is configured in a `.gin` file.

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGxpbmVhcjFbTGluZWFyXVxuICBlbnZbUGVuZHVsdW0gLyBDYXJ0UG9sZSAvIE1vdW50YWluQ2FyIGVudmlyb25tZW50XSAtLT4gYmFzZWxpbmUxYVtCYXNlbGluZV1cbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGh5YnJpZDFhW0h5YnJpZF1cbiAgc3ViZ3JhcGggUElMQ08xW1BJTENPXVxuICAgIGJhc2VsaW5lMWFcbiAgICBoeWJyaWQxYVxuICBlbmRcbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGJhc2VsaW5lMWJbQmFzZWxpbmVdXG4gIGVudltQZW5kdWx1bSAvIENhcnRQb2xlIC8gTW91bnRhaW5DYXIgZW52aXJvbm1lbnRdIC0tPiBoeWJyaWQxYltIeWJyaWRdXG4gIHN1YmdyYXBoIEREUEcxW0REUEddXG4gICAgYmFzZWxpbmUxYlxuICAgIGh5YnJpZDFiXG4gIGVuZCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0IiwidGhlbWVWYXJpYWJsZXMiOnsiYmFja2dyb3VuZCI6IndoaXRlIiwicHJpbWFyeUNvbG9yIjoiI0VDRUNGRiIsInNlY29uZGFyeUNvbG9yIjoiI2ZmZmZkZSIsInRlcnRpYXJ5Q29sb3IiOiJoc2woODAsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsInByaW1hcnlCb3JkZXJDb2xvciI6ImhzbCgyNDAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwic2Vjb25kYXJ5Qm9yZGVyQ29sb3IiOiJoc2woNjAsIDYwJSwgODMuNTI5NDExNzY0NyUpIiwidGVydGlhcnlCb3JkZXJDb2xvciI6ImhzbCg4MCwgNjAlLCA4Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5VGV4dENvbG9yIjoiIzEzMTMwMCIsInNlY29uZGFyeVRleHRDb2xvciI6IiMwMDAwMjEiLCJ0ZXJ0aWFyeVRleHRDb2xvciI6InJnYig5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSwgOS41MDAwMDAwMDAxKSIsImxpbmVDb2xvciI6IiMzMzMzMzMiLCJ0ZXh0Q29sb3IiOiIjMzMzIiwibWFpbkJrZyI6IiNFQ0VDRkYiLCJzZWNvbmRCa2ciOiIjZmZmZmRlIiwiYm9yZGVyMSI6IiM5MzcwREIiLCJib3JkZXIyIjoiI2FhYWEzMyIsImFycm93aGVhZENvbG9yIjoiIzMzMzMzMyIsImZvbnRGYW1pbHkiOiJcInRyZWJ1Y2hldCBtc1wiLCB2ZXJkYW5hLCBhcmlhbCIsImZvbnRTaXplIjoiMTZweCIsImxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJub2RlQmtnIjoiI0VDRUNGRiIsIm5vZGVCb3JkZXIiOiIjOTM3MERCIiwiY2x1c3RlckJrZyI6IiNmZmZmZGUiLCJjbHVzdGVyQm9yZGVyIjoiI2FhYWEzMyIsImRlZmF1bHRMaW5rQ29sb3IiOiIjMzMzMzMzIiwidGl0bGVDb2xvciI6IiMzMzMiLCJlZGdlTGFiZWxCYWNrZ3JvdW5kIjoiI2U4ZThlOCIsImFjdG9yQm9yZGVyIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwiYWN0b3JCa2ciOiIjRUNFQ0ZGIiwiYWN0b3JUZXh0Q29sb3IiOiJibGFjayIsImFjdG9yTGluZUNvbG9yIjoiZ3JleSIsInNpZ25hbENvbG9yIjoiIzMzMyIsInNpZ25hbFRleHRDb2xvciI6IiMzMzMiLCJsYWJlbEJveEJrZ0NvbG9yIjoiI0VDRUNGRiIsImxhYmVsQm94Qm9yZGVyQ29sb3IiOiJoc2woMjU5LjYyNjE2ODIyNDMsIDU5Ljc3NjUzNjMxMjglLCA4Ny45MDE5NjA3ODQzJSkiLCJsYWJlbFRleHRDb2xvciI6ImJsYWNrIiwibG9vcFRleHRDb2xvciI6ImJsYWNrIiwibm90ZUJvcmRlckNvbG9yIjoiI2FhYWEzMyIsIm5vdGVCa2dDb2xvciI6IiNmZmY1YWQiLCJub3RlVGV4dENvbG9yIjoiYmxhY2siLCJhY3RpdmF0aW9uQm9yZGVyQ29sb3IiOiIjNjY2IiwiYWN0aXZhdGlvbkJrZ0NvbG9yIjoiI2Y0ZjRmNCIsInNlcXVlbmNlTnVtYmVyQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvciI6InJnYmEoMTAyLCAxMDIsIDI1NSwgMC40OSkiLCJhbHRTZWN0aW9uQmtnQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvcjIiOiIjZmZmNDAwIiwidGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsInRhc2tCa2dDb2xvciI6IiM4YTkwZGQiLCJ0YXNrVGV4dExpZ2h0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0RGFya0NvbG9yIjoiYmxhY2siLCJ0YXNrVGV4dE91dHNpZGVDb2xvciI6ImJsYWNrIiwidGFza1RleHRDbGlja2FibGVDb2xvciI6IiMwMDMxNjMiLCJhY3RpdmVUYXNrQm9yZGVyQ29sb3IiOiIjNTM0ZmJjIiwiYWN0aXZlVGFza0JrZ0NvbG9yIjoiI2JmYzdmZiIsImdyaWRDb2xvciI6ImxpZ2h0Z3JleSIsImRvbmVUYXNrQmtnQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JvcmRlckNvbG9yIjoiZ3JleSIsImNyaXRCb3JkZXJDb2xvciI6IiNmZjg4ODgiLCJjcml0QmtnQ29sb3IiOiJyZWQiLCJ0b2RheUxpbmVDb2xvciI6InJlZCIsImxhYmVsQ29sb3IiOiJibGFjayIsImVycm9yQmtnQ29sb3IiOiIjNTUyMjIyIiwiZXJyb3JUZXh0Q29sb3IiOiIjNTUyMjIyIiwiY2xhc3NUZXh0IjoiIzEzMTMwMCIsImZpbGxUeXBlMCI6IiNFQ0VDRkYiLCJmaWxsVHlwZTEiOiIjZmZmZmRlIiwiZmlsbFR5cGUyIjoiaHNsKDMwNCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGUzIjoiaHNsKDEyNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU0IjoiaHNsKDE3NiwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU1IjoiaHNsKC00LCAxMDAlLCA5My41Mjk0MTE3NjQ3JSkiLCJmaWxsVHlwZTYiOiJoc2woOCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU3IjoiaHNsKDE4OCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIn19LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGxpbmVhcjFbTGluZWFyXVxuICBlbnZbUGVuZHVsdW0gLyBDYXJ0UG9sZSAvIE1vdW50YWluQ2FyIGVudmlyb25tZW50XSAtLT4gYmFzZWxpbmUxYVtCYXNlbGluZV1cbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGh5YnJpZDFhW0h5YnJpZF1cbiAgc3ViZ3JhcGggUElMQ08xW1BJTENPXVxuICAgIGJhc2VsaW5lMWFcbiAgICBoeWJyaWQxYVxuICBlbmRcbiAgZW52W1BlbmR1bHVtIC8gQ2FydFBvbGUgLyBNb3VudGFpbkNhciBlbnZpcm9ubWVudF0gLS0-IGJhc2VsaW5lMWJbQmFzZWxpbmVdXG4gIGVudltQZW5kdWx1bSAvIENhcnRQb2xlIC8gTW91bnRhaW5DYXIgZW52aXJvbm1lbnRdIC0tPiBoeWJyaWQxYltIeWJyaWRdXG4gIHN1YmdyYXBoIEREUEcxW0REUEddXG4gICAgYmFzZWxpbmUxYlxuICAgIGh5YnJpZDFiXG4gIGVuZCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0IiwidGhlbWVWYXJpYWJsZXMiOnsiYmFja2dyb3VuZCI6IndoaXRlIiwicHJpbWFyeUNvbG9yIjoiI0VDRUNGRiIsInNlY29uZGFyeUNvbG9yIjoiI2ZmZmZkZSIsInRlcnRpYXJ5Q29sb3IiOiJoc2woODAsIDEwMCUsIDk2LjI3NDUwOTgwMzklKSIsInByaW1hcnlCb3JkZXJDb2xvciI6ImhzbCgyNDAsIDYwJSwgODYuMjc0NTA5ODAzOSUpIiwic2Vjb25kYXJ5Qm9yZGVyQ29sb3IiOiJoc2woNjAsIDYwJSwgODMuNTI5NDExNzY0NyUpIiwidGVydGlhcnlCb3JkZXJDb2xvciI6ImhzbCg4MCwgNjAlLCA4Ni4yNzQ1MDk4MDM5JSkiLCJwcmltYXJ5VGV4dENvbG9yIjoiIzEzMTMwMCIsInNlY29uZGFyeVRleHRDb2xvciI6IiMwMDAwMjEiLCJ0ZXJ0aWFyeVRleHRDb2xvciI6InJnYig5LjUwMDAwMDAwMDEsIDkuNTAwMDAwMDAwMSwgOS41MDAwMDAwMDAxKSIsImxpbmVDb2xvciI6IiMzMzMzMzMiLCJ0ZXh0Q29sb3IiOiIjMzMzIiwibWFpbkJrZyI6IiNFQ0VDRkYiLCJzZWNvbmRCa2ciOiIjZmZmZmRlIiwiYm9yZGVyMSI6IiM5MzcwREIiLCJib3JkZXIyIjoiI2FhYWEzMyIsImFycm93aGVhZENvbG9yIjoiIzMzMzMzMyIsImZvbnRGYW1pbHkiOiJcInRyZWJ1Y2hldCBtc1wiLCB2ZXJkYW5hLCBhcmlhbCIsImZvbnRTaXplIjoiMTZweCIsImxhYmVsQmFja2dyb3VuZCI6IiNlOGU4ZTgiLCJub2RlQmtnIjoiI0VDRUNGRiIsIm5vZGVCb3JkZXIiOiIjOTM3MERCIiwiY2x1c3RlckJrZyI6IiNmZmZmZGUiLCJjbHVzdGVyQm9yZGVyIjoiI2FhYWEzMyIsImRlZmF1bHRMaW5rQ29sb3IiOiIjMzMzMzMzIiwidGl0bGVDb2xvciI6IiMzMzMiLCJlZGdlTGFiZWxCYWNrZ3JvdW5kIjoiI2U4ZThlOCIsImFjdG9yQm9yZGVyIjoiaHNsKDI1OS42MjYxNjgyMjQzLCA1OS43NzY1MzYzMTI4JSwgODcuOTAxOTYwNzg0MyUpIiwiYWN0b3JCa2ciOiIjRUNFQ0ZGIiwiYWN0b3JUZXh0Q29sb3IiOiJibGFjayIsImFjdG9yTGluZUNvbG9yIjoiZ3JleSIsInNpZ25hbENvbG9yIjoiIzMzMyIsInNpZ25hbFRleHRDb2xvciI6IiMzMzMiLCJsYWJlbEJveEJrZ0NvbG9yIjoiI0VDRUNGRiIsImxhYmVsQm94Qm9yZGVyQ29sb3IiOiJoc2woMjU5LjYyNjE2ODIyNDMsIDU5Ljc3NjUzNjMxMjglLCA4Ny45MDE5NjA3ODQzJSkiLCJsYWJlbFRleHRDb2xvciI6ImJsYWNrIiwibG9vcFRleHRDb2xvciI6ImJsYWNrIiwibm90ZUJvcmRlckNvbG9yIjoiI2FhYWEzMyIsIm5vdGVCa2dDb2xvciI6IiNmZmY1YWQiLCJub3RlVGV4dENvbG9yIjoiYmxhY2siLCJhY3RpdmF0aW9uQm9yZGVyQ29sb3IiOiIjNjY2IiwiYWN0aXZhdGlvbkJrZ0NvbG9yIjoiI2Y0ZjRmNCIsInNlcXVlbmNlTnVtYmVyQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvciI6InJnYmEoMTAyLCAxMDIsIDI1NSwgMC40OSkiLCJhbHRTZWN0aW9uQmtnQ29sb3IiOiJ3aGl0ZSIsInNlY3Rpb25Ca2dDb2xvcjIiOiIjZmZmNDAwIiwidGFza0JvcmRlckNvbG9yIjoiIzUzNGZiYyIsInRhc2tCa2dDb2xvciI6IiM4YTkwZGQiLCJ0YXNrVGV4dExpZ2h0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0Q29sb3IiOiJ3aGl0ZSIsInRhc2tUZXh0RGFya0NvbG9yIjoiYmxhY2siLCJ0YXNrVGV4dE91dHNpZGVDb2xvciI6ImJsYWNrIiwidGFza1RleHRDbGlja2FibGVDb2xvciI6IiMwMDMxNjMiLCJhY3RpdmVUYXNrQm9yZGVyQ29sb3IiOiIjNTM0ZmJjIiwiYWN0aXZlVGFza0JrZ0NvbG9yIjoiI2JmYzdmZiIsImdyaWRDb2xvciI6ImxpZ2h0Z3JleSIsImRvbmVUYXNrQmtnQ29sb3IiOiJsaWdodGdyZXkiLCJkb25lVGFza0JvcmRlckNvbG9yIjoiZ3JleSIsImNyaXRCb3JkZXJDb2xvciI6IiNmZjg4ODgiLCJjcml0QmtnQ29sb3IiOiJyZWQiLCJ0b2RheUxpbmVDb2xvciI6InJlZCIsImxhYmVsQ29sb3IiOiJibGFjayIsImVycm9yQmtnQ29sb3IiOiIjNTUyMjIyIiwiZXJyb3JUZXh0Q29sb3IiOiIjNTUyMjIyIiwiY2xhc3NUZXh0IjoiIzEzMTMwMCIsImZpbGxUeXBlMCI6IiNFQ0VDRkYiLCJmaWxsVHlwZTEiOiIjZmZmZmRlIiwiZmlsbFR5cGUyIjoiaHNsKDMwNCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGUzIjoiaHNsKDEyNCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIiwiZmlsbFR5cGU0IjoiaHNsKDE3NiwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU1IjoiaHNsKC00LCAxMDAlLCA5My41Mjk0MTE3NjQ3JSkiLCJmaWxsVHlwZTYiOiJoc2woOCwgMTAwJSwgOTYuMjc0NTA5ODAzOSUpIiwiZmlsbFR5cGU3IjoiaHNsKDE4OCwgMTAwJSwgOTMuNTI5NDExNzY0NyUpIn19LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)


Briefly, the **linear** controller is an engineered controller designed around the operating point. It can be viewed as a non-updating network with only one layer.

<img src="resources/architecture_linear.png" alt="drawing" width="400"/>

The PILCO and DDPG **baseline** policies are learnt, multi-layer networks.

<img src="resources/architecture_nonlinear.png" alt="drawing" width="400"/>

Finally, our LEOC **hybrid** policies integrate the linear and non-linear controllers.

<img src="resources/architecture.png" alt="drawing" width="600"/>

For more technical details, please refer to Section 4 of the paper.

Training a policy is easy. Simply run `run.py` in the root directory with the appropriate `.gin` config file. For instance, to obtain a trained DDPG baseline policy for CartPole, one could run
```
python3 run.py -file data/Cartpole_DDPG_Baseline.gin
```
Analogous commands to train other controllers could be run with the respective `.gin` configuration files.

#### Demos
The trained stable policies in each of the experimental environments would behave like the following:
<img src="resources/pendulum.gif" alt="drawing" width="400"/>
<img src="resources/cartpole.gif" alt="drawing" width="400"/>
<img src="resources/mountaincar.gif" alt="drawing" width="400"/>

#### Rewards curve: Figure 6
When the policies are trained, their training rewards have been dumped in the `pickle` folder. To visualise these rewards, run
```
python3 plotting/plot_learning_curve.py
```

#### Transient responses: Figure 7 & Table 1
One could obtain the impulse and step responses of the trained controllers. Note that the current setup requires a controller in each of the environments.
```
python3 plotting/plot_response.py
```
To output the metrics presented in Table 1, run
```
python3 plotting/compute_metrics.py
```

#### Robustness analysis: Figure 8
Finally, with multiple trained policies in each configuration, we could test their robustness.
```
python3 plotting/plot_robustness.py
```

## Credits:

The `pilco` folder of this implementation is forked off an existing [PILCO repo](https://github.com/nrontsis/PILCO). Similarly, the DDPG implementation relies heavily on [Tensorflow Agents](https://www.tensorflow.org/agents). Credits also go out to OpenAI for and its [gym environments](https://github.com/openai/gym/tree/master/gym/envs/classic_control) for making testing possible.
Neural network/architecture plots have been made with the amazing tools built at [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).

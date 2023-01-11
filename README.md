# Graph network simulators for rigid objects

Code and parameters to accompany the CoRL 2022 paper
**Graph Network Simulators can learn discontinuous, rigid contact dynamics** ([paper](https://openreview.net/pdf?id=rbIzq-I84i_))<br/>
_Kelsey R. Allen*, Tatiana Lopez-Guevara*, Yulia Rubanova, Kimberly Stachenfeld,
Alvaro Sanchez-Gonzalez, Peter Battaglia, Tobias Pfaff_

The code here provides an implementation of the Encode-Process-Decode
graph network architecture in jax, model weights for this architecture trained
on 256 trajectories of real cube tosses from [contact-nets](https://proceedings.mlr.press/v155/pfrommer21a.html), and an example of rolling out an example validation trajectory
from the ContactNets dataset. PLEASE BE AWARE the provided weights were trained
on the [Contact-Nets](https://proceedings.mlr.press/v155/pfrommer21a.html) data
and are unlikely to work for other datasets. Please retrain the weights using
the method discussed in the paper (injecting noise during training, training
without shape matching) if interested in using this code for a new dataset.

## Usage

### in a google colab
Open the [google colab](https://colab.research.google.com/github/deepmind/gnn_single_rigids/blob/master/demo_rollout.ipynb) and run all cells.

### with jupyter notebook / locally
To install the necessary requirements (run these commands from the directory
that you wish to clone `gnn_single_rigids` into):

```shell
git clone https://github.com/deepmind/gnn_single_rigids.git
python3 -m venv rigids_venv
source rigids_venv/bin/activate
pip install --upgrade pip
pip install -r ./gnn_single_rigids/requirements.txt
```
When done with this codebase, you can deactivate the virtual environment with `deactivate` from
the command line.

Additionally install jupyter notebook if not already installed with
`pip install notebook`

Change into your new directory:

```shell
cd gnn_single_rigids
```

Download the dataset and model weights from google cloud:

```shell
wget -O ./gns_params.pkl https://storage.googleapis.com/dm_gnn_single_rigids/gns_params.pkl
wget -O ./example_real_toss.pkl https://storage.googleapis.com/dm_gnn_single_rigids/example_real_toss.pkl
```

Now you should be ready to go! Open `demo_rollout.ipynb` inside
a jupyter notebook and run *from third cell* onwards.

## Citing this work

If you use this work, please cite the following paper
```
@misc{inversedesign_2022,
  title = {Graph Network Simulators can learn discontinuous, rigid contact dynamics},
  author = {Kelsey R. Allen and
               Tatiana Lopez{-}Guevara and
               Yulia Rubanova and
               Kimberly L. Stachenfeld and
               Alvaro Sanchez{-}Gonzalez and
               Peter W. Battaglia and
               Tobias Pfaff},
  journal = {Conference on Robot Learning},
  year = {2022},
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

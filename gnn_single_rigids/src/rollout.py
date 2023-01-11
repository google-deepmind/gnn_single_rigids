# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Rollout functions for producing a graph net simulator rollout."""
from typing import Any, Dict, Sequence, Callable, Mapping, Tuple

import haiku as hk
import jax
import jraph


HaikuModel = Callable[
    [jraph.GraphsTuple], Tuple[jraph.GraphsTuple, Mapping[str, Any]]
]


def _single_step(
    simulator_state: jraph.GraphsTuple,
    dynamics_fn: Callable[
        [jraph.GraphsTuple], jraph.GraphsTuple],
    forward_graph_fn: Callable[
        [jraph.GraphsTuple, jraph.GraphsTuple], jraph.GraphsTuple
    ],
    next_simulator_state: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
  """Rollout step."""
  # Compute the future for dynamics features, with a padded graph.
  simulator_state = jraph.pad_with_graphs(
      simulator_state,
      n_node=simulator_state.n_node.sum() + 1,
      n_edge=simulator_state.n_edge.sum() + 1,
      n_graph=simulator_state.n_node.shape[0] + 1)
  dynamics_output = dynamics_fn(simulator_state)
  simulator_state.nodes["p:world_position"] = dynamics_output.nodes[
      "p:world_position"]

  simulator_state = jraph.unpad_with_graphs(simulator_state)

  # Forward the predictions to build the next graph.
  return forward_graph_fn(simulator_state, next_simulator_state)


def _rollout(
    ground_truth_trajectory: Sequence[jraph.GraphsTuple],
    dynamics_fn: Callable[
        [jraph.GraphsTuple], jraph.GraphsTuple],
    forward_graph_fn: Callable[
        [jraph.GraphsTuple, jraph.GraphsTuple], jraph.GraphsTuple
    ],
) -> Sequence[jraph.GraphsTuple]:
  """Rolls out a model over a trajectory by feeding its own predictions."""

  output_sequence = [ground_truth_trajectory[0]]

  for next_simulator_state in ground_truth_trajectory[1:]:
    output = _single_step(
        output_sequence[-1],
        dynamics_fn=dynamics_fn,
        forward_graph_fn=forward_graph_fn,
        next_simulator_state=next_simulator_state)

    output_sequence.append(output)

  return output_sequence


def get_predicted_trajectory(input_trajectory: Sequence[jraph.GraphsTuple],
                             network_weights: Dict[Any, Any],
                             haiku_model_fn: Callable[[], HaikuModel],
                             forward_graph_fn) -> Sequence[jraph.GraphsTuple]:
  """Returns rollout trajectory given input trajectory and model information.

  Args:
    input_trajectory: a trajectory of jraph.GraphsTuples representing a
      sequence of states.
    network_weights: The learned simulator model parameters.
    haiku_model_fn: The haiku model function representing the learned simulator.
    forward_graph_fn:
  Returns:
    rollout_trajectory: The predicted trajectory based on the first entry of
      input_trajectory.
  """
  network_state = network_weights["state"]
  params = network_weights["params"]

  @hk.transform_with_state
  def forward(input_):
    model = haiku_model_fn()
    return model(input_)

  @jax.jit
  def dynamics_fn(input_):
    output, unused_network_state = forward.apply(params, network_state,
                                                 jax.random.PRNGKey(42), input_)
    return output

  return _rollout(
      input_trajectory,
      dynamics_fn=dynamics_fn,
      forward_graph_fn=forward_graph_fn)

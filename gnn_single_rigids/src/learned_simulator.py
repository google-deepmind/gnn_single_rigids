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

"""JAX implementation of Graph Networks Simulator.

JAX equivalent of:

https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/learned_simulator.py
"""

from typing import Any, Mapping

import haiku as hk
import jax.numpy as jnp
import jraph

from gnn_single_rigids.src import graph_network
from gnn_single_rigids.src import normalizers


def _euler_integrate_position(
    position_sequence: jnp.array, finite_diff_estimate: jnp.array
) -> jnp.array:
  """Integrates finite difference estimate to position (assuming dt=1)."""
  # Uses an Euler integrator to go from position(order=0), velocity(order=1)
  # or acceleration(order=2) to position, assuming dt=1 corresponding to
  # the size of the finite difference.
  previous_position = position_sequence[:, -1]
  previous_velocity = previous_position - position_sequence[:, -2]
  next_acceleration = finite_diff_estimate
  next_velocity = previous_velocity + next_acceleration
  next_position = previous_position + next_velocity
  return next_position


class LearnedSimulator(hk.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(self,
               *,
               graph_network_kwargs: Mapping[str, Any],
               flatten_features_fn=None,
               name="LearnedSimulator"):
    """Inits the model.

    Args:
      graph_network_kwargs: Keyword arguments to pass to the learned part of the
        graph network `model.EncodeProcessDecode`.
      flatten_features_fn: Function that takes the input graph and dataset
        metadata, and returns a graph where node and edge features are a single
        array of rank 2, and without global features. The function will be
        wrapped in a haiku module, which allows the flattening fn to instantiate
        its own variable normalizers.
      name: Name of the Haiku module.
    """
    super().__init__(name=name)
    self._graph_network_kwargs = graph_network_kwargs
    self._graph_network = None

    # Wrap flatten function in a Haiku module, so any haiku modules created
    # by the function are reused in case of multiple calls.
    self._flatten_features_fn = hk.to_module(flatten_features_fn)(
        name="flatten_features_fn")

  def _maybe_build_modules(self, input_graph: jraph.GraphsTuple):
    if self._graph_network is None:
      num_dimensions = input_graph.nodes["world_position"].shape[-1]
      self._graph_network = graph_network.EncodeProcessDecode(
          name="encode_process_decode",
          node_output_size=num_dimensions,
          **self._graph_network_kwargs)

      self._target_normalizer = normalizers.AccumulatedNormalizer(
          name="target_normalizer")

  def __call__(
      self, input_graph: jraph.GraphsTuple, padded_graph: bool = True
  ) -> jraph.GraphsTuple:
    self._maybe_build_modules(input_graph)

    flat_graphs_tuple = self._encoder_preprocessor(
        input_graph, padded_graph=padded_graph)
    normalized_prediction = self._graph_network(flat_graphs_tuple).nodes
    next_position = self._decoder_postprocessor(normalized_prediction,
                                                input_graph)

    return input_graph._replace(
        nodes={"p:world_position": next_position},
        edges={},
        globals={},
        senders=input_graph.senders[:0],
        receivers=input_graph.receivers[:0],
        n_edge=(input_graph.n_edge * 0),
    )

  def _encoder_preprocessor(
      self, input_graph: jraph.GraphsTuple, padded_graph: jraph.GraphsTuple
  ) -> jraph.GraphsTuple:
    graph_with_flat_features = self._flatten_features_fn(
        input_graph, is_padded_graph=padded_graph)

    return graph_with_flat_features

  def _decoder_postprocessor(
      self, normalized_prediction: jnp.array, input_graph: jraph.GraphsTuple
  ) -> jnp.array:
    position_sequence = input_graph.nodes["world_position"]

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    prediction = self._target_normalizer.inverse(normalized_prediction)

    new_position = _euler_integrate_position(position_sequence, prediction)
    return new_position

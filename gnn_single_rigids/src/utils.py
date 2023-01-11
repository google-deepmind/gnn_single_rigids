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

"""Functions for featurizing graph network and running dynamics."""
from typing import Tuple

import jax.numpy as jnp
import jraph
import numpy as np
import scipy.linalg
import scipy.spatial.transform
import tree

from gnn_single_rigids.src import normalizers


def flatten_features(
    input_graph: jraph.GraphsTuple,
    is_padded_graph: bool,
    floor_clamp_dist: float,
    floor_height: float = 0.0) -> jraph.GraphsTuple:
  """Returns GraphsTuple with a single array of features per node/edge type."""

  # Normalize the elements of the graph.
  normalizer = normalizers.GraphElementsNormalizer(
      template_graph=input_graph, is_padded_graph=is_padded_graph)
  output_nodes = {}
  output_edges = {}

  # Extract important features from the position_sequence.
  position_sequence = input_graph.nodes["world_position"]
  input_nodes = input_graph.nodes

  velocity_sequence = time_diff(position_sequence)  # Finite-difference.

  # Collect node features.
  node_feats = []

  # Normalized velocity sequence, flattening spatial axis.
  flat_velocity_sequence = jnp.reshape(velocity_sequence,
                                       [velocity_sequence.shape[0], -1])

  # Normalize velocity and add to features
  node_feats.append(
      normalizer.normalize_node_array("velocity", flat_velocity_sequence),)

  # External mask.
  node_feats.append(input_nodes["external_mask"][:, None].astype(jnp.float32))

  # Distance to the floor.
  floor_dist = input_nodes["world_position"][:, -1, 2:3]
  floor_dist = jnp.clip(floor_dist - floor_height, a_max=floor_clamp_dist)
  node_feats.append(normalizer.normalize_node_array("floor_dist", floor_dist))

  # Rest position
  mesh_position = input_nodes["mesh_position"]
  node_feats.append(
      normalizer.normalize_node_array("mesh_position", mesh_position),)

  # global position
  node_position = input_nodes["world_position"][:, -1]

  output_nodes = jnp.concatenate(node_feats, axis=-1)

  # mesh edges
  mesh_edge_feats = []

  # add relative edge distances + norm
  rel_dist = (
      node_position[input_graph.receivers] -
      node_position[input_graph.senders])
  mesh_edge_feats.append(
      normalizer.normalize_edge_array("rel_dist", rel_dist),)

  norm = safe_edge_norm(rel_dist, input_graph, is_padded_graph, keepdims=True)
  mesh_edge_feats.append(
      normalizer.normalize_edge_array("rel_dist_norm", norm))

  # add relative rest edge distances + norm
  rel_dist = (
      mesh_position[input_graph.receivers] -
      mesh_position[input_graph.senders])
  mesh_edge_feats.append(
      normalizer.normalize_edge_array("rest_dist", rel_dist))

  norm = safe_edge_norm(rel_dist, input_graph, is_padded_graph, keepdims=True)
  mesh_edge_feats.append(
      normalizer.normalize_edge_array("rest_dist_norm", norm))

  # flatten features for graph network
  output_edges = jnp.concatenate(mesh_edge_feats, axis=-1)

  return input_graph._replace(nodes=output_nodes, edges=output_edges)


def time_diff(input_sequence):
  """Returns time difference between successive timepoints."""
  return jnp.diff(input_sequence, axis=1)


def safe_edge_norm(
    array: jnp.array,
    graph: jraph.GraphsTuple,
    is_padded_graph: bool,
    keepdims=False,
) -> jnp.array:
  """Compute vector norm, preventing nans in padding elements."""
  # In the padding graph all edges are connected to the same node with the
  # same position, this means that when computing the norm of the relative
  # distances we end up with situations f(x) = norm(x-x). The gradient of this
  # function should be 0. However, when applying backprop the norm function is
  # not differentiable at zero. To avoid this, we simply add an epsilon to the
  # padding graph.
  if is_padded_graph:
    padding_mask = jraph.get_edge_padding_mask(graph)
    epsilon = 1e-8
    perturb = jnp.logical_not(padding_mask) * epsilon
    array += jnp.expand_dims(perturb, range(1, len(array.shape)))
  return jnp.linalg.norm(array, axis=-1, keepdims=keepdims)


def _shape_matching(
    x: jnp.array, x0: jnp.array
) -> Tuple[jnp.array, jnp.array]:
  """Calculates global transformation that best matches the rest shape [PBD]."""
  # compute the center of mass (assuming shape is symmetric)
  t0 = x0.mean(axis=0, keepdims=True)
  tx = x.mean(axis=0, keepdims=True)

  # get nodes centered at zero
  q = x0 - t0
  p = x - tx

  # solve the system to find best transformation that matches the rest shape
  mat_pq = np.dot(p.T, q)
  rx, _ = scipy.linalg.polar(mat_pq)

  # convert rotation to scipy transform
  rx_matx = scipy.spatial.transform.Rotation.from_matrix(rx)
  trans = tx - t0
  return trans, rx_matx


def forward_graph(
    graph_with_prediction: jraph.GraphsTuple,
    next_gt_graph: jraph.GraphsTuple,
    shape_matching_inference: bool = True,
) -> jraph.GraphsTuple:
  """Updates the graph with input predictions.

  Args:
    graph_with_prediction: GraphsTuple with predictions from network for
      updated node positions at next time-step.
    next_gt_graph: GraphsTuple representing the ground truth graph at the next
      time-step.
    shape_matching_inference: If set to true, uses shape matching to maintain
      object shape across time-step by finding best global object translation/
      rotation to maintain rest shapes and respect node position predictions.

  Returns:
    next_graph: GraphsTuple with updated node positions.
  """

  node_features = graph_with_prediction.nodes
  node_predictions_world_pos = node_features["p:world_position"]

  if shape_matching_inference:
    rest_pos = node_features["mesh_position"]
    center = jnp.mean(rest_pos, axis=0, keepdims=True)
    trans, rot = _shape_matching(node_predictions_world_pos, rest_pos - center)
    node_predictions_world_pos = rot.apply(rest_pos - center) + trans

  new_position_with_history = jnp.concatenate(
      [
          node_features["world_position"][:, 1:],
          node_predictions_world_pos[:, jnp.newaxis],
      ],
      axis=1,
  )

  # copy graph structure
  next_graph = tree.map_structure(lambda x: x, next_gt_graph)

  # update world positions
  next_graph.nodes["world_position"] = new_position_with_history
  return next_graph

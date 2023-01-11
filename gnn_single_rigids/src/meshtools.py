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

"""Meshtools for creating and manipulating meshes."""
from typing import NamedTuple, Tuple
import numpy as np


class Mesh(NamedTuple):
  """Mesh object.

  Attributes:
    verts: [num_vertices, num_dims] containing vertex positions for mesh nodes.
    faces: [num_faces, face_size] contains indices indices joining sets of
      vertices. Supports triangles (face_size=3) and quads(face_size=4).

  """
  verts: np.ndarray
  faces: np.ndarray


def make_xy_plane() -> Mesh:
  """Creates a unit plane in x/y."""
  verts = np.array([[0.5, -0.5, 0],
                    [-0.5, 0.5, 0],
                    [-0.5, -0.5, 0],
                    [0.5, 0.5, 0]])
  tris = np.array([[0, 1, 2], [0, 3, 1]])
  return Mesh(verts, tris)


def make_unit_box() -> Mesh:
  """Creates a unit box."""
  verts = np.array([[-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]])
  quads = np.array([[2, 3, 1, 0],
                    [4, 5, 7, 6],
                    [3, 7, 5, 1],
                    [4, 6, 2, 0],
                    [6, 7, 3, 2],
                    [1, 5, 4, 0]])
  return Mesh(verts, quads)


def triangulate(faces: np.array) -> np.array:
  """Splits quads into triangles."""
  if faces.shape[1] == 3:
    return faces
  elif faces.shape[1] == 4:
    return np.concatenate([faces[:, [0, 1, 3]],
                           faces[:, [1, 2, 3]]], axis=0)
  else:
    raise ValueError("only triangles and quads are supported")


def transform(mesh: Mesh, translate=(0, 0, 0), scale=(1, 1, 1)) -> Mesh:
  """Translates and scales mesh."""
  verts = mesh.verts
  verts = verts * np.array(scale)[None] + np.array(translate)[None]
  return Mesh(verts, mesh.faces)


def triangles_to_edges(faces: np.array) -> Tuple[np.array, np.array]:
  """Computes mesh edges from triangles."""
  # collect edges from triangles
  edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, 2::-2]],
                         axis=0)

  senders, receivers = np.moveaxis(edges, 1, 0)

  # create two-way connectivity
  return (np.concatenate([senders, receivers], axis=0),
          np.concatenate([receivers, senders], axis=0))

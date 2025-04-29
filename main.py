from typing import Callable
from PIL import Image
import math
import numpy as np


image = Image.open("img.jpg")
img_array = np.array(image)
points = [(100, 150), (110, 160), (200, 300), (210, 310), (500, 500)]

def dist(A: tuple[int, int], B: tuple[int, int]) -> float:
  return math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)

def RangeQuery(points: list, distFunc: Callable, Q: tuple[int, int], eps: int) -> list:
  neighbors = []
  for p in points:
    if distFunc(Q, p) <= eps:
      neighbors.append(p)
  return neighbors

def DBSCAN(points, distFunc, eps, minPts) -> dict:
  C = 0
  label = {}
  for P in points:
    if label.get(P): continue
    neighbors = RangeQuery(points, distFunc, P, eps)
    if len(neighbors) < minPts:
      label[P] = "noise"
      continue
    C = C + 1
    label[P] = C
    S = neighbors
    S.remove(P)
    for Q in S:
      if label.get(Q) == 'noise': label[Q] = C
      if label.get(P): continue
      label[Q] = C
      neighbors = RangeQuery(points, distFunc, Q, eps)
      if len(neighbors) >= minPts:
        S += neighbors
  return label

print(DBSCAN(points, dist, 300, 3))
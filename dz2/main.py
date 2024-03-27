from dataclasses import dataclass
from typing import List, Tuple, TypeAlias
from itertools import starmap
from matplotlib import pyplot as plt
import numpy as np

Point: TypeAlias = Tuple[float, float]

@dataclass
class MarkedPoint:
    x: Point
    marker: int


FIRST_CLASS_MARKER: int = 0
SECOND_CLASS_MARKER: int = 1

speed_k: float = 0.1


points: List[MarkedPoint] = [
    MarkedPoint((1,1), FIRST_CLASS_MARKER),
    MarkedPoint((1,2), FIRST_CLASS_MARKER),
    MarkedPoint((1,4), FIRST_CLASS_MARKER),
    MarkedPoint((2,3), SECOND_CLASS_MARKER),
    MarkedPoint((2,3), SECOND_CLASS_MARKER),
    MarkedPoint((2,4), SECOND_CLASS_MARKER),
]
weights: List[float] = [1, 1, 1]


def activation_function(h: float) -> int:
    if h <= 0:
        return FIRST_CLASS_MARKER
    else:
        return SECOND_CLASS_MARKER

def process(point: Point) -> int:
    global weights
    
    h = weights[0] * point[0] + weights[1] * point[1] + weights[-1]


    return activation_function(h)
        

def adjust_weights(input: Point, needed_output: int) -> bool:
    global weights

    real = process(input)
    if needed_output == real:
        return True

    global speed_k
    if needed_output == 1 and real == 0:
        k = speed_k
    else:
        k = -speed_k
    
    for i in range(len(input)):
        weights[i] += k * input[i]
    weights[-1] += k
    return False





points1 = list(map(
    lambda p: p.x, 
    filter(
        lambda p: p.marker == FIRST_CLASS_MARKER,
        points
    )
))
points2 = list(map(
    lambda p: p.x, 
    filter(
        lambda p: p.marker == SECOND_CLASS_MARKER,
        points
    )
))
x1 = list(map(lambda p: p[0], points1))
y1 = list(map(lambda p: p[1], points1))
print(list(points1), x1, y1)
x2 = list(map(lambda p: p[0], points2))
y2 = list(map(lambda p: p[1], points2))

def redraw():
    plt.clf()
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)

    linspace = np.linspace(-5, 5, 2)
    global weights
    if weights[1] == 0:
        plt.plot([weights[0] - weights[-1] for _ in linspace], linspace)
    else:
        plt.plot(linspace, (weights[-1] - linspace * weights[0]) / weights[1])
    plt.draw()
    
def print_equation():
    global weights
    if weights[1] == 0:
        print(f"x={weights[0] - weights[-1]}")
    else:
        print(f"y = {weights[-1]/weights[1]} + ({-weights[0] / weights[1]}) * x")
plt.ion()
redraw()





while True:
    is_optimal = True
    for marked_point in points:
        input("Press <Enter> to continue")
        is_optimal &= adjust_weights(marked_point.x, marked_point.marker)
        print_equation()
        redraw()
    if is_optimal:
        break



plt.ioff()
plt.show()

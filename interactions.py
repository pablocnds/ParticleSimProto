import numpy as np
import math

def distance_vector(v1, v2):
    """
    Vector v1->v2
    """
    return v2 - v1

def complex_force(p1, p2, SCALE=20):
    d_v = distance_vector(p1,p2)
    d = np.linalg.norm(d_v)
    return math.log2(d) * (20/(d*d)) * (d_v / d) * SCALE

def lennard_jones_force(p1, p2, SCALE=1):
    d_v = distance_vector(p1,p2)
    d = np.linalg.norm(d_v)
    d_u = d_v / d
    f = SCALE * (6 * (math.pow(1/d, 6) - 2)) / math.pow(d, 13)
    return -f * d_u


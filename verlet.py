import numpy as np
from math import log2, ceil, pow, sqrt
import interactions


class ArrayShapeException(Exception): pass
class InitialArgumentsException(Exception): pass


class MyArray:
    """
    Light wrapper for numpy array, inspired in c++ std vector mechanisms for efficiency.
    Can represent numeric arrays of values or 2d vectors.
    """
    def __init__(self, is_value_vector: bool, initial_data=[], data_type=np.float64) -> None:
        """
        - is_value_vector: False for flat array or True for array of 2d vectors
        """
        self.size = len(initial_data)
        allo_sz = 100
        if self.size != 0:
            allo_sz = 100 * int(pow(2, ceil(log2(ceil(self.size/100)))))

        self._array = np.array([0] if is_value_vector else [(0,0)], dtype=data_type)
        if is_value_vector:
            self._array.resize((allo_sz,2))
        else:
            self._array.resize(allo_sz)
        self.d = len(self._array.shape)

        for i,e in enumerate(initial_data):
            self._array[i] = e
        if not ( self.d == 1 or \
                (self.d == 2 and self._array.shape[1] == 2)):
            raise ArrayShapeException(f"Expected array shape is (n) or (n,2). Got {self._array.shape}.")
    
    def append(self, new_value):
        """
        Fast append.
        If used size surpasses allocated memory, it doubles it.
        """
        if self._array.size == self.size:
            if self.d == 1:
                self._array = np.resize(self._array, self._array.size * 2)
            else:
                self._array = np.resize(self._array, (self._array.size, 2))

        self._array[self.size] = new_value
        self.size += 1

    def get_copy(self):
        return MyArray(self.d == 2, initial_data=self._array[:self.size], data_type=self._array.dtype)

    def reset(self):
        """Sets all values to 0"""
        self._array.fill(0)

    def set_values(self, new_array):
        if self._array.shape != new_array._array.shape or self.size != new_array.size:
            raise ArrayShapeException("Cannot set values from differently shaped array")
        self._array = new_array._array

    def __getitem__(self, key):
        if type(key) == tuple:
            k = key[0]
        else:
            k = key
        if k >= self.size:
            raise IndexError("MyArray index out of bounds")
        return self._array[key]
    
    def __setitem__(self, key, value):
        if type(key) == tuple:
            k = key[0]
        else:
            k = key
        if k >= self.size:
            raise IndexError("MyArray index out of bounds")
        self._array[key] = value
    
    def __add__(self, value):
        if type(value) == float:
            x = self._array + value
        elif self.d == 1 and len(value._array.shape) == 2:
            x = self._array[:, None] + value._array
        elif self.d == 2 and len(self._array.shape) == 1:
            x = self._array + value._array[:, None]
        else:
            x = self._array + value._array
        return MyArray(is_value_vector=(self.d == 2 or value.d == 2), initial_data=x[:self.size], data_type=self._array.dtype)

    def __sub__(self, value):
        if type(value) == float:
            x = self._array - value
        elif self.d == 1 and len(value._array.shape) == 2:
            x = self._array[:, None] - value._array
        elif self.d == 2 and len(self._array.shape) == 1:
            x = self._array - value._array[:, None]
        else:
            x = self._array - value._array
        return MyArray(is_value_vector=(self.d == 2 or value.d == 2), initial_data=x[:self.size], data_type=self._array.dtype)

    def __mul__(self, value):
        if type(value) == float:
            x = self._array * value
        elif self.d == 1 and len(value._array.shape) == 2:
            x = self._array[:, None] * value._array
        elif self.d == 2 and len(value._array.shape) == 1:
            x = self._array * value._array[:, None]
        else:
            x = self._array * value._array
        return MyArray(is_value_vector=(self.d == 2 or value.d == 2), initial_data=x[:self.size], data_type=self._array.dtype)
    
    def __len__(self):
        return self.size
    
    def __str__(self) -> str:
        return str(self._array[:self.size])


class ParticleView:
    """
    Abstract view of a 2D particle. 
    The data of the particle is set and held in global arrays.
    Destruction of this object should be done carefully as the global data is not deleted.
    """
    def __init__(self, pos_x, pos_y, v_x, v_y, mass, global_pos_array: MyArray, global_vel_array: MyArray, global_a_array: MyArray, global_m_array: MyArray):
        # Assumes all global arrays have same size and are alligned
        self._gl_pos_arr = global_pos_array
        self._gl_v_arr = global_vel_array
        self._gl_a_arr = global_a_array
        self._gl_m_arr = global_m_array

        self.ind = self._gl_m_arr.size

        self._gl_pos_arr.append((pos_x,pos_y))
        self._gl_v_arr.append((v_x,v_y))
        self._gl_a_arr.append((0,0))
        self._gl_m_arr.append(mass)
        assert self._gl_pos_arr.size == self.ind + 1
        assert self._gl_a_arr.size == self._gl_pos_arr.size
        assert self._gl_m_arr.size == self._gl_pos_arr.size
        assert self._gl_v_arr.size == self._gl_pos_arr.size

    def get_pos(self):
        return self._gl_pos_arr[self.ind]
    
    def set_pos(self, value):
        self._gl_pos_arr[self.ind] = value
    
    def get_a(self):
        return self._gl_a_arr[self.ind]
    
    def set_a(self, value):
        self._gl_a_arr[self.ind] = value
    
    def get_m(self):
        return self._gl_m_arr[self.ind]
    
    def __getitem__(self, key):
        """
        Gets the x or y position of the particle.
        """
        return self._gl_pos_arr[self.ind, key]
    
    def __setitem__(self, key, value):
        """
        Sets the x or y position of the particle.
        """
        self._gl_pos_arr[self.ind, key] = value
    
    def __str__(self) -> str:
        return self.get_pos()


class VerletSystem:
    def __init__(self, delta_t, initial_positions=[], initial_velocities=[]):
        """
        - delta_t: the delta time for all next frames. Should always be constant.
        - intial_positions: list of tuples with initial position vectors.
        """
        if len(initial_positions) != len(initial_velocities):
            raise InitialArgumentsException("Initial positions and initial velocities have to be same length.")
        self.all_positions = MyArray(is_value_vector=True)
        self.all_accelerations = MyArray(is_value_vector=True)
        self.all_velocities = MyArray(is_value_vector=True)
        self.all_masses = MyArray(is_value_vector=False)
        self.particles = []
        self._delta_t = delta_t
        for p,v in zip(initial_positions, initial_velocities):
            self.create_particle(p[0],p[1],v[0],v[1])

        # For Simple Verlet Integration
        #self.all_positions_prev = self.all_positions.get_copy()

    def create_particle(self, posx, posy, vx, vy):
        p = ParticleView(posx, posy, vx, vy, 1, self.all_positions, self.all_velocities, self.all_accelerations, self.all_masses)
        self.particles.append(p)

    def simulate_step(self):
        self._calculate_movement_verlet()

    def _calculate_accelerations(self, positions):
        n = len(positions)
        forces = MyArray(True, [(0,0)]*n)
        for i in range(n):
            for j in range(i+1,n):
                f_vec = interactions.lennard_jones_force(positions[i],positions[j])
                forces[i] = forces[i] + f_vec
                forces[j] = forces[j] - f_vec
        return forces * self.all_masses

    def _calculate_movement_verlet(self):
        # Simple Verlet
        # new_pos = self.all_positions * 2. - self.all_positions_prev + \
        #          (self.all_masses * self.all_forces * self._delta_t * self._delta_t)
        # self.all_positions_prev.set_values(self.all_positions)

        # Velocity Verlet
        new_pos = self.all_positions + (self.all_velocities * self._delta_t) + \
                 (self.all_accelerations * self._delta_t * self._delta_t * 0.5)
        new_acc = self._calculate_accelerations(new_pos)
        new_vel = self.all_velocities + ((self.all_accelerations + new_acc) * self._delta_t * 0.5)

        self.all_positions.set_values(new_pos)
        self.all_velocities.set_values(new_vel)
        self.all_accelerations.set_values(new_acc)

    @staticmethod
    def basic_force_function(p1, p2):
        """
        For testing.
        It's not differentiable which means it can gain/lose energy over time.
        """
        d = VerletSystem.distance(p1[0], p1[1], p2[0], p2[1])
        if d < 1.:
            f = (d-1)*10
        elif d < 2.:
            f = (d-1)
        elif d < 6.:
            f = -(d-2)*0.25+1
        else:
            return np.array((0,0))
        f *= 100
        ab = np.array((p2[0] - p1[0], p2[1] - p1[1]))
        return f * ab / np.linalg.norm(ab)

    @staticmethod
    def distance(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return abs(sqrt(dx * dx + dy * dy))



if __name__ == "__main__":
    from graphicInterface import GraphicalInterface

    FPS=60

    graphics = GraphicalInterface((500,500), scale=100, set_fps=FPS)
    sys = VerletSystem(delta_t=(1/FPS), initial_positions=[(0.7,0),(-0.6,0),(0,0.7),(0,-0.6)], initial_velocities=[(0,0)]*4)

    graphics.nextFrame(sys.all_positions)
    while not graphics.isDone():
        sys.simulate_step()
        graphics.nextFrame(sys.all_positions)


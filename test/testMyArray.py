"""
Basic functional tests for MyArray class.
"""

import unittest
import sys
import os

from numpy import float64, int64

sys.path.append(os.path.join(sys.path[0], ".."))
from verlet import MyArray


class TestMyArray(unittest.TestCase):
    def test_initial_data(self):
        with self.assertRaises(ValueError):
            MyArray(["a","b"], data_type=int)
        a = MyArray([i for i in range(201)])
        self.assertEqual(a.size, 201)
        self.assertEqual(a._array.size, 400)
        self.assertTrue(type(a[5]) is float64)
        
        b = MyArray()
        self.assertEqual(b.size, 0)
        self.assertEqual(b._array.size, 100)

        c = MyArray([1], data_type=int64)
        self.assertEqual(c.size, 1)
        self.assertEqual(c._array.size, 100)
        self.assertTrue(type(c[0]) is int64)


    def test_append(self):
        a = MyArray()
        for i in range(110):
            a.append(i)
        self.assertEqual(a.size, 110)
        self.assertEqual(a._array.size, 200)
        self.assertTrue(all([i == a[i] for i in range(110)]))
        self.assertEqual(a[109], 109)
        with self.assertRaises(IndexError):
            a[110]


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
from utils import Utils

class UtilsTest(unittest.TestCase):
    
    def test_01_argmax(self):
        self.assertEqual(Utils.argmax(self, [0,0,1]), 2)
    
    def test_02_argmax(self):
        self.assertEqual(Utils.argmax(self, [0,1,0]), 1)

    def test_03_argmax(self):
        self.assertEqual(Utils.argmax(self, [1,0,0]), 0)

if __name__ == '__main__':
    unittest.main()
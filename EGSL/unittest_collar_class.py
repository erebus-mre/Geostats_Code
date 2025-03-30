import unittest
import pandas as pd
import numpy as np
from egsl import Collar  # Assuming egsl.py is in the EGSL directory

class TestCollar(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = {
            'DHID': ['BH1', 'BH2', 'BH3', 'BH4'],
            'X': [100, 150, 200, 250],
            'Y': [100, 150, 200, 250],
            'Z': [0, -10, -20, -30],
            'ROCK': ['A','B','A','C']
        }
        self.df = pd.DataFrame(self.data)
        self.collar = Collar(self.df, 'DHID', 'X', 'Y', 'Z')

    def test_init(self):
        # Test if the Collar object is initialized correctly
        self.assertEqual(self.collar.dhid, 'DHID')
        self.assertEqual(self.collar.xcoord, 'X')
        self.assertEqual(self.collar.ycoord, 'Y')
        self.assertEqual(self.collar.zcoord, 'Z')
        self.assertTrue(isinstance(self.collar, pd.DataFrame))
        self.assertTrue(isinstance(self.collar, Collar))
        self.assertEqual(len(self.collar), 4)

    def test_init_type_error(self):
        # Test if TypeError is raised when df is not a DataFrame
        with self.assertRaises(TypeError):
            Collar("not a dataframe", 'DHID', 'X', 'Y', 'Z')

    def test_init_value_error(self):
        # Test if ValueError is raised when required columns are missing
        with self.assertRaises(ValueError):
            Collar(self.df, 'DHID', 'X', 'Y', 'MISSING')

    def test_getitem_dataframe(self):
        # Test if __getitem__ returns a Collar object when a DataFrame is selected
        new_collar = self.collar[['DHID', 'X','Y','Z']]
        self.assertTrue(isinstance(new_collar, Collar))
        self.assertEqual(new_collar.dhid, 'DHID')
        self.assertEqual(new_collar.xcoord, 'X')
        self.assertEqual(new_collar.ycoord, 'Y')
        self.assertEqual(new_collar.zcoord, 'Z')

    def test_getitem_series(self):
        # Test if __getitem__ returns a Series when a single column is selected
        series = self.collar['DHID']
        self.assertTrue(isinstance(series, pd.Series))

    def test_alpha_to_int(self):
        # Test if alpha_to_int correctly converts alpha-numeric values to integers
        col_map = self.collar.alpha_to_int('ROCK')
        self.assertIn('ROCK_i', self.collar.columns)
        self.assertEqual(self.collar['ROCK_i'].dtype, np.int64)
        self.assertEqual(len(col_map), 3)
        self.assertEqual(self.collar['ROCK_i'].iloc[0],0)
        self.assertEqual(self.collar['ROCK_i'].iloc[1],1)
        self.assertEqual(self.collar['ROCK_i'].iloc[2],0)
        self.assertEqual(self.collar['ROCK_i'].iloc[3],2)

    def test_alpha_to_int_value_error(self):
        # Test if ValueError is raised when the column does not exist
        with self.assertRaises(ValueError):
            self.collar.alpha_to_int('MISSING')

    def test_numeric_col(self):
        # Test if numeric_col returns the correct numeric columns
        num_tab = self.collar.numeric_col()
        self.assertTrue(isinstance(num_tab, pd.DataFrame))
        self.assertEqual(len(num_tab), 3)
        self.assertIn('X', num_tab['Column'].values)
        self.assertIn('Y', num_tab['Column'].values)
        self.assertIn('Z', num_tab['Column'].values)

    def test_loc_plot(self):
        # Test if loc_plot returns an Axes object (basic check)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_ax = self.collar.loc_plot(ax)
        self.assertEqual(ax, returned_ax)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()

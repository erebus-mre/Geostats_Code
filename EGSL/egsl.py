import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
import os


class Grid:
    def __init__(self,griddef):
        
        self.xo = griddef[0][0]
        self.yo = griddef[0][1]
        self.zo = griddef[0][2]
        self.xs = griddef[1][0]
        self.ys = griddef[1][1]
        self.zs = griddef[1][2]
        self.xn = griddef[2][0]
        self.yn = griddef[2][1]
        self.zn = griddef[2][2]

        self.block_vol = self.xs*self.ys*self.zs
        self.n_blocks = self.xn*self.yn*self.zn

        self.data = {}

    def gsl_str(self):
        return_string = f'{self.xn} {self.xo:.3f} {self.xs:.3f}\n{self.yn} {self.yo:.3f} {self.ys:.3f}\n{self.zn} {self.zo:.3f} {self.zs:.3f}\n'
        return return_string    

    def add_data(self, col_name,col_data):
        if len(col_data) != self.n_blocks:
            print(f'Error: Data array length ({len(col_data)}) does not match grid size ({self.n_blocks})')
            pass
        elif col_data.ndim != 1:
                print(f'Error: Data array ({len(col_data)}) is not one dimensional')
                pass
        else:        
            self.data.update({col_name:col_data})

######################################################################################
class Collar(pd.DataFrame):
    """
    Represents collar data for boreholes, inheriting from pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the collar data.
        borehole_id (str): The name of the column representing the borehole ID.
        x (str): The name of the column representing the X coordinate.
        y (str): The name of the column representing the Y coordinate.
        z (str): The name of the column representing the Z coordinate.

    Raises:
        TypeError: If 'df' is not a pandas DataFrame.
        ValueError: If any of the specified column names (borehole_id, x, y, z) are not found in the DataFrame.
    """
    def __init__(self, df, dhid, x, y, z):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The 'df' argument must be a pandas DataFrame.")

        # Check if required columns exist
        required_columns = [dhid, x, y, z]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} are missing from the DataFrame.")

        super().__init__(df.copy())
        self.dhid = dhid
        self.xcoord = x
        self.ycoord = y
        self.zcoord = z

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame):
            new_collar = Collar(result, self.dhid, self.xcoord, self.ycoord, self.zcoord)
            # Copy any extra attributes from the original Collar object
            for attr_name in self.__dict__:
                if attr_name not in new_collar.__dict__:
                    setattr(new_collar, attr_name, getattr(self, attr_name))
            return new_collar
        else:
            return result
              
    def alpha_to_int(self,column):
        """Convert a column of alpha-numeric values to integers."""
        if column not in self.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        
        # Create a mapping dictionary
        unique_values = self[column].unique()
        col_map = {val: idx for idx, val in enumerate(unique_values)}
        
        # Map the values to integers
        self[column + "_i"] = self[column].map(col_map).astype(int)  # Add the new column directly to self
        return col_map  # Return the mapping dictionary for reference

    def numeric_col(self):
        num_list = self.select_dtypes(include=[np.number]).columns
        num_dict = {position+1:name for position,name in enumerate(num_list)}
        tab = pd.DataFrame(num_dict,index=[0]).T
        tab.columns =['Column']
        return tab
    
    def loc_plot(self,ax,**kwargs):
        """Plot collar locations."""
        ax.scatter(self[self.xcoord], self[self.ycoord], **kwargs)  
        ax.set_xlabel(self.xcoord)
        ax.set_ylabel(self.ycoord)
        ax.set_aspect('equal')
        return ax
        



    
    # def to_gsl(self,outfile):
    #     df2gsl(outfile,self.numeric_col())

######################################################################################
class Survey():
    def __init__(self, df, dhid, dip, azimuth, depth):
        self.df = df
        self.dhid = dhid
        self.dip = dip
        self.azimuth = azimuth
        self.depth = depth

    def map_dhid(self, dhid_map):
        self.df['dhid_n'] = self.df.DHID.map(dhid_map)

    def numeric_col(self):
        num_list = self.df.select_dtypes(include=[np.number]) 
        return num_list  

    def to_gsl(self,outfile):
        df2gsl(outfile,self.numeric_col())
        

class Interval(pd.DataFrame):
    def __init__(self, df, dhid, frmi, toi):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The 'df' argument must be a pandas DataFrame.")
        
        super().__init__(df.copy())  # Initialize the DataFrame part
        self.dhid = dhid
        self.frmi = frmi
        self.toi = toi

        # Force re-cast 'from' and 'to' columns to float64
        self[self.frmi] = self[self.frmi].astype(float)
        self[self.toi] = self[self.toi].astype(float)
   
    def check_gaps_overlaps(self):

        go_err = self._get_gaps_overlaps()
        n_gaps = len(go_err[go_err['misc'] > 0])
        n_overlaps = len(go_err[go_err['misc'] < 0])
        if n_gaps > 0:
            print(f'Error: {n_gaps} gaps found in the interval data')
        if n_overlaps > 0:    
            print(f'Error: {n_overlaps} overlaps found in the interval data')        
            


    def _get_gaps_overlaps(self):
        # Check if required columns exist
        if self.dhid not in self.columns or self.frmi not in self.columns or self.toi not in self.columns:
            raise ValueError(f"Columns {self.dhid}, {self.frmi}, or {self.toi} do not exist in the DataFrame.")

        # Check if required columns are numeric
        if not pd.api.types.is_numeric_dtype(self[self.frmi]) or not pd.api.types.is_numeric_dtype(self[self.toi]):
            raise TypeError(f"Columns {self.frmi} and {self.toi} must be numeric.")   
        
        try:
            self[self.frmi] = self[self.frmi].astype(float)
            self[self.toi] = self[self.toi].astype(float)
        except ValueError as e:
            print(f"TypeError during float conversion: {e}")
            print(f"Problematic 'from' values: {self[pd.to_numeric(self[self.frmi], errors='coerce').isnull()]}")
            print(f"Problematic 'to' values: {self[pd.to_numeric(self[self.toi], errors='coerce').isnull()]}")
            raise  # Re-raise the exception to stop execution
                
        # Check for gaps and overlaps
        tdf = self.sort_values(by=[self.dhid, self.frmi]).copy()
        tdf['next_dhid'] = tdf[self.dhid].shift(-1)
        tdf['next_from'] = tdf[self.frmi].shift(-1).astype(float) # Explicit cast to float
        tdf['misc'] = (tdf['next_from'] - tdf[self.toi]).astype(float) # Explicit cast to float
        go_err = tdf[(tdf[self.dhid] == tdf['next_dhid']) & (abs(tdf['misc']) > 0.001)].copy()
        return go_err

    def map_dhid(self, dhid_map):
        if not isinstance(dhid_map, dict):
            raise TypeError("dhid_map must be a dictionary")
        self['dhid_n'] = self[self.dhid].map(dhid_map)



    def numeric_col(self):
        num_list = self.select_dtypes(include=[np.number])
        return num_list

    def to_gsl(self, outfile):
        df = self[self.numeric_col().columns]
        df =df.fillna(-999)
        df2gsl(outfile, df)

    def gap_err(self):
        ge = self._get_gaps_overlaps()
        return ge[ge['misc'] > 0]
      
    def overlap_err(self):
        oe = self._get_gaps_overlaps()
        return oe[oe['misc'] < 0]
    
    def fill_gaps(self):
        """
        Fills gaps in the interval data.

        Modifies the Interval DataFrame in place.
        """
        gaps_overlaps = self._get_gaps_overlaps()
        gaps = gaps_overlaps[gaps_overlaps['misc'] > 0]
        if len(gaps) == 0:
            print("No gaps found.")
            return self

        filler = gaps[[self.dhid, self.toi, 'next_from']].copy() #added .copy()
        filler = filler.rename(columns={self.toi: self.frmi, 'next_from': self.toi})

        self._add_missing_columns(filler)
        self._concat_and_sort(filler)
        #print(len(self))
        return self

    def _add_missing_columns(self, filler):
        """Adds missing columns to the filler DataFrame."""
        diff = set(self.columns).symmetric_difference(set(filler.columns))
        for col in diff:
            filler[col] = np.nan


    def _concat_and_sort(self, filler):
        """Concatenates and sorts the DataFrame."""  
        self_copy = pd.concat([self, filler], ignore_index=True)
        self_copy.sort_values(by=[self.dhid, self.frmi], inplace=True)
        self_copy.reset_index(drop=True, inplace=True)
        self.loc[:,:] = self_copy.loc[:, :]  # Update the original DataFrame
        
      
        


class GSL:
    def __init__(self, exe_path, working_path):
        self.exe_path = exe_path
        self.working_path = os.getcwd() + '/' + working_path

        # Get a list of all files in the exe_path
        files = os.listdir(exe_path)
        self.exe_files = [f for f in files if f.endswith('.exe')]

        # Create working directory
        if not os.path.exists(self.working_path):
            # Create the directory
            os.makedirs(self.working_path)

    def run(self, exe_file, par_file = 'create'):
        """Run GSLIB executable with parameter file."""
        
        exe_path = self.exe_path + exe_file
        par_path = self.working_path + par_file
        #print(exe_path)
        #print(self.working_path)
        #print(par_file)
        
        try:    
            process = subprocess.Popen(
            exe_path,
            cwd=self.working_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            )

            # Create a new parameter file
            if par_file == 'create':
                process.stdin.write("\n")
            else:
                process.stdin.write(par_file + "\n") 

            process.stdin.flush()  # Important: flush the buffer
            process.stdin.close() #Close the pipe

            stdout, stderr = process.communicate()

            return_code = process.returncode  
            if return_code != 0:
                print(f"Error: Executable returned non-zero exit code: {return_code}")
            if stderr:
                print(f"Standard Error: \n {stderr}")

            if stdout:
                print(f"Standard Output: \n {stdout}")

            return  return_code

        except FileNotFoundError:
            print(f"Error: Executable not found at {exe_path}")
            return None, None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None, None     

def gsl2df(data_file):
    """Convert GSLIB Geo-EAS files to a pandas DataFrame for use with Python
    methods.
    data_file: dataframe
    """

    columns = []
    with open(data_file) as f:
        head = [next(f) for _ in range(2)]  # read first two lines
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns

        for icol in range(ncol):  # read over the column names
            head = next(f)
            columns.append(head.split()[0])

        data = np.loadtxt(f, skiprows=0)
        df = pd.DataFrame(data)
        df.columns = columns
        return df     

def df2gsl(data_file, df):
    """Convert pandas DataFrame to a GSLIB Geo-EAS file for use with GSLIB
    methods.
    data_file: file name
    df: dataframe
    """
    ncol = len(df.columns)
    nrow = len(df.index)

    with open(data_file, "w") as f:
        f.write(data_file + "\n")
        f.write(str(ncol) + "\n")

        for icol in range(ncol):
            f.write(df.columns[icol] + "\n")
        for irow in range(nrow):
            for icol in range(ncol):
                f.write(str(df.iloc[irow, icol]) + " ")
            f.write("\n")                    

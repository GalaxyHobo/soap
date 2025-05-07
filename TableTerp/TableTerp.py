import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import glob
import re
import os

def get_root_filename(file_path):
    """
    Extracts the root file name from a given file path.
    
    Parameters:
    file_path (str): The full file path.
    
    Returns:
    str: The root file name.
    """
    # Get the file name with extension
    file_name_with_ext = os.path.basename(file_path)
    
    # Split the file name and extension and get the root file name
    root_file_name = os.path.splitext(file_name_with_ext)[0]
    
    return root_file_name

def read_data(file_path):
    """
    Reads data from a text file and returns a pandas DataFrame.
    Also returns the number of dependent variables.
    """
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        num_dependent_vars = 1
        numdep_match = re.match(r'\s*NUMDEP\s*=\s*(\d+)\s*', first_line)
        if numdep_match:
            num_dependent_vars = int(numdep_match.group(1))
            df = pd.read_csv(file_path, sep='\s+', skiprows=1)
        else:
            df = pd.read_csv(file_path, sep='\s+')
    return df, num_dependent_vars

def create_interpolator(df, num_dependent_vars, bound_err=True):
    """
    Creates interpolators for the given DataFrame based on the number of columns and dependent variables.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data columns.
    num_dependent_vars (int): Number of dependent variables.
    
    Returns:
    list: List of interpolator functions.
    """
    columns = df.columns
    num_columns = len(columns)
    num_independent_vars = num_columns - num_dependent_vars

    if num_columns < 2 or num_independent_vars < 1:
        raise ValueError("Insufficient columns for interpolation.")

    interpolators = []

    for i in range(num_dependent_vars):
        if num_independent_vars == 1:
            x_col = columns[0]
            y_col = columns[num_independent_vars + i]
            interpolator = interp1d(df[x_col], df[y_col], kind='linear', bounds_error=bound_err, fill_value=None)
            interpolators.append(interpolator)
        elif num_independent_vars == 2:
            x_col = columns[0]
            y_col = columns[1]
            z_col = columns[num_independent_vars + i]
            grid_x = np.unique(df[x_col])
            grid_y = np.unique(df[y_col])
            grid_z = np.zeros((len(grid_x), len(grid_y)))
            for xi, x in enumerate(grid_x):
                for yi, y in enumerate(grid_y):
                    value = df[(df[x_col] == x) & (df[y_col] == y)][z_col].values
                    grid_z[xi, yi] = value[0] if value.size > 0 else np.nan
            interpolator = RegularGridInterpolator((grid_x, grid_y), grid_z, method='linear', bounds_error=bound_err, fill_value=None)
            interpolators.append(interpolator)
        elif num_independent_vars == 3:
            x_col = columns[0]
            y_col = columns[1]
            z_col = columns[2]
            w_col = columns[num_independent_vars + i]
            grid_x = np.unique(df[x_col])
            grid_y = np.unique(df[y_col])
            grid_z = np.unique(df[z_col])
            grid_w = np.zeros((len(grid_x), len(grid_y), len(grid_z)))
            for xi, x in enumerate(grid_x):
                for yi, y in enumerate(grid_y):
                    for zi, z in enumerate(grid_z):
                        values = df[(df[x_col] == x) & (df[y_col] == y) & (df[z_col] == z)][w_col].values
                        grid_w[xi, yi, zi] = values[0] if values.size > 0 else np.nan
            interpolator = RegularGridInterpolator((grid_x, grid_y, grid_z), grid_w, method='linear', bounds_error=bound_err, fill_value=None)
            interpolators.append(interpolator)
        else:
            raise ValueError("Unsupported number of columns for interpolation.")
    

    '''
    grid_values = [np.unique(df[col]) for col in columns[:num_independent_vars]]
    interpolators = []

    for i in range(num_dependent_vars):
        z_col = columns[num_independent_vars + i]
        values = df.pivot_table(index=columns[0], columns=columns[1], values=z_col).values
        interpolator = RegularGridInterpolator(grid_values, values, method='linear', bounds_error=bound_err, fill_value=None)
        interpolators.append(interpolator)
    '''
    return interpolators

def process_files(file_list, bound_err=True):
    """
    Reads multiple data files, creates interpolators, and stores them in a dictionary.
    
    Parameters:
    file_list (list): List of file paths to process.
    
    Returns:
    dict: Dictionary of interpolators keyed by file names.
    """
    interpolators = {}
    
    for file_path in file_list:
        df, num_dependent_vars = read_data(file_path)
        # Get base file name for use as table name
        table_name = get_root_filename(file_path)
        try:
            interpolators[table_name] = create_interpolator(df, num_dependent_vars, bound_err)
        except ValueError as e:
            print(f"File {file_path}: {e}")
    
    return interpolators

def interpolate_values(interpolators, table_name, *values_arrays):
    """
    Interpolates values using the interpolators for the specified file.
    
    Parameters:
    interpolators (dict): Dictionary of interpolators keyed by file names.
    table_name (str): table name to retrieve the interpolators.
    values_arrays (tuple of np.ndarray): Arrays of independent variable values to interpolate.
    
    Returns:
    list: List of interpolated values arrays.
    """
    table_interpolators = interpolators.get(table_name)
    
    if table_interpolators is None:
        print(f"No interpolators found for table {table_name}")
        return None
    
    interpolated_values = []
    for interpolator in table_interpolators:
        try:
            if len(values_arrays) == 1:
                interpolated_array = interpolator(values_arrays[0])
            else:
                points = np.column_stack(values_arrays)
                # print(f"Interpolating points: {points}")
                interpolated_array = interpolator(points)
            # print(f"Interpolated results: {interpolated_array}")
            interpolated_values.append(interpolated_array)
        except Exception as e:
            print(f"Interpolation error for table {table_name} with values {values_arrays}: {e}")
    ret_val = np.stack(interpolated_values, axis=-1)
    if ret_val.shape == (1,):
        return ret_val[0]
    if ret_val.shape == (1, 1):
        return ret_val[0, 0]
    return ret_val

'''
# Sample usage
# Get list of text files in the current directory
file_list = glob.glob('/Users/fs272/AeroProgs/ARADIA/TableTerp/*.apdat')

# Process the files and create interpolators
bound_err = True # If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used. Default is True.
interpolators = process_files(file_list, bound_err)
print(interpolators)
print()


print(" *** Interpolation Example: 1-D with one output ****")
# Example interpolation values
values = np.array([0, 5, 15, 25, 30])
print(f"values: {values}")
# Table name 
table_name = 'simple1d'
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated values for file {table_name}: {interpolated_values}")
print()


print(" *** Interpolation Example: 2-D with one output ****")
# Table name (program takes root file name as table anme)
table_name = 'polar1' 
# Example interpolation values
cl_values = np.array([0.025, 0.075, 0.75])
mach_values = np.array([0.5, 0.98, 1.95])
print(f"cl_values: {cl_values}")
print(f"mach_values: {mach_values}")
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, cl_values, mach_values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated CD values for table {table_name}: {interpolated_values}")
print()


print(" *** Interpolation Example: 2-D with two outputs ****")
# Table name 
table_name = 'milpower'
# Example interpolation values
alt_values = np.array([ -1000., 0., 2500., 30000.])
mach_values = np.array([    0., 0.,  0.45,   0.85])
print(f"alt_values: {alt_values}")
print(f"mach_values: {mach_values}")
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, alt_values, mach_values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated values [THRUST  FUEL_FLOW]: {table_name}: {interpolated_values}")
print('thrust AND fuel flow at first condition is:', interpolated_values[0])
print('where thrust at first condition is:', interpolated_values[0,0])
print('and fuel flow at first condition is:', interpolated_values[0,1])
print('thrust at ALL conditions is:', interpolated_values[:,0]) # returns first column
print('and fuel flow at ALL conditions is:', interpolated_values[:,1]) # returns second column
print('1.05*fuel flow at ALL conditions is:', 1.05 * interpolated_values[:,1]) # returns second column with factor applied
print()


print(" *** Interpolation Example: 3-D with three outputs ****")
# Table name 
table_name = 'biz_jet_engine'
# Example interpolation values
alt_values = np.array([ 0.0, 5000., 30000.])
mach_values = np.array([0.50, 0.50,   0.60])
pla_values = np.array([ 10,  90.,    89.])
print(f"alt_values: {alt_values}")
print(f"mach_values: {mach_values}")
print(f"pla_values: {pla_values}")
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, alt_values, mach_values, pla_values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated values [AUX  FNET  WF]: {table_name}: {interpolated_values}")


# Example: out of bounds
print(" *** Interpolation Example ****")
x_values = np.array([0])
y_values = np.array([15])
print(f"x_values: {x_values}")
print(f"y_values: {y_values}")
# Table name 
table_name = 'simple2d'
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, x_values, y_values)
# Print results
print(f"Interpolated values for file {table_name}: {interpolated_values}")
print()



# Sample usage
# Get list of tables files in the current directory
file_list = glob.glob('/Users/fs272/AeroProgs/ARADIA/TableTerp/kc135/*.apdat')

# Process the files and create interpolators
bound_err = True # If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used. Default is True.
interpolators = process_files(file_list, bound_err)
print(interpolators)
print()

print(" *** Interpolation Example: 1-D with one output ****")
# Example interpolation values
values = np.array([0.11, 0.415, 0.72])
print(f"values: {values}")
# Table name 
table_name = 'kc135_est_aoa'
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated values for file {table_name}: {interpolated_values}")
print()


print(" *** Interpolation Example: 2-D with one output ****")
# Table name (program takes root file name as table anme)
table_name = 'kc135_est_polar' 
# Example interpolation values
mach_values = np.array([0.65, 0.77, 0.82])
cl_values = np.array([0.0, 0.4, 0.35])
print(f"mach_values: {mach_values}")
print(f"cl_values: {cl_values}")
# Excercise interpolator for the table
interpolated_values = interpolate_values(interpolators, table_name, mach_values, cl_values)
# Print results
print('Returned values type:', type(interpolated_values))
print(f"Interpolated CD values for table {table_name}: {interpolated_values}")
print()
'''
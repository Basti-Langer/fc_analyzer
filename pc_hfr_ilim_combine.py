def combine_pc_hfr_ilim(cond,pattern,mea):
    import pandas as pd
    import os 
    import glob

    # Paths to the CSV files
    file1_paths = glob.glob(os.path.join(cond, '**/*pc-summary*.csv'), recursive=True)
    file2_paths = glob.glob(os.path.join(cond, '**/*RTO2*.csv'), recursive=True)

    # Ensure there is at least one file for each type
    if len(file1_paths) == 0 or len(file2_paths) == 0:
        print("Required files not found.")
        return

    # Use the first matching files for both types
    file1 = file1_paths[0]
    file2 = file2_paths[0]

    # Read the two CSV files
    df1 = pd.read_csv(file1, delimiter='\t')  # Adjust delimiter if needed
    df2 = pd.read_csv(file2, delimiter='\t')  # Adjust delimiter if needed

    # Select columns 3 and 8 (0-based indexing)
    columns_to_add = df2.iloc[:, [2, 7]]  # 3rd column is index 2, 8th column is index 7

    # Add the columns from the second file to the first file
    df1 = pd.concat([df1, columns_to_add], axis=1)

    # Create the output file path
    basename = os.path.basename(cond)
    output_file = os.path.join(cond, f'{pattern}{mea}_{basename}_pc_hfr_ilim-summary.csv')

    # Save the result to a new CSV file
    df1.to_csv(output_file, index=False, sep='\t')

    print(f'Combined file saved as {output_file}')          

def combine_pc_hfr_ilim_in_folder():

    import tkinter as tk
    from tkinter import filedialog
    from .paths import base_paths
    from .paths import mea_from_path

    # Create a GUI window to select directory
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    cond = filedialog.askdirectory(title="Select Directory")
    
    if not cond:
        print("No directory selected.")
        return
    pattern = base_paths()[3]
    mea = mea_from_path(cond)

    combine_pc_hfr_ilim(cond, pattern, mea)
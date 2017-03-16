import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import os
#plt.style.use('ggplot')
from BJH_function import BJH_calculation
from BJH_function import read_3flex

direct = str(os.getcwd())+'\\Data_N2_trixial\\'
#print(os.getcwd(),cwd)
filename = '1.xls'

my_file = read_3flex.file_3flex(filename,direct)
my_file.get_iso()
my_file.to_csv('newfile.csv')

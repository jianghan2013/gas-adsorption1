import xlrd
import numpy as np
import pandas as pd
import os

class file_3flex_co2():
    def __init__(self,filename,direct=None):
        # filename: input file name
        if direct is None:
            self.direct = os.getcwd() # use current directory
            self.filename = filename
        else:
            self.direct = direct # use user input directory
            self.filename = self.direct + filename

    def get_iso(self):
        self.p_ads,self.q_ads = parse_3flex_iso(self.filename)
    def get_psd(self):
        self.Davg_3flex, self.Vp_dlogD_3flex = parse_3flex_psd(self.filename)


    def to_csv(self,filename='new.csv',type='iso',force=False):
        # type = 'ads' export adsorption part, 'des' export desorption part, 'all', export raw data
        # force: True force to write the file even the file exist.
        if type == 'iso':
            dataframe = pd.DataFrame(data={'p_ads':self.p_ads,'q_ads':self.q_ads})
            dataframe.to_csv(self.direct + 'ISO_'+filename, index = False)
        else:
            dataframe = pd.DataFrame(data={'Davg': self.Davg_3flex, 'Vp_dlogD': self.Vp_dlogD_3flex})
            dataframe.to_csv(self.direct + 'PSD_'+filename, index = False)




# ------------- function
def parse_3flex_iso(filename):
    xl_workbook = xlrd.open_workbook(filename)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # find the start position row0 and col0
    for row in range(0, min(40, xl_sheet.nrows)):
        for col in range(0, 10):
            value = xl_sheet.cell_value(row, col)
            if value == 'Relative Pressure (P/Po)':
                # print(xl_sheet.cell_value(row, col))
                row0 = row + 2
                col0 = col
                break

    # find the end position row1
    for row in range(row0, xl_sheet.nrows):
        value = xl_sheet.cell_value(row, col0)
        if value == '':  # if it is not a number
            row1 = row
            break

    # get the pressure and adsorption quantity
    p = np.array([xl_sheet.cell_value(row, col0) for row in range(row0, row1)])
    q = np.array([xl_sheet.cell_value(row, col0 + 2) for row in range(row0, row1)])

    return p, q

def parse_3flex_psd(filename):
    xl_workbook = xlrd.open_workbook(filename)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # find the start position row0 and col0
    for row in range(0, min(40, xl_sheet.nrows)):
        for col in range(26*3, xl_sheet.ncols):
            value = xl_sheet.cell_value(row, col)
            if value == 'dV/dlog(W) Pore Volume':
                # print(xl_sheet.cell_value(row, col))
                row0 = row + 2
                col0 = col
                break

    # find the end position row1
    for row in range(row0, xl_sheet.nrows):
        value = xl_sheet.cell_value(row, col0)
        if value == '':  # if it is not a number
            row1 = row
            break

    # get the pressure and adsorption quantity
    Davg_3flex = np.array([xl_sheet.cell_value(row, col0) for row in range(row0, row1)])
    Vp_dlogD_3flex = np.array([xl_sheet.cell_value(row, col0 + 1) for row in range(row0, row1)])

    return Davg_3flex, Vp_dlogD_3flex





if __name__ == '__main__':
    direct ='F:/3Flex Data_3/Export_CO2/CO2_EF11354/'
    input_file_list = ['EOG_EF11354_HFailed_1_CO2(2rd)_t2p1_May_3.xls', 'EOG_EF11354_HFailed_2(40-140)_CO2(2rd)_t3p2_Apr_13.xls',
                       'EOG_EF11354_Intact_1_CO2(2rd)_t4p3_Apr_13.xls', 'EOG_EF11354_Intact_2(40-140)_CO2(2rd)_t2p1_Apr_13.xls',
                       'EOG_EF11354_VFailed_1_CO2(2rd)_t3p2_May_3.xls', 'EOG_EF11354_VFailed_2(40-140)_CO2(2rd)_t8p3_May_3.xls',
                       ]
    output_file_list = ['1_223_HF_1_co2.csv', '1_223_HF_2_co2.csv',
                        '1_223_In_1_co2.csv' , '1_223_In_2_co2.csv',
                        '1_223_VF_1_co2.csv', '1_223_VF_2_co2.csv'
                        ]

    for i, inputfile in enumerate(input_file_list):
        print(i)
        my_file = file_3flex_co2(inputfile, direct)
        my_file.get_iso()
        my_file.get_psd()
        my_file.to_csv(output_file_list[i],type='iso')
        my_file.to_csv(output_file_list[i], type='psd')
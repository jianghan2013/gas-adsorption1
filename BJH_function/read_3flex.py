import xlrd
import numpy as np
import pandas as pd
import os

class file_3flex():
    def __init__(self,filename,direct=None):
        # filename: input file name
        if direct is None:
            self.direct = os.getcwd() # use current directory
            self.filename = filename
        else:
            self.direct = direct # use user input directory
            self.filename = self.direct + filename

    def get_iso(self):
        self.p,self.q = parse_3flex_iso(self.filename)
        self.p_ads, self.q_ads, self.p_des, self.q_des = split_isotherm(self.p,self.q)

    def get_psd_dft_3flex(self):
        self.Davg_3flex, self.Vp_dlogD_3flex = parse_3flex_psd_dlogD(self.filename)
        self.Davg_3flex, self.Vp_3flex = parse_3flex_psd(self.filename)

    def iso_to_csv(self,filename='new.csv',type='ads',force=False):
        # type = 'ads' export adsorption part, 'des' export desorption part, 'all', export raw data
        # force: True force to write the file even the file exist.
        if type == 'ads':
            dataframe = pd.DataFrame(data={'p_ads':self.p_ads,'q_ads':self.q_ads})
        elif type =='des':
            dataframe = pd.DataFrame(data={'p_des': self.p_des, 'q_des': self.q_des})
        elif type == 'full':
            dataframe = pd.DataFrame(data={'p': self.p, 'q': self.q})
        # save the file
        if type == 'ads':
            dataframe.to_csv(self.direct + 'ISO_'+filename, index=False)
        else:
            dataframe.to_csv(self.direct + 'ISO_' + type+ '_'+ filename, index=False)

    def psd_3flex_to_csv(self,filename='new.csv'):
        dataframe = pd.DataFrame(data={'Davg_3flex': self.Davg_3flex, 'Vp_3flex': self.Vp_3flex,
                                       'Vp_dlogD_3flex': self.Vp_dlogD_3flex})
        dataframe.to_csv(self.direct + 'PSD_3flex_'+ filename, index=False)



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


def split_isotherm(p, q):
    p = np.array(p)
    q = np.array(q)
    id_pmax = np.argmax(p)
    p_ads = p[0:id_pmax + 1]
    q_ads = q[0:id_pmax + 1]
    p_des = p[id_pmax:]
    q_des = q[id_pmax:]
    return p_ads, q_ads, p_des, q_des


def parse_3flex_psd(filename):
    xl_workbook = xlrd.open_workbook(filename)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # find the start position row0 and col0
    for row in range(0, min(40, xl_sheet.nrows)):
        for col in range(1, xl_sheet.ncols):
            value = xl_sheet.cell_value(row, col)
            if value == 'Incremental Pore Volume':
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
    Vp_3flex = np.array([xl_sheet.cell_value(row, col0 + 1) for row in range(row0, row1)])

    return Davg_3flex, Vp_3flex

def parse_3flex_psd_dlogD(filename):
    xl_workbook = xlrd.open_workbook(filename)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # find the start position row0 and col0
    for row in range(0, min(40, xl_sheet.nrows)):
        for col in range(1, xl_sheet.ncols):
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
    filename ='C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/raw_data_directory.xlsx'
    root_direct = 'F:/3Flex Data_3/Export_N2/'

    # read the direct list information
    pd_direct = pd.read_excel(filename,'direct')

    for i in range(3):
        core_name = pd_direct['core_name'][i]
        print('----core_name',core_name)


        # go into subfolder directory
        pd_file = pd.read_excel(filename, core_name)
        root_direct1 = root_direct + pd_direct['direct'][i]
        for ind,infile in enumerate(pd_file['input_file']):
            outfile = pd_file['output_file'][ind]
            print(i,infile)
            my_file = file_3flex(filename = infile, direct = root_direct1)
            my_file.get_iso()
            my_file.get_psd_dft_3flex()

            # save to file
            my_file.iso_to_csv(outfile,type='ads')
            my_file.iso_to_csv(outfile, type='full')
            my_file.psd_3flex_to_csv(outfile)
    #print(iso_file_direct)

    '''
     = 'F:/3Flex Data_3/Export_N2/3_14/'
    input_file_list = ['EOG_3_14_failed_1_N2_t2p2_Sep11.xls', 'EOG_3_14_failed_2_N2_t5p3_Nov_7.xls',
                       'EOG_3_14_failed_3_N2_t4p2_Feb_24.xls', 'EOG_3_14_failed_4(0-40)_N2_t5p3_Mar_12.xls',
                       'EOG_3_14_intact_1_t2p2_N2_Oct_3.xls', 'EOG_3_14_intact_2_t3p3_N2_Oct_3.xls',
                       'EOG_3_14_intact_3_N2_t8p3_Feb_15.xls'
                       ]
    output_file_list = ['3_14_HF_1_N2.csv', '3_14_HF_2_N2.csv', '3_14_HF_3_N2.csv', '3_14_HF_4_N2.csv',
                        '3_14_In_1_N2.csv', '3_14_In_2_N2.csv', '3_14_In_3_N2.csv',
                        ]

    for i, inputfile in enumerate(input_file_list):
        print(i)
        my_file = file_3flex(inputfile, direct)
        # getting data
        my_file.get_iso()
        my_file.get_psd_dft_3flex()
        # save to file
        my_file.iso_to_csv(output_file_list[i])
        my_file.psd_3flex_to_csv(output_file_list[i])
    '''
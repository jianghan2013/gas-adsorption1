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

    def to_csv(self,filename='new.csv',type='ads',force=False):
        # type = 'ads' export adsorption part, 'des' export desorption part, 'all', export raw data
        # force: True force to write the file even the file exist.
        if type == 'ads':
            dataframe = pd.DataFrame(data={'p_ads':self.p_ads,'q_ads':self.q_ads})
        elif type =='des':
            dataframe = pd.DataFrame(data={'p_des': self.p_des, 'q_des': self.q_des})
        else:
            dataframe = pd.DataFrame(data={'p': self.p, 'q': self.q})
        dataframe.to_csv(self.direct + filename, index=False)


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


if __name__ == '__main__':
    direct ='F:/3Flex Data_3/Export_N2/4_34/degas 8hrs(USed)/'
    input_file_list = ['EOG_4_34_Hfailed_1_N2_t7p3_20150906.xls', 'EOG_4_34_Hfailed_2_N2_t4p2_20150617.xls',
                       'EOG_4_34_Hfailed_3(40-140)_N2_t4p1_20150906.xls',
                       'EOG_4_34_Intact_1_N2_t2p1_20150617.xls', 'EOG_4_34_Intact_2_N2_t5p3_20150617.xls',
                       'EOG_4_34_Intact_3_N2_t5p2_20150906.xls'
                       ]
    output_file_list = ['ISO_4_34_HF_1_N2.csv', 'ISO_4_34_HF_2_N2.csv', 'ISO_4_34_HF_3_N2.csv',
                        # 'ISO_3_14_HF_4_N2.csv',
                        'ISO_4_34_In_1_N2.csv', 'ISO_4_34_In_2_N2.csv', 'ISO_4_34_In_3_N2.csv',
                        ]

    for i, inputfile in enumerate(input_file_list):
        print(i)
        my_file = file_3flex(inputfile, direct)
        my_file.get_iso()
        my_file.to_csv(output_file_list[i])
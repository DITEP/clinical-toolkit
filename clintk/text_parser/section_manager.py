"""
Module to manage sections found by parser
"""
import pandas as pd
from .parser_utils import main_parser


def main_splitter(df, columns):
    """ splits all the entries of df

    Using `main_splitter` causes to split texts into several rows, one text
    is split into the number of sections it contains

    Parameters
    ----------
    df : pd.DataFrame

    columns : list of str

    Returns
    -------
    pd.DataFrame


    """
    df_res = pd.DataFrame(columns=columns)
    for index, row in df.iterrows():
        dic = main_parser(row['report'], str(row['patient_id']) + ' ' + row['original_date'])
        split = splitter(row['patient_id'],
                         row['original_date'],
                         row['cycle'],
                         dic)
        new_df = pd.DataFrame(split, columns=columns)

        df_res = pd.concat([df_res, new_df])
    return df_res


def splitter(patient_id, date, cycle, report_dict):
    """ splits the report into the number of keys in report_dict
    
    Parameters
    ----------
    patient_id
    date
    cycle
    report_dict

    Returns
    -------
    """
    return [{'patient_id': patient_id,
             'date': date,
             'cycle': cycle,
             'section': key,
             'text': report_dict[key]} for key in report_dict]


def reduce_dic(dico, sections):
    """ merges key, values of a dictionary

    @TODO find sections names using regex

    Parameters
    ----------
    dico : dict

    sections : list of str
        name of the sections to keep as in `ReportsParser.sections`

    Returns
    -------
    str
        concatenated contents of sections

    """
    res = ''
    # if sections id not None
    if sections:
        for key, value in dico.items():
            if key in sections:
                res += ' ' + value
    #keep all the sections
    else:
        for key, value in dico.items():
            res += ' ' + value

    return res

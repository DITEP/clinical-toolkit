"""
"""
import pandas as pd
from .text_parser import main_parser


def main_splitter(df, columns= ['patient_id', 'date', 'cycle', 'section', 'text']):
    """ splits all the entries of df

    Parameters
    ----------
    df
    columns

    Returns
    -------

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


def reduce_dic(dico, remove):
    """ merges key, values of a dictionary

    Parameters
    ----------
    dico
    remove

    Returns
    -------

    """
    res = ''
    for key, value in dico.items():
        if key not in remove:
            res += ' ' + value

    return res

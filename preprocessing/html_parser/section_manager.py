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
    """    splits the report into the number of keys in
        report_dict
    Parameters
    ----------
    patient_id
    date
    cycle
    report_dict

    Returns
    -------
    """
    # p = len(report_dict.keys())
    # ids = np.array([patient_id] * p)
    # dates = np.array([date] * p)
    # keys, texts = np.zeros((p,), dtype= str), np.zeros((p,), dtype= str)
    # for index, key in enumerate(report_dict):
    #     keys[index] = key
    #     texts[index] = report_dict[key]
    # res = np.vstack((ids, dates, keys, texts))
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
            res += ' ' + key + ' ' + value

    return res

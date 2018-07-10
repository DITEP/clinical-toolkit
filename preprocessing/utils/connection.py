"""
connection scripts2 to a remote MySQL db server

"""
import pandas as pd

from sqlalchemy import create_engine
from getpass import getpass


def sql2df(engine, table):
    """ Builds a DataFrame using a table from the
    database to which engine is connected

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Engine instance connected to a remote or local
        sql database
    table : str
        name of the table you wish to retrieve

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    # >>>uri = 'mysql+mysqldb://user:password@192.0.0.1/test'

    """
    return pd.read_sql_table(table, engine)


def get_engine(user, ip, db, use_password='yes', driver='pymysql'):
    """ returns engine instance connected to a given database

    Parameters
    ----------
    user : str
        username
    ip : str
        ip adress of the sql server
    db : str
        name of the database
            use_password : str, {'yes', 'no'}, default='yes
        'yes' to use a password to connect, if 'yes', the
        password will have to be entered in the terminal

    driver : str, default='pymysql'
        name of the driver used for MySQL-Python connexion,
        depends on your installation
        @see http://docs.sqlalchemy.org/en/latest/dialects/mysql.html
        for details

    Returns
    -------
    sqlalchemy.Engine instance

    """
    if use_password == 'yes':
        passwd = ':' + getpass('Password for {}: '.format(user))
    else:
        passwd = ''
    uri = 'mysql+{}://{}{}@{}/{}'.format(driver, user, passwd, ip, db)

    return create_engine(uri)

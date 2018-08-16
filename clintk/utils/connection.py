"""
connection script for linking to a remote SQL server

The two functions are intended to be used with each other


>>> engine = get_engine('username', '192.0.0.1', 'database')
Password for username:
>?
>>> df = sql2df(engine, 'table')

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
        dataframe representation of the `table` in database


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
        name of the driver used for MySQL-Python connexion, depends on your
        installation
        Check http://docs.sqlalchemy.org/en/latest/dialects/mysql.html
        for details

    Returns
    -------
    sqlalchemy.Engine
        engine can then be used for SQL related tasks

    """
    if use_password == 'yes':
        passwd = ':' + getpass('Password for {}: '.format(user))
    else:
        passwd = ''
    uri = 'mysql+{}://{}{}@{}/{}'.format(driver, user, passwd, ip, db)

    return create_engine(uri)

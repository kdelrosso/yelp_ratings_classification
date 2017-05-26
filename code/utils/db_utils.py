import pandas as pd
import psycopg2

def setup_sql_connection():
    """Return a connection to the yelp database."""

    conn = psycopg2.connect(dbname='yelp', user='kdelrosso', password='password', host='localhost')
    c = conn.cursor()
    return conn

def run_query(query):
    """Return the query results as a pandas DataFrame.

    Parameters
    ----------
    query: string, sql code to execute
    """

    conn = setup_sql_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    return df

"""Read-only PostgreSQL helper for the BigComp 2027 data profile (no credentials stored here).
Reads PG_HOST / PG_PORT / PG_DATABASE (or PG_DBNAME) / PG_USER / PG_PASSWORD from the environment."""
import os, psycopg2, psycopg2.extensions
psycopg2.extensions.register_type(psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values, "DEC2FLOAT", lambda v, c: float(v) if v is not None else None))
def conn():
    g = lambda *names, default=None: next((os.environ[n] for n in names if os.environ.get(n)), default)
    c = psycopg2.connect(host=g('PG_HOST','PGHOST',default='localhost'), port=int(g('PG_PORT','PGPORT',default='5432')),
                         dbname=g('PG_DBNAME','PG_DATABASE','PGDATABASE',default='pubg_survival'),
                         user=g('PG_USER','PGUSER',default='postgres'), password=g('PG_PASSWORD','PGPASSWORD'))
    c.set_session(readonly=True, autocommit=True)          # SELECT-only session
    cur = c.cursor(); cur.execute("SET statement_timeout='120s'"); cur.close()
    return c

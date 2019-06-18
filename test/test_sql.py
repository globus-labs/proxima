from proxima.data import SQLDataStorage
from math import isclose
import os

db_file = '__test__prox'
sqlite_url = 'sqlite:///{db_file}'.format(db_file=db_file)
sqlite_table = 'data'

def test_sqlite():
    # Should this be broken into smaller tests?
    list_size = 100
    data = SQLDataStorage(sqlite_url, table=sqlite_table)
    inputs = [float(i) for i in range(0,list_size)]
    outputs = [float(i**3) for i in range(0,list_size)]
    data.add_pairs(inputs, outputs)

    # Check that data sizes line up
    assert(len(data.inputs) == list_size )
    assert(len(data.outputs) == list_size )

    # Check that data can be retrieved
    r = data.get_all_data()
    assert(len(r) == list_size)

    # Add tear down of database (delete file directly?)
    os.remove('./{db_file}'.format(db_file=db_file))





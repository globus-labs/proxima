"""Classes to manage storage of input/output pairs"""

from typing import Tuple, List, Iterator
from random import shuffle
from sqlalchemy import create_engine
import pandas as pd

class BaseDataSource:
    """Abstract class for managing training data"""

    def add_pair(self, inputs, outputs):
        """Add input/output pair to data store"""
        raise NotImplementedError

    def add_pairs(self, inputs, outputs):
        """Add many inputs and outputs"""
        for i, o in zip(inputs, outputs):
            self.add_pair(i, o)

    def get_all_data(self) -> Tuple[List, List]:
        """Get all of the training data

        Order of training entries is not ensured to be in the same order

        Returns:
            (tuple) List of inputs and list of outputs
        """
        raise NotImplementedError

    def count(self) -> int:
        """Get the database size

        Returns:
             (int) Number of records
        """
        raise NotImplementedError

    def iterate_over_data(self, batch_size: int) -> Iterator[Tuple[List, List]]:
        """Produce the training data as a generator

        # TODO (wardlt): Should we assert orderings be random? I think "yes," but implementing that could be difficult

        Args:
            batch_size (int): Batch size
        Yields:
            Batches of input/output pairs
        """
        raise NotImplementedError


class InMemoryDataStorage(BaseDataSource):
    """Store input/output pairs in memory without any persistence mechanism"""

    def __init__(self):
        # TODO (wardlt): Do we want to optimize for insertion or random access
        self.inputs = list()
        self.outputs = list()

    def count(self) -> int:
        return len(self.inputs)

    def add_pair(self, inputs, outputs):
        self.inputs.append(inputs)
        self.outputs.append(outputs)

    def add_pairs(self, inputs, outputs):
        self.inputs.extend(inputs)
        self.outputs.extend(outputs)

    def get_all_data(self):
        return list(self.inputs), list(self.outputs)

    def iterate_over_data(self, batch_size: int):
        # Get the indices in a random order
        indices = range(len(self.inputs))
        shuffle(indices)

        # Generate batches
        for start in range(0, len(self.inputs), batch_size):
            batch_inds = indices[start:start + batch_size]
            yield [self.inputs[i] for i in batch_inds], [self.outputs[i] for i in batch_inds]

class SQLDataStorage(BaseDataSource):
    """Store and interact with input/output pairs from a SQL database"""

    def __init__(self, url, table="data"):
        self.inputs = list()
        self.outputs = list()
        self.url = url  # sqlalchemy.engine.Engine or sqlite3.Connection
        self.table = table
        self.engine = create_engine(url,
                                    echo=False)

    def add_pair(self, inputs, outputs):
        # Send a single pair set to SQL
        df = pd.DataFrame({"inputs": [inputs], "outputs": [outputs]})
        df.set_index('inputs', inplace=True)
        df.to_sql(self.table, con=self.engine, if_exists="append")

        self.inputs.append(inputs)
        self.outputs.append(outputs)

    def add_pairs(self, inputs, outputs):
        # Send a list of pair sets to SQL
        df = pd.DataFrame({"inputs": inputs, "outputs": outputs})
        df.set_index('inputs', inplace=True)
        df.to_sql(self.table, con=self.engine, if_exists="append")

        self.inputs.extend(inputs)
        self.outputs.extend(outputs)

    def get_all_data(self):
        q = "SELECT * FROM {table}".format(table=self.table)
        return self.engine.execute(q).fetchall()

    def iterate_over_data(self, batch_size: int):
        res = True
        q = "SELECT * FROM {table}".format(table=self.table)
        r = self.engine.execute(q)
        while res:
            res = r.fetchmany(batch_size)
            yield res

    def clear_cache(self):
        self.inputs = []
        self.outputs = []

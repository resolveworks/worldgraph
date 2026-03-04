import os

import pytest
from dotenv import load_dotenv

from worldgraph.embed import Embedder

load_dotenv()


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])

import os

import pytest
from dotenv import load_dotenv

from worldgraph.constants import RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import Graph

load_dotenv()


def embed_relations(graphs: list[Graph], embedder: Embedder) -> dict:
    """Collect all unique relations from graphs and embed them."""
    relations = sorted({edge.relation for graph in graphs for edge in graph.edges})
    return embedder.embed(relations, template=RELATION_TEMPLATE)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])

import os

import pytest
from dotenv import load_dotenv

from worldgraph.constants import NAME_EDGE, RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import Graph

load_dotenv()


def embed_relations(graphs: list[Graph], embedder: Embedder) -> dict:
    """Collect all unique relations from graphs, add NAME_EDGE, and embed them."""
    relations = sorted(
        {
            edge.relation
            for graph in graphs
            for edge in graph.edges
            if edge.relation != NAME_EDGE
        }
    )
    return embedder.embed([*relations, NAME_EDGE], template=RELATION_TEMPLATE)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])

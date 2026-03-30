import os

import pytest
from dotenv import load_dotenv

from worldgraph.constants import RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import Graph
from worldgraph.match import build_rel_sim

load_dotenv()


def compute_rel_sim(graphs: list[Graph], embedder: Embedder) -> dict:
    """Collect all unique relations from graphs and return pairwise rel_sim."""
    relations = sorted({edge.relation for graph in graphs for edge in graph.edges})
    embeddings = embedder.embed(relations, template=RELATION_TEMPLATE)
    return build_rel_sim(set(relations), embeddings)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])

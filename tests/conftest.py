import os

import pytest
from dotenv import load_dotenv

from worldgraph.constants import RELATION_TEMPLATE
from worldgraph.embed import Embedder
from worldgraph.graph import Graph
from worldgraph.match import build_rel_clusters, build_rel_sim

load_dotenv()


def compute_rel_clusters(
    graphs: list[Graph], embedder: Embedder, threshold: float = 0.8
) -> dict[str, int]:
    """Collect all unique relations from graphs and return cluster assignments."""
    relations = sorted(
        {edge.relation for graph in graphs for edge in graph.edges.values()}
    )
    embeddings = embedder.embed(relations, template=RELATION_TEMPLATE)
    rel_sim = build_rel_sim(set(relations), embeddings)
    return build_rel_clusters(rel_sim, threshold)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(os.environ["EMBEDDING_MODEL"])

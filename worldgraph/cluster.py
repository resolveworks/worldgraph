import json
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def collect_relations(extractions: list[dict]) -> list[str]:
    """Collect unique relation phrases across all articles."""
    seen = set()
    relations = []
    for article in extractions:
        for rel in article["relations"]:
            phrase = rel["relation"]
            if phrase not in seen:
                seen.add(phrase)
                relations.append(phrase)
    return relations


def embed_relations(relations: list[str], model_name: str) -> np.ndarray:
    """Embed relation phrases using fastembed."""
    model = TextEmbedding(model_name=model_name)
    embeddings = list(model.embed(relations))
    return np.array(embeddings)


def cluster_relations(
    embeddings: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative clustering on cosine similarity.

    Returns (labels, similarity_matrix).
    """
    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    similarity = normed @ normed.T

    # Convert similarity to distance for linkage (1 - similarity)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0, None)  # numerical stability

    condensed = squareform(distance)
    Z = linkage(condensed, method="average")

    # fcluster with distance threshold (1 - similarity_threshold)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    # Convert to 0-indexed Python ints
    labels = [int(l) - 1 for l in labels]

    return labels, similarity


def pick_representative(
    members: list[str],
    member_indices: list[int],
    similarity: np.ndarray,
) -> str:
    """Pick the medoid — member with highest avg similarity to other members."""
    if len(members) == 1:
        return members[0]

    idx = np.array(member_indices)
    sub_sim = similarity[np.ix_(idx, idx)]
    avg_sim = sub_sim.mean(axis=1)
    best = int(np.argmax(avg_sim))
    return members[best]


def run_clustering(
    input_path: Path,
    output_path: Path,
    model_name: str,
    threshold: float,
) -> None:
    """Run the full embed & cluster pipeline."""
    with open(input_path) as f:
        extractions = json.load(f)

    relations = collect_relations(extractions)
    click.echo(f"Found {len(relations)} unique relation phrases")

    click.echo(f"Embedding with {model_name}...")
    embeddings = embed_relations(relations, model_name)

    click.echo(f"Clustering (threshold={threshold})...")
    labels, similarity = cluster_relations(embeddings, threshold)

    # Build cluster structures
    cluster_map: dict[int, list[tuple[str, int]]] = {}
    for i, (phrase, label) in enumerate(zip(relations, labels)):
        cluster_map.setdefault(label, []).append((phrase, i))

    clusters = []
    relation_to_cluster = {}

    for cluster_id, members_with_idx in sorted(cluster_map.items()):
        members = [m[0] for m in members_with_idx]
        indices = [m[1] for m in members_with_idx]
        representative = pick_representative(members, indices, similarity)

        clusters.append(
            {
                "id": cluster_id,
                "representative": representative,
                "members": members,
            }
        )
        for phrase in members:
            relation_to_cluster[phrase] = cluster_id

    output = {
        "model": model_name,
        "threshold": threshold,
        "clusters": clusters,
        "relation_map": relation_to_cluster,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    multi = [c for c in clusters if len(c["members"]) > 1]
    click.echo(
        f"\n{len(clusters)} clusters ({len(multi)} with multiple members)"
    )
    for c in multi:
        click.echo(f"  [{c['representative']}]: {', '.join(c['members'])}")
    click.echo(f"\nWrote {output_path}")

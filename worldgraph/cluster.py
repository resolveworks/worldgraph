import json
from pathlib import Path

import click
import numpy as np
from fastembed import TextEmbedding
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def collect_entity_types(extractions: list[dict]) -> list[str]:
    """Collect unique entity type strings across all articles."""
    seen = set()
    types = []
    for article in extractions:
        for ent in article["entities"]:
            t = ent["type"]
            if t not in seen:
                seen.add(t)
                types.append(t)
    return types


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


def embed_phrases(phrases: list[str], model: TextEmbedding) -> np.ndarray:
    """Embed a list of phrases using fastembed."""
    embeddings = list(model.embed(phrases))
    return np.array(embeddings)


def embed_relations(relations: list[str], model: TextEmbedding) -> np.ndarray:
    """Embed relation phrases.

    Wraps each phrase as "A {phrase} B" before embedding to give the model
    syntactic context — it encodes the meaning of the verb phrase rather than
    just surface tokens.
    """
    wrapped = [f"A {phrase} B" for phrase in relations]
    return embed_phrases(wrapped, model)


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


def build_initial_graphs(
    extractions: list[dict],
    relation_map: dict[str, int],
    type_map: dict[str, str],
) -> list[dict]:
    """Build per-article graphs from extraction data.

    Each article becomes one graph with one occurrence per entity and one
    article per edge.
    """
    graphs = []
    for article in extractions:
        article_id = article["article_id"]
        entity_ids = {e["id"] for e in article["entities"]}

        entities = []
        for e in article["entities"]:
            entities.append(
                {
                    "id": e["id"],
                    "name": e["name"],
                    "type": type_map.get(e["type"], e["type"]),
                    "occurrences": [
                        {
                            "article_id": article_id,
                            "entity_id": e["id"],
                            "name": e["name"],
                        }
                    ],
                }
            )

        edges = []
        for rel in article["relations"]:
            src, tgt = rel["source"], rel["target"]
            if src not in entity_ids or tgt not in entity_ids:
                continue
            cluster_id = relation_map.get(rel["relation"])
            if cluster_id is None:
                continue
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "cluster_id": cluster_id,
                    "articles": [article_id],
                }
            )

        graphs.append({"id": article_id, "entities": entities, "edges": edges})

    return graphs


def run_clustering(
    input_path: Path,
    output_path: Path,
    model_name: str,
    threshold: float,
) -> None:
    """Run the full embed & cluster pipeline."""
    with open(input_path) as f:
        extractions = json.load(f)

    model = TextEmbedding(model_name=model_name)

    # --- Cluster entity types ---
    entity_types = collect_entity_types(extractions)
    click.echo(f"Found {len(entity_types)} unique entity types")

    click.echo(f"Embedding entity types with {model_name}...")
    type_embeddings = embed_phrases(entity_types, model)

    click.echo(f"Clustering entity types (threshold={threshold})...")
    type_labels, type_similarity = cluster_relations(type_embeddings, threshold)

    type_cluster_map: dict[int, list[tuple[str, int]]] = {}
    for i, (t, label) in enumerate(zip(entity_types, type_labels)):
        type_cluster_map.setdefault(label, []).append((t, i))

    type_map: dict[str, str] = {}
    multi_type_clusters = []
    for cluster_id, members_with_idx in sorted(type_cluster_map.items()):
        members = [m[0] for m in members_with_idx]
        indices = [m[1] for m in members_with_idx]
        representative = pick_representative(members, indices, type_similarity)
        for t in members:
            type_map[t] = representative
        if len(members) > 1:
            multi_type_clusters.append((representative, members))

    click.echo(
        f"{len(type_cluster_map)} type clusters "
        f"({len(multi_type_clusters)} with multiple members)"
    )
    for rep, members in multi_type_clusters:
        click.echo(f"  [{rep}]: {', '.join(members)}")

    # --- Cluster relation phrases ---
    relations = collect_relations(extractions)
    click.echo(f"\nFound {len(relations)} unique relation phrases")

    click.echo(f"Embedding relations with {model_name}...")
    embeddings = embed_relations(relations, model)

    click.echo(f"Clustering relations (threshold={threshold})...")
    labels, similarity = cluster_relations(embeddings, threshold)

    # Build cluster structures
    cluster_map: dict[int, list[tuple[str, int]]] = {}
    for i, (phrase, label) in enumerate(zip(relations, labels)):
        cluster_map.setdefault(label, []).append((phrase, i))

    cluster_labels: dict[str, str] = {}
    relation_to_cluster: dict[str, int] = {}

    multi_clusters = []
    for cluster_id, members_with_idx in sorted(cluster_map.items()):
        members = [m[0] for m in members_with_idx]
        indices = [m[1] for m in members_with_idx]
        representative = pick_representative(members, indices, similarity)

        cluster_labels[str(cluster_id)] = representative
        for phrase in members:
            relation_to_cluster[phrase] = cluster_id
        if len(members) > 1:
            multi_clusters.append((representative, members))

    # Build per-article graphs
    graphs = build_initial_graphs(extractions, relation_to_cluster, type_map)

    output = {
        "cluster_labels": cluster_labels,
        "graphs": graphs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    click.echo(
        f"\n{len(cluster_labels)} clusters ({len(multi_clusters)} with multiple members)"
    )
    for rep, members in multi_clusters:
        click.echo(f"  [{rep}]: {', '.join(members)}")
    click.echo(f"\n{len(graphs)} article graphs built")
    click.echo(f"Wrote {output_path}")

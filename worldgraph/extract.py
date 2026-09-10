import os
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from worldgraph.graph import Graph, Role, save_graph

load_dotenv()

SYSTEM_PROMPT = """You are an event extraction system building a graph of world facts from a news article. Entities are things in the world — people, organizations, places, and things. Events are the facts the article asserts: things that happen or hold between participants.

Be thorough: capture every asserted event and every participant. Use the exact names as they appear in the text.

Rules:
- Extract only what the article asserts as fact. Denied, disputed, or merely alleged claims are not extracted.
- Every event has a label and participants. The label is the base-form verb phrase without tense: 'acquire', 'be headquartered in' — never 'acquired', 'will acquire', 'is headquartered in'. A label contains only the event itself, never entity names. Entities that a fact refers to are nodes, not label content.
- Each participant has a role: 'agent' for the one doing or bringing about the event, 'patient' for anything else participating — the object acted on, a place, a capacity, a beneficiary. These are the only two roles.
- Qualifiers of a fact — a role, a place, a scope — are participants of that event, never separate events: if Tessa Corin manages Halden Freight as managing director for Vesterby, that is one 'manage' event with agent Tessa Corin and patients Halden Freight, managing director, and Vesterby.
- A participant may be another event: if someone joins a visit or one event causes another, the participating event is a participant. 'Ivo Brandt joined the visit' is a 'join' event with agent Ivo Brandt and the visit event as patient; 'the closure caused the suspension' is a 'cause' event with the closure event as agent and the suspension event as patient.
- The media is not part of the world graph: the publishing outlet, journalists, photographers, and the act of reporting never appear as entities or events.
- An event needs at least one participant. An action with no entity or event participant produces no event.

Each entity should have a short unique id, e.g. 'e1', 'e2', and the name as it appears in the text. Each event should have a short unique id, e.g. 'v1', 'v2'."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


class Participant(BaseModel):
    role: Role = Field(
        description="'agent' for the one doing or bringing about the event; "
        "'patient' for anything else participating — object, place, capacity, beneficiary"
    )
    ref: str = Field(description="The 'id' of the participating entity or event")


class Event(BaseModel):
    id: str = Field(
        description="Short unique identifier for this event, e.g. 'v1', 'v2'"
    )
    label: str = Field(
        description="Base-form verb phrase without tense, e.g. 'acquire', 'be headquartered in' — "
        "never 'acquired', 'will acquire', or 'is headquartered in'"
    )
    participants: list[Participant] = Field(min_length=1)


class Extraction(BaseModel):
    entities: list[Entity]
    events: list[Event]

    @model_validator(mode="after")
    def _validate_ids(self) -> "Extraction":
        """Entity and event ids are globally unique and live in one shared
        reference namespace; every participant must resolve to an entity or
        an event of this extraction, and no event may participate in itself.
        Forward references between events are valid.
        """
        entity_ids = [entity.id for entity in self.entities]
        event_ids = [event.id for event in self.events]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(f"duplicate entity ids: {sorted(entity_ids)}")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError(f"duplicate event ids: {sorted(event_ids)}")
        shared = sorted(set(entity_ids) & set(event_ids))
        if shared:
            raise ValueError(
                f"entity and event ids must be disjoint, shared: {shared}"
            )
        known = set(entity_ids) | set(event_ids)
        unknown = sorted(
            {
                participant.ref
                for event in self.events
                for participant in event.participants
                if participant.ref not in known
            }
        )
        if unknown:
            raise ValueError(f"participants reference unknown ids: {unknown}")
        self_refs = sorted(
            event.id
            for event in self.events
            if any(participant.ref == event.id for participant in event.participants)
        )
        if self_refs:
            raise ValueError(f"events participating in itself: {self_refs}")
        return self


def build_agent(model: str) -> Agent[object, Extraction]:
    """Single source of truth for the extraction agent — used by the pipeline and the eval harness."""
    return Agent(
        model,
        instructions=SYSTEM_PROMPT,
        output_type=Extraction,
        model_settings={"thinking": "low"},
    )


def extract_article(agent: Agent[object, Extraction], text: str) -> Extraction:
    """Extract entities and events from a single article's text."""
    prompt = f"""Extract all entities and events from this news article.

{text}"""

    return agent.run_sync(prompt).output


def extraction_to_graph(article_id: str, extraction: Extraction) -> Graph:
    """Convert an extraction into a runtime graph: entities become entity
    nodes, events become event nodes named by their label, and participants
    become role edges from the event node to the participant node.
    """
    graph = Graph(id=article_id)
    node_ids: dict[str, str] = {}
    for entity in extraction.entities:
        node_ids[entity.id] = graph.add_entity(entity.name).id
    for event in extraction.events:
        node_ids[event.id] = graph.add_event(event.label).id
    for event in extraction.events:
        for participant in event.participants:
            graph.add_edge(
                node_ids[event.id],
                node_ids[participant.ref],
                participant.role,
            )

    graph.validate()
    return graph


def run_extraction(article_files: list[Path], output_dir: Path) -> None:
    """Run extraction on all article text files, writing one graph JSON per article.

    The filename stem is the article id.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(os.environ["EXTRACTION_MODEL"])

    for i, article_file in enumerate(article_files, 1):
        article_id = article_file.stem
        out_path = output_dir / f"{article_id}.json"
        if out_path.exists():
            click.echo(
                f"[{i}/{len(article_files)}] Skipping (already extracted): {article_file.name}"
            )
            continue

        click.echo(f"[{i}/{len(article_files)}] Extracting from: {article_file.name}")
        extraction = extract_article(agent, article_file.read_text())

        graph = extraction_to_graph(article_id, extraction)
        save_graph(graph, out_path)

    click.echo(f"\nWrote graphs to {output_dir}/")

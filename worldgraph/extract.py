import os
from pathlib import Path

import click
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from worldgraph.graph import Graph, Role, save_graph

SYSTEM_PROMPT = """You are an event extraction system. You turn a news article into a graph of world facts: entities (named people, organizations, places, and things) and events (the facts the article asserts about them). Graphs from different articles are later matched to each other purely by structure — participant roles and entity names — so two articles describing the same fact must yield the same graph shape. Consistency in every decision below is what makes that matching possible.

# Entities

- An entity must be named: the article gives it a name. Descriptions without a name — "a bystander", "several contractors", "the company's fleet" — never become entities; a nameless node cannot be matched across articles.
- Use the name as it appears, without the leading article.
- The publishing outlet, its journalists, and its photographers never appear as entities; the act of reporting never appears as an event.

# Events

- An event is a fact the article asserts, with a label and at least two participants. Every participant is an entity of this article or another event of this article — never an unnamed mention. A happening that involves only one participant produces no event.
- Extract only what the article asserts as fact. Denied, alleged, or source-attributed claims produce nothing. Speech and perception acts — announce, say, report, admit, deny, expect, discover — are never events, but content the article presents as true through them is extracted on its own: "the airline admitted it had falsified maintenance logs" yields a "falsify" event; "the airline denied falsifying maintenance logs" yields nothing.
- Be thorough: capture every asserted event that has two or more named participants.

# Coordination and unnamed objects

- Split coordinated lists into one event per item: "opened offices in Drovik and Selje" is two events, identical except for the location. Coordinated agents and patients split the same way.
- An unnamed object folds into the event label instead of becoming a participant: "bought twenty aircraft from Aldermont Works" has no aircraft entity — the label is "buy aircraft", with agent and source participants.

# Labels

- The label is the base-form verb phrase, without tense or aspect, keeping its particles and prepositions: "call off", "look into", "be based in" — never "called off", "will look into", "is based in".
- A label contains the event itself plus at most a folded unnamed object — never an entity name.

# Roles

Each participant gets exactly one role from this closed set:

- 'agent' — the one doing or bringing about the event
- 'patient' — the thing acted on, changed, or that the event is about
- 'recipient' — a person or organization receiving something in a transfer ("awarded to", "sent to")
- 'beneficiary' — the party something is done for or in the name of ("for", "on behalf of")
- 'source' — origin of motion or transfer ("from")
- 'destination' — a place that is the endpoint of motion or transfer ("moved to", "travelled to")
- 'location' — a static place ("in", "at")
- 'capacity' — the title or role a participant acts in ("as CEO")
- 'instrument' — the tool or means ("with drones", "via email")
- 'price' — the monetary amount paid, exchanged, fined, or raised ("for $4 billion")

Assign roles canonically:

- Employment and titles are person-anchored whatever the wording — active, passive, or appositive: the agent is the person, the patient the organization, the capacity the title. "Aldermont Group employs Daria Solberg as chief financial officer" and "Daria Solberg, the chief financial officer of Aldermont Group" assert the same event: agent Daria Solberg, patient Aldermont Group, capacity chief financial officer. The capacity is always the title — never the person and never the place.
- "visit" takes the place as patient; motion verbs take it as destination. Organizations are patients, never locations; facilities and cities are locations or destinations.
- Qualifiers of a fact — a title, a place, a scope — are participants of that event with their proper role, never separate events and never label content.

# Events as participants

A participant may be another event: when someone joins a visit or one event causes another, the participating event is referenced by its id as a participant of the containing event.

# Identifiers

Each entity gets a short unique id ("e1", "e2", ...) and each event a short unique id ("v1", "v2", ...); participants reference these ids."""


class Entity(BaseModel):
    id: str = Field(
        description="Short unique identifier for this entity, e.g. 'e1', 'e2'"
    )
    name: str = Field(description="Entity name as it appears in the article")


class Participant(BaseModel):
    role: Role = Field(
        description="One of: 'agent' (doer), 'patient' (thing acted on), "
        "'recipient' (animate receiver in a transfer), "
        "'beneficiary' (done for or in the name of), "
        "'source' (origin of motion/transfer), "
        "'destination' (place endpoint of motion/transfer), "
        "'location' (static place), 'capacity' (title or role acted in), "
        "'instrument' (tool or means), "
        "'price' (monetary amount paid, exchanged, fined, or raised)"
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
    prompt = f"""<article>
{text}
</article>

Extract the entities and events the article asserts."""

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

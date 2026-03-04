"""Project-wide constants."""

from enum import Enum


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"


NAME_EDGE = "is named"

RELATION_TEMPLATE = "A {} B".format

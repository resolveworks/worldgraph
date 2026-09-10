"""Project-wide constants."""

# Confidence at which propagation commits a merge, gated on structural
# evidence. Events reach this bar from the neutral prior on participant
# structure alone; entities from their name seed plus matched events.
MERGE_THRESHOLD = 0.7

# Prior confidence for event pairs. Events carry no name similarity — their
# labels are never compared — so they start at maximum uncertainty and are
# lifted or suppressed purely by structural evidence. 0.5 is the neutral
# point of the evidence rule: an unlifted event counterpart contributes
# neither positive nor negative evidence.
EVENT_PRIOR = 0.5

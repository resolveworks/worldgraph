"""Matcher eval layer: scenario ladder + parameter sweep + report.

Tests prove the matcher's mechanism and quantify over parameters; this
package is the sibling layer that determines what the parameter VALUES
should be. Scenarios are tiny deterministic corpora with focal pairs
tagged by intended outcome; a grid sweep over ``MatcherParams`` makes
the boundary between "too small to merge" and "enough corroboration to
merge" visible — which scenarios merge in which parameter regions, with
how much margin — without asserting it. No LLM or embedding calls.
"""

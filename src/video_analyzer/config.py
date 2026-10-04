"""Safe YAML analysis settings and explicit command-line overrides."""

import re
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError

from .errors import AnalysisError


class SettingsLoader(yaml.SafeLoader):
    # Copy resolver lists so importing this module never changes PyYAML globally.
    yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"]
        for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(
                    "while loading settings",
                    node.start_mark,
                    "configuration keys must be strings",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while loading settings",
                    node.start_mark,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


SettingsLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF")
)


def load_settings(path: Path) -> dict:
    """Read one mapping, preserving relative paths for resolution from the cwd."""
    try:
        with path.open(encoding="utf-8") as source:
            loader = SettingsLoader(source)
            try:
                node = loader.get_single_node()
                if node is None or (
                    isinstance(node, yaml.ScalarNode)
                    and node.tag == "tag:yaml.org,2002:null"
                    and node.value == ""
                ):
                    return {}
                settings = loader.construct_document(node)
            finally:
                loader.dispose()
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AnalysisError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise AnalysisError(f"Configuration {path} must contain a mapping of analysis settings")
    return settings


def merge_settings(settings: dict, overrides: dict) -> dict:
    """Merge by leaf without mutating either source; validation follows the merge."""
    merged = dict(settings)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged

"""Configuration management for the graph system.

This module provides:
1. GraphConfig: A Pydantic model that stores high-level graph configurations
   and node-type-specific configurations.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class GraphConfig(BaseModel):
    """
    Configuration management for Graph objects.

    Attributes:
        config: General configuration options
        node_configs: Node-type specific configurations
    """

    config: Dict[str, Any] = Field(default_factory=dict)
    node_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> "GraphConfig":
        """
        Create a GraphConfig from a dictionary.

        Args:
            config: An optional dictionary of configuration data.

        Returns:
            A GraphConfig instance populated by the provided dictionary, or defaults if None.
        """
        return cls(config=config or {})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value from the main config dictionary.

        Args:
            key: The configuration key to look up.
            default: A default value if the key does not exist.

        Returns:
            The configuration value or the provided default.
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value in the main config dictionary.

        Args:
            key: The configuration key.
            value: The value to store.
        """
        self.config[key] = value

    def get_node_config(self, node_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve configuration for a specific node type.

        Args:
            node_type: The node type name.

        Returns:
            A dictionary of node-specific config values, or None if not found.
        """
        return self.node_configs.get(node_type)

    def set_node_config(self, node_type: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a specific node type.

        Args:
            node_type: The node type name.
            config: A dictionary of config values for this node type.
        """
        self.node_configs[node_type] = config 
"""Configuration management for the graph system.

This module provides:
1. GraphConfig: A Pydantic model that stores high-level graph configurations
   and node-type-specific configurations.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator
import os


class GraphConfig(BaseModel):
    """
    Configuration management for Graph objects.

    Attributes:
        max_parallel: Maximum number of parallel nodes to execute
        timeout: Maximum execution time in seconds
        retry_count: Number of retries for failed operations
        config: General configuration options
        node_configs: Node-type specific configurations
    """

    max_parallel: int = Field(
        default=4,
        description="Maximum number of parallel nodes to execute",
        gt=0
    )
    timeout: int = Field(
        default=60,
        description="Maximum execution time in seconds",
        gt=0
    )
    retry_count: int = Field(
        default=3,
        description="Number of retries for failed operations",
        ge=0
    )
    config: Dict[str, Any] = Field(default_factory=dict)
    node_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        validate_assignment = True

    @model_validator(mode='before')
    def load_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        try:
            if 'max_parallel' not in values and 'ALCHEMIST_MAX_PARALLEL' in os.environ:
                values['max_parallel'] = int(os.environ['ALCHEMIST_MAX_PARALLEL'])
            if 'timeout' not in values and 'ALCHEMIST_TIMEOUT' in os.environ:
                values['timeout'] = int(os.environ['ALCHEMIST_TIMEOUT'])
            if 'retry_count' not in values and 'ALCHEMIST_RETRY_COUNT' in os.environ:
                values['retry_count'] = int(os.environ['ALCHEMIST_RETRY_COUNT'])
        except ValueError:
            # If environment variables are invalid, use defaults
            pass
        return values

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> "GraphConfig":
        """
        Create a GraphConfig from a dictionary.

        Args:
            config: An optional dictionary of configuration data.

        Returns:
            A GraphConfig instance populated by the provided dictionary, or defaults if None.
        """
        return cls(**(config or {}))

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
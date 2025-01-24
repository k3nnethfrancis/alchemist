"""Configuration management for the graph system."""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class GraphConfig(BaseModel):
    """
    Configuration management for Graph objects.
    
    Attributes:
        config: Raw configuration dictionary
        node_configs: Node type specific configurations
    """
    
    config: Dict[str, Any] = Field(default_factory=dict)
    node_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> "GraphConfig":
        """Create config from dictionary."""
        return cls(config=config or {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config[key] = value
        
    def get_node_config(self, node_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific node type."""
        return self.node_configs.get(node_type)
        
    def set_node_config(self, node_type: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific node type."""
        self.node_configs[node_type] = config 
"""Factory for creating geometry encoders."""

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .vggt_encoder import VGGTEncoder


def create_geometry_encoder(config) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.
    
    Args:
        config: GeometryEncoderConfig instance with encoder configuration.
    Returns:
        Geometry encoder instance
    """

    encoder_type = config.encoder_type.lower()
    if encoder_type == "vggt":
        return VGGTEncoder(config)
    raise ValueError(f"Unsupported geometry encoder type: {encoder_type}. Only 'vggt' is kept in this release.")


def get_available_encoders():
    """Get list of available encoder types."""
    return ["vggt"]

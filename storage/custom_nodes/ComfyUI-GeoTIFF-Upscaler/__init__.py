try:
    import spandrel_extra_arches
    spandrel_extra_arches.install()
    print("[GeoTIFF Upscaler] Successfully registered extra architectures (SRFormer, SCSRFormer, DAT, HAT, SAFMN, GRL) in Spandrel.")
except Exception as e:
    pass

from .geotiff_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

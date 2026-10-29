"""Browser only: native pixel plugins, no bundled example DICOM or palettes.

The application supports native monochrome classic MR/PET, not palette-color
objects. A runtime hook stubs pydicom.examples before its unconditional import.
"""

hiddenimports = [
    "pydicom.pixels.decoders.base", "pydicom.pixels.decoders.rle",
    "pydicom.pixels.encoders.native",
]
excludedimports = ["pydicom.examples", "pydicom.cli"]
datas = []

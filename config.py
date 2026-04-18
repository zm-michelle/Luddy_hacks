from __future__ import annotations

import string
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PNG_EXTENSION = ".png"

# Keep the recognizer literal: case, punctuation, and spaces are preserved.
DEFAULT_CHARSET = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + " "
    + r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
)

CTC_BLANK_INDEX = 0

FONT_FAMILIES = {
    "typewriter": [
        "Courier New",
        "Liberation Mono",
        "DejaVu Sans Mono",
        "Courier",
    ],
    "sans": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ],
    "serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Liberation Serif",
    ],
}

FONT_FILE_HINTS = {
    "Courier New": ["courier new", "cour", "courier"],
    "Courier": ["courier"],
    "Liberation Mono": ["liberationmono", "liberation mono"],
    "DejaVu Sans Mono": ["dejavusansmono", "dejavu sans mono"],
    "Arial": ["arial"],
    "Helvetica": ["helvetica"],
    "DejaVu Sans": ["dejavusans.ttf", "dejavu sans"],
    "Liberation Sans": ["liberationsans", "liberation sans"],
    "Times New Roman": ["times new roman", "timesnewroman", "times"],
    "Times": ["times"],
    "DejaVu Serif": ["dejavuserif", "dejavu serif"],
    "Liberation Serif": ["liberationserif", "liberation serif"],
}

COMMON_FONT_DIRS = [
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
]

DEFAULT_PAGE_SIZE = (540, 258)  # width, height; matches SimulatedNoisyOffice patches
DEFAULT_DETECTOR_SIZE = (540, 258)  # width, height
DEFAULT_RECOGNIZER_HEIGHT = 32
DEFAULT_RECOGNIZER_WIDTH = 512

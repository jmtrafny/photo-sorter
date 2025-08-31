"""
Built-in default labels configuration for photo organizer.

This module provides default label configurations that work well for most
personal photo collections. These labels are compiled into the executable,
eliminating the need for external configuration files.

The labels are designed based on common photo categories found in personal
collections, with carefully chosen synonyms to improve CLIP's recognition
accuracy across different visual styles and contexts.

Label Structure:
    Each label is a dictionary with:
    - "prompt": Base text prompt (used as primary synonym)
    - "synonyms": List of alternative terms and phrases
    - "weight": Multiplier for scores (1.0 = neutral, >1.0 = boost, <1.0 = reduce)

Synonym Strategy:
    - Include both formal terms ("people") and casual ("selfie")
    - Cover different visual contexts (close-up, group, portrait)
    - Balance specificity with generality
    - Avoid overlapping terms between categories

Weight Guidelines:
    - 1.0: Standard weight for balanced categories
    - >1.0: Boost important categories (e.g., pets are often special)
    - <1.0: Reduce weight for categories that might over-match
"""

# Core label configuration used when no custom labels are provided
DEFAULT_LABELS = {
    "people": {
        "prompt": "people",
        "synonyms": [
            "people",           # Primary term
            "person",           # Singular form
            "a person",         # Natural language variant
            "group of people",  # Multiple subjects
            "portrait",         # Formal photos
            "selfie",           # Casual self-photos
            "man",              # Gender-specific terms
            "woman",
            "face",             # Focus on facial features
            "human",            # Generic human term
            "child"             # Age-specific (overlaps with "kids" category)
        ],
        "weight": 1.0  # Standard weight - people photos are common
    },
    "pets": {
        "prompt": "pets",
        "synonyms": [
            "pets",     # Generic term
            "pet",      # Singular
            "dog",      # Most common pets
            "cat",
            "puppy",    # Age-specific variants
            "kitten",
            "hamster",  # Small pets
            "rabbit",
            "animal"    # Broad term (could overlap with wildlife)
        ],
        "weight": 1.1  # Slight boost - pet photos are often precious to users
    },
    "landscape": {
        "prompt": "landscape",
        "synonyms": [
            "landscape",     # Formal photography term
            "scenery",       # Casual term
            "mountains",     # Specific geographical features
            "mountain",      # Singular form
            "beach",
            "desert",
            "forest",
            "waterfall",
            "lake",
            "sunset",        # Time-based natural phenomena
            "sunrise",
            "city skyline",  # Urban landscapes
            "river",
            "nature"         # Broad natural scenes
        ],
        "weight": 1.0
    },
    "food": {
        "prompt": "food",
        "synonyms": [
            "food",
            "meal",
            "pizza",
            "burger",
            "sushi",
            "cake",
            "dessert",
            "dish",
            "drink",
            "fruit"
        ],
        "weight": 1.0
    },
    "documents": {
        "prompt": "documents",
        "synonyms": [
            "documents",
            "document",
            "paper",
            "whiteboard",
            "presentation slide",
            "slide",
            "screenshot of text"
        ],
        "weight": 1.0
    },
    "screenshots": {
        "prompt": "screenshot",
        "synonyms": [
            "screenshot",      # Digital captures
            "screen capture",  # Alternative term
            "UI screenshot",   # User interface captures
            "code screenshot"  # Programming-related
        ],
        "weight": 0.9  # Lower weight - can be over-eager on digital images
    },
    "cars": {
        "prompt": "car",
        "synonyms": [
            "car",
            "truck",
            "motorcycle",
            "vehicle",
            "automobile"
        ],
        "weight": 0.9
    },
    "flowers": {
        "prompt": "flower",
        "synonyms": [
            "flower",
            "rose",
            "tulip",
            "daisy",
            "garden flower"
        ],
        "weight": 0.9
    },
    "kids": {
        "prompt": "kids",
        "synonyms": [
            "kids",
            "child",
            "children",
            "baby",
            "kid",
            "toddler"
        ],
        "weight": 1.0
    },
    "events": {
        "prompt": "event",
        "synonyms": [
            "wedding",        # Formal celebrations
            "birthday party", # Personal celebrations
            "concert",        # Entertainment events
            "parade",         # Public events
            "festival"        # Cultural events
        ],
        "weight": 1.0
    }
    # Note: Event detection can be challenging as it relies on context
    # rather than specific objects. May need adjustment based on usage.
}
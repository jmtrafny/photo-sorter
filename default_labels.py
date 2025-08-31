"""
Built-in default labels configuration for photo organizer.
This provides a good starting point that works for most photo collections.
"""

DEFAULT_LABELS = {
    "people": {
        "prompt": "people",
        "synonyms": [
            "people",
            "person",
            "a person",
            "group of people",
            "portrait",
            "selfie",
            "man",
            "woman",
            "face",
            "human",
            "child"
        ],
        "weight": 1.0
    },
    "pets": {
        "prompt": "pets",
        "synonyms": [
            "pets",
            "pet",
            "dog",
            "cat",
            "puppy",
            "kitten",
            "hamster",
            "rabbit",
            "animal"
        ],
        "weight": 1.1
    },
    "landscape": {
        "prompt": "landscape",
        "synonyms": [
            "landscape",
            "scenery",
            "mountains",
            "mountain",
            "beach",
            "desert",
            "forest",
            "waterfall",
            "lake",
            "sunset",
            "sunrise",
            "city skyline",
            "river",
            "nature"
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
            "screenshot",
            "screen capture",
            "UI screenshot",
            "code screenshot"
        ],
        "weight": 0.9
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
            "wedding",
            "birthday party",
            "concert",
            "parade",
            "festival"
        ],
        "weight": 1.0
    }
}
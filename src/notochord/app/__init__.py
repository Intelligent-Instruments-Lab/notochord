from .server import main as server
from .prompt import main as prompt
from .harmonizer import main as harmonizer
from .improviser import main as improviser
from .homunculus import main as homunculus
from .txalaparta import main as txalaparta

try:
    from .morse import main as morse
except ImportError:
    def morse():
        print('morse code app requires transformers package')
        exit(0)
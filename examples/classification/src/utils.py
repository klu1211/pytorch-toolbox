from pathlib import Path


def get_modules_at_path(path):
    return (
        p.stem
        for p in Path(__file__).parent.glob("*")
        if (p.name not in [Path(__file__).name, "__pycache__"])
    )

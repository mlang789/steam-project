#!/usr/bin/env python3
"""
Dump selected project files to a text file for LLM context.
"""

from pathlib import Path
import os

ROOT_FILES = {
}

INCLUDED_DIRS = {
    "src"
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".db", ".sqlite",
    ".ipynb", ".pkl", ".md"
}

# Dossiers à ignorer complètement lors du parcours
SKIP_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    "node_modules", # Souvent utile d'ignorer aussi pour le frontend
    ".idea",
    ".vscode",
    ".env"
}


def should_skip_file(path: Path) -> bool:
    return (
        not path.is_file()
        or path.suffix.lower() in SKIP_EXTENSIONS
    )


def dump_file(file_path: Path, project_root: Path, output):
    rel_path = file_path.relative_to(project_root)

    output.write("\n" + "=" * 80 + "\n")
    output.write(f"FILE: {rel_path}\n")
    output.write("=" * 80 + "\n\n")

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            output.write(f.read())
            output.write("\n")
    except Exception as e:
        output.write(
            f"[ERROR] Could not read file ({type(e).__name__}: {e})\n"
        )


def main() -> None:
    script_path = Path(__file__).resolve()
    # Adaptez ce niveau de parent selon l'emplacement de votre script
    project_root = script_path.parent.parent 

    dump_dir = project_root / "dump"
    dump_dir.mkdir(exist_ok=True)

    output_file_path = dump_dir / "repo_dump_v2.txt"

    with output_file_path.open("w", encoding="utf-8") as output:
        output.write(f"# Repository dump for project: {project_root.name}\n")
        output.write("# Included: main.py, README, src/, sql/\n\n")

        # --- Root files ---
        for filename in ROOT_FILES:
            file_path = project_root / filename
            if file_path.exists() and not should_skip_file(file_path):
                dump_file(file_path, project_root, output)

        # --- Included directories ---
        for dirname in INCLUDED_DIRS:
            dir_path = project_root / dirname
            if not dir_path.exists():
                continue

            # Modification ici : on récupère 'dirnames' pour filtrer le parcours
            for dirpath, dirnames, filenames in os.walk(dir_path):
                # Modification "in-place" de dirnames pour empêcher os.walk 
                # de descendre dans les dossiers indésirables (.venv, etc.)
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

                for filename in sorted(filenames):
                    file_path = Path(dirpath) / filename
                    if should_skip_file(file_path):
                        continue
                    dump_file(file_path, project_root, output)

    # Minimal console output
    print(f'Fichier texte enregistré dans : "{output_file_path}"')


if __name__ == "__main__":
    main()
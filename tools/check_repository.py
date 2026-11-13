"""Check source imports and local documentation links without importing ML runtimes."""

import ast
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("benchmarks", "tests", "tools", "controls", "src")
LOCAL_PACKAGES = {"benchmarks", "tests", "tools", "controls", "kohakufa_cute"}


def module_exists(name):
    base = ROOT / ("src" if name.split(".")[0] == "kohakufa_cute" else "")
    path = base / name.replace(".", "/")
    return path.with_suffix(".py").is_file() or path.is_dir()


def check_source(path):
    relative = path.relative_to(ROOT)
    try:
        tree = ast.parse(path.read_text(), filename=str(relative))
    except SyntaxError as error:
        yield f"{relative}:{error.lineno}: invalid Python syntax"
        return
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [node.module]
            if node.module == "benchmarks":
                names += ["benchmarks." + item.name for item in node.names if item.name != "*"]
        for name in names:
            if name.split(".")[0] in LOCAL_PACKAGES and not module_exists(name):
                yield f"{relative}:{node.lineno}: unresolved local module {name}"
        # Check path constructions, not historical metadata or synthetic strings.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            parent = parents.get(node)
            path_call = isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "Path"
            path_join = isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
            if (path_call or path_join) and re.fullmatch(
                r"(?:benchmarks|tests|src/kohakufa_cute|docs)/[\w./-]+\.(?:py|md)", value
            ):
                if not (ROOT / value).is_file():
                    yield f"{relative}:{node.lineno}: missing source path {value}"


def check_links(path):
    for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text()):
        if ":" in target or target.startswith(("#", "/")):
            continue
        destination = unquote(target.partition("#")[0])
        if destination and not (path.parent / destination).exists():
            yield f"{path.relative_to(ROOT)}: missing local link {target}"


def main():
    sources = sorted(path for directory in SOURCE_DIRS for path in (ROOT / directory).rglob("*.py"))
    documents = [ROOT / "README.md"] + sorted(
        path for directory in ("docs", "benchmarks", "results") for path in (ROOT / directory).rglob("*.md")
    )
    errors = [error for path in sources for error in check_source(path)]
    errors += [error for path in documents for error in check_links(path)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {len(sources)} Python files and {len(documents)} Markdown files; local modules and links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

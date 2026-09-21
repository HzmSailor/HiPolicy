#!/usr/bin/env python3
"""Check executable neural bodies against fingerprints from the original source."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StripDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


class IgnoreCheckpointIO(ast.NodeTransformer):
    def visit_If(self, node):
        names = {part.id for part in ast.walk(node.test) if isinstance(part, ast.Name)}
        names.update(part.attr for part in ast.walk(node.test) if isinstance(part, ast.Attribute))
        if names.intersection({'checkpoint_every', 'should_checkpoint'}):
            return None
        return self.generic_visit(node)


class NormalizeSubscripts(ast.NodeTransformer):
    """Match Python 3.9 AST fingerprints when running under Python 3.8."""
    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_ExtSlice(self, node):
        return ast.Tuple(elts=[self.visit(dim) for dim in node.dims], ctx=ast.Load())


def digest(nodes, kind=None):
    tree = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    if kind == 'training':
        tree = IgnoreCheckpointIO().visit(tree)
    tree = NormalizeSubscripts().visit(StripDocstrings().visit(tree))
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def body(path, class_name, method, profile=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == method)
    nodes = node.body
    if profile:
        branch = nodes[0]
        if isinstance(branch, ast.If) and any(
            isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id == 'benchmark'
            for part in ast.walk(branch.test)
        ):
            nodes = branch.body if profile == 'robotwin_v1' else branch.orelse
    return nodes


def check():
    records = json.loads((ROOT / 'docs/computation_manifest.json').read_text())
    failures = []
    for record in records:
        nodes = body(ROOT / record['release'], record['class'], record['method'], record.get('profile'))
        if digest(nodes, record.get('kind')) != record['sha256']:
            failures.append(f"{record['release']}:{record['class']}.{record['method']} ({record.get('profile', 'shared')})")
    if failures:
        raise AssertionError('Original computation changed:\n' + '\n'.join(failures))
    return len(records)


if __name__ == '__main__':
    print(f'{check()} original computation bodies match (networks, losses and training loops; checkpoint I/O excluded).')

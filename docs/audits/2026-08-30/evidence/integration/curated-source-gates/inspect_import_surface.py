"""Conservative import-time source inventory; does not import repository modules."""
import ast
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[5] / 'code'

class ImportSurface(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.calls = []
    def visit_If(self, node):
        # These benchmark CLIs are not run when imported.
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and any(isinstance(x, ast.Constant) and x.value == '__main__' for x in node.test.comparators):
            for item in node.orelse:
                self.visit(item)
            return
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        for item in [*node.decorator_list, *node.args.defaults, *[x for x in node.args.kw_defaults if x is not None]]:
            self.visit(item)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Lambda(self, node):
        for item in [*node.args.defaults, *[x for x in node.args.kw_defaults if x is not None]]:
            self.visit(item)
    def visit_Import(self, node):
        self.imports.append(node)
    visit_ImportFrom = visit_Import
    def visit_Call(self, node):
        self.calls.append({'line': node.lineno, 'call': ast.unparse(node)})
        self.generic_visit(node)


def existing_module(name):
    path = ROOT.joinpath(*name.split('.'))
    return [p for p in [path.with_suffix('.py'), path / '__init__.py'] if p.is_file()]


def imported_paths(path, imports):
    module = path.relative_to(ROOT).with_suffix('').parts
    package = list(module[:-1])
    for node in imports:
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        else:
            prefix = package[:len(package) - node.level + 1] if node.level else []
            if node.module:
                prefix += node.module.split('.')
            base = '.'.join(prefix)
            names = [base, *[f'{base}.{item.name}' for item in node.names if item.name != '*']]
        for name in names:
            parts = name.split('.')
            for size in range(1, len(parts) + 1):
                yield from existing_module('.'.join(parts[:size]))

initial = sorted(set(ROOT.glob('ch*/baseline_*.py')) | set(ROOT.glob('ch*/optimized_*.py')))
queue = list(initial)
records = {}
while queue:
    path = queue.pop()
    key = str(path.relative_to(ROOT))
    if key in records:
        continue
    tree = ast.parse(path.read_text())
    visitor = ImportSurface()
    visitor.visit(tree)
    dependencies = sorted(set(imported_paths(path, visitor.imports)))
    records[key] = {'calls': visitor.calls, 'local_imports': [str(p.relative_to(ROOT)) for p in dependencies]}
    queue.extend(dependencies)
result = {'method': 'AST module/class body calls + definition decorators/defaults, excludes __main__ branches and function bodies; conservative local import closure, not proof of all third-party side effects', 'initial_count': len(initial), 'closure_count': len(records), 'modules': records}
(EVIDENCE / 'import-surface.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'initial_count': len(initial), 'closure_count': len(records), 'calls': sum(len(x['calls']) for x in records.values())}))
for path, record in sorted(records.items()):
    for call in record['calls']:
        value = call['call']
        if any(word in value for word in ('subprocess', 'requests.', 'urlopen', 'download', 'from_pretrained', 'load_inline', 'load(', '.cuda(', 'torch.randn', 'torch.empty', 'torch.ones', 'torch.zeros', 'init_process_group', 'detect_', 'configure', 'initialize', 'setup_')):
            print(f"{path}:{call['line']}: {value[:260]}")

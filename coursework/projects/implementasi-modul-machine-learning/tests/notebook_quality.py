"""Pemeriksaan semantik ringan untuk kualitas source code notebook."""

import ast


METRIC_FUNCTIONS = {
    "accuracy_score",
    "balanced_accuracy_score",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
}
SEARCH_CONSTRUCTORS = {"GridSearchCV", "RandomizedSearchCV"}
SELECTION_FUNCTIONS = {"argmax", "argmin", "max", "min", "nlargest", "nsmallest"}


def _call_name(call):
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _assigned_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name
            for element in target.elts
            for name in _assigned_names(element)
        }
    return set()


def _referenced_names(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _holdout_names(tree):
    holdout = set()
    training = set()
    assignments = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
        key=lambda node: node.lineno,
    )

    for node in assignments:
        if not (
            isinstance(node.value, ast.Call)
            and _call_name(node.value) == "train_test_split"
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.List, ast.Tuple))
        ):
            continue
        outputs = node.targets[0].elts
        split_inputs = set()
        for argument in node.value.args[:2]:
            split_inputs.update(_referenced_names(argument))
        derives_from_training = bool(split_inputs) and split_inputs <= training

        if not derives_from_training:
            for index in (1, 3):
                if index < len(outputs):
                    holdout.update(_assigned_names(outputs[index]))

        safe_output_indices = (0, 1, 2, 3) if derives_from_training else (0, 2)
        for index in safe_output_indices:
            if index < len(outputs):
                training.update(_assigned_names(outputs[index]))

    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if _referenced_names(assignment.value) & holdout:
                aliases = {
                    name
                    for target in assignment.targets
                    for name in _assigned_names(target)
                }
                new_aliases = aliases - holdout
                if new_aliases:
                    holdout.update(new_aliases)
                    changed = True
    return holdout


def _search_objects(tree):
    objects = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _call_name(node.value) not in SEARCH_CONSTRUCTORS:
            continue
        for target in node.targets:
            objects.update(_assigned_names(target))
    return objects


def _uses_any_name(nodes, names):
    return any(_referenced_names(node) & names for node in nodes)


def _is_search_fit(call, search_objects):
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "fit":
        return False
    owner = call.func.value
    if isinstance(owner, ast.Name) and owner.id in search_objects:
        return True
    return isinstance(owner, ast.Call) and _call_name(owner) in SEARCH_CONSTRUCTORS


def _is_holdout_metric(call, holdout):
    call_name = _call_name(call)
    checks_metric = call_name in METRIC_FUNCTIONS
    checks_estimator_score = (
        isinstance(call.func, ast.Attribute) and call.func.attr == "score"
    )
    return (checks_metric or checks_estimator_score) and _uses_any_name(
        [*call.args, *(kw.value for kw in call.keywords)], holdout
    )


def _contains_holdout_metric(node, holdout):
    return any(
        _is_holdout_metric(child, holdout)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    )


def _metric_results_from_candidate_loops(tree, holdout):
    tainted = set()
    loop_nodes = [
        child for child in ast.walk(tree) if isinstance(child, (ast.For, ast.AsyncFor))
    ]

    for loop in loop_nodes:
        for node in ast.walk(loop):
            if isinstance(node, ast.Assign) and _contains_holdout_metric(node.value, holdout):
                for target in node.targets:
                    tainted.update(_assigned_names(target))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and any(_contains_holdout_metric(argument, holdout) for argument in node.args)
            ):
                tainted.add(node.func.value.id)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            derived = set()
            if isinstance(node, ast.Assign) and _referenced_names(node.value) & tainted:
                for target in node.targets:
                    derived.update(_assigned_names(target))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and _uses_any_name(node.args, tainted)
            ):
                derived.add(node.func.value.id)
            new_names = derived - tainted
            if new_names:
                tainted.update(new_names)
                changed = True
    return tainted


def _contains_selection_call(node):
    return any(
        _call_name(child) in SELECTION_FUNCTIONS
        or (isinstance(child.func, ast.Attribute) and child.func.attr == "sort_values")
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    )


def find_model_selection_leaks(source: str) -> list[str]:
    """Kembalikan pelanggaran ketika holdout ikut menentukan konfigurasi model."""

    tree = ast.parse(source)
    holdout = _holdout_names(tree)
    search_objects = _search_objects(tree)
    loop_metric_results = _metric_results_from_candidate_loops(tree, holdout)
    leaks = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_search_fit(node, search_objects):
            if _uses_any_name([*node.args, *(kw.value for kw in node.keywords)], holdout):
                leaks.append(
                    f"baris {node.lineno}: GridSearchCV.fit memakai holdout"
                )

        if (
            isinstance(node, ast.Assign)
            and _referenced_names(node.value) & loop_metric_results
            and _contains_selection_call(node.value)
        ):
            leaks.append(f"baris {node.lineno}: hasil holdout memilih kandidat")

    return sorted(set(leaks))


def assert_no_model_selection_leakage(source: str) -> None:
    """Gagalkan tes dengan seluruh temuan anti-leakage yang terdeteksi."""

    leaks = find_model_selection_leaks(source)
    assert not leaks, "\n".join(leaks)

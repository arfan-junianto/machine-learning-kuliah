import pytest

from notebook_quality import (
    assert_no_model_selection_leakage,
    find_model_selection_leaks,
)


SAFE_CV_AND_FINAL_EVALUATION = """
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2)
cv_scores = []
for depth in [2, 4, 6]:
    candidate = DecisionTreeClassifier(max_depth=depth)
    cv_scores.append(cross_val_score(candidate, X_train, y_train, cv=5).mean())
best_depth = [2, 4, 6][int(np.argmax(cv_scores))]
best_model = DecisionTreeClassifier(max_depth=best_depth).fit(X_train, y_train)
final_accuracy = accuracy_score(y_holdout, best_model.predict(X_holdout))
"""


SAFE_VALIDATION_SELECTION = """
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)
validation_scores = []
for depth in [2, 4, 6]:
    candidate = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
    validation_scores.append(accuracy_score(y_val, candidate.predict(X_val)))
selected_depth = [2, 4, 6][int(np.argmax(validation_scores))]
final_model = DecisionTreeClassifier(max_depth=selected_depth).fit(X_train_full, y_train_full)
final_accuracy = accuracy_score(y_test, final_model.predict(X_test))
"""


REPEATED_HOLDOUT_SELECTION = """
train_features, holdout_features, train_labels, holdout_labels = train_test_split(
    X, y, test_size=0.2
)
scores = []
for depth in [2, 4, 6]:
    candidate = DecisionTreeClassifier(max_depth=depth).fit(train_features, train_labels)
    scores.append(
        accuracy_score(holdout_labels, candidate.predict(holdout_features))
    )
best_depth = [2, 4, 6][int(np.argmax(scores))]
"""


GRID_SEARCH_ON_HOLDOUT = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
search = GridSearchCV(SVC(), {"C": [0.1, 1, 10]}, cv=5)
search.fit(X_test, y_test)
"""


ALIASED_HOLDOUT_SELECTION = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
evaluation_features = X_test
evaluation_labels = y_test
scores = []
for kernel in ["linear", "rbf"]:
    candidate = SVC(kernel=kernel).fit(X_train, y_train)
    score = candidate.score(evaluation_features, evaluation_labels)
    scores.append(score)
selected_kernel = ["linear", "rbf"][int(np.argmax(scores))]
"""


def test_allows_cross_validation_on_training_data_and_one_final_evaluation():
    assert find_model_selection_leaks(SAFE_CV_AND_FINAL_EVALUATION) == []
    assert_no_model_selection_leakage(SAFE_CV_AND_FINAL_EVALUATION)


def test_allows_model_selection_on_validation_split_derived_from_training_data():
    assert find_model_selection_leaks(SAFE_VALIDATION_SELECTION) == []


def test_rejects_repeated_holdout_metrics_inside_candidate_loop():
    leaks = find_model_selection_leaks(REPEATED_HOLDOUT_SELECTION)

    assert len(leaks) == 1
    assert "baris 11" in leaks[0]
    assert "hasil holdout memilih kandidat" in leaks[0]


def test_rejects_grid_search_fitted_on_holdout_data():
    leaks = find_model_selection_leaks(GRID_SEARCH_ON_HOLDOUT)

    assert len(leaks) == 1
    assert "baris 4" in leaks[0]
    assert "GridSearchCV.fit memakai holdout" in leaks[0]


def test_tracks_holdout_aliases_without_relying_on_variable_names():
    leaks = find_model_selection_leaks(ALIASED_HOLDOUT_SELECTION)

    assert len(leaks) == 1
    assert "baris 10" in leaks[0]
    assert "hasil holdout memilih kandidat" in leaks[0]


def test_assertion_reports_all_detected_leaks():
    with pytest.raises(AssertionError, match="GridSearchCV.fit memakai holdout"):
        assert_no_model_selection_leakage(GRID_SEARCH_ON_HOLDOUT)

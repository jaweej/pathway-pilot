import linopy
import numpy as np
import pytest

from pathway_pilot.cplex_callable import _assign_solution


def test_assign_solution_restores_exact_fixed_variables():
    model = linopy.Model()
    decision = model.add_variables(lower=0, name="decision")
    exact_fixed_value = 96_658_253.11031501
    fixed = model.add_variables(
        lower=exact_fixed_value,
        upper=exact_fixed_value,
        name="fixed",
    )
    model.add_constraints(decision >= 1, name="minimum")
    model.objective = decision - fixed

    primal_by_label = {
        int(decision.labels): 1.0,
        int(fixed.labels): 96_658_300.0,
    }
    primal = np.array(
        [primal_by_label[int(label)] for label in model.matrices.vlabels]
    )
    dual = np.zeros(len(model.matrices.clabels))

    _assign_solution(model, primal, dual, objective=-96_658_299.0)

    assert float(fixed.solution) == exact_fixed_value
    assert model.objective.value == pytest.approx(1.0 - exact_fixed_value)

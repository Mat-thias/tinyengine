# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   milp_allocator.py
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
#  - Tolulope Matthew Busoye, matthew@powerlabstech.com
#  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

"""
MILP activation allocator
============================================================

Assigns every activation tensor a byte offset in the single SRAM buffer, so
that tensors alive at the same time never share an address and the peak
address touched is minimal. Replaces the greedy FirstFit pass, which reaches
the lower bound on MCUNet-shaped graphs but not on arbitrary ones.

Notation
--------
    T                   set of tensors, one per rectangle
    [s_t, e_t)          lifetime of t in layer indices, end exclusive
    z_t                 effective_size -- bytes t occupies while alive,
                        including the right-inplace workspace (Delta + input)
    input(t)            tensor whose buffer t is computed in place over
    N                   address alignment, 4 or 32 bytes
    M                   big-M slack constant, taken as the SRAM size
    x_t                 placement of t, its byte offset         (decision)
    a_t                 x_t counted in N-byte words             (decision)
    Z                   peak address touched                    (decision)
    b_uv                1 if u lies below v, else 0             (decision)

Model
-----
    min  Z
    s.t. Z         >= x_t + z_t                     for all t
         x_t        = N * a_t                       for all t
         x_u + z_u <= x_v + M(1 - b_uv)             for overlapping u, v
         x_v + z_v <= x_u + M b_uv                  for overlapping u, v
         x_t        = x_input(t)                    for in-place t

The two big-M rows form a disjunction: whichever way b_uv is set, one row
binds and the other goes slack, so u lies wholly below v or wholly above it.
An ILP cannot state "or" directly, hence the binary and the slack constant.

Only pairs whose lifetimes intersect are constrained; everything else may
share an address freely. An in-place pair is never among them -- the
scheduler starts an in-place output's lifetime where its input's ends, so
the two only touch, and the aliasing equality is all that ties them
together. Were the pair also forced apart, the two constraints would
contradict and the model would be infeasible.
"""

import os
import itertools
import pyomo.environ as pyo

from .base_allocator import BaseAllocator

__all__ = ["MILPAllocator"]

SOLVER = "cbc"
SOLVER_PATH = "/home/matthias/Documents/Research/opt/enterprise-analytics-dev/pai_anop/features/opt/solvers/linux/ampl.linux64/ampl.linux-intel64"


class MILPAllocator(BaseAllocator):

    def sortSize(self):
        """
        Unsupported: FirstFit needs its rectangles ordered because the answer
        depends on insertion order. A MILP has no insertion order, so a call
        here means the caller still assumes the greedy allocator.
        """
        raise AttributeError(f"{self.__class__.__name__} should not be sorted for allocation.")

    def define_model(self):
        """Build the model above over self.rectangles. See module docstring for notation."""
        N = 32 if self.allign_memory_32 else 4

        # ==================================================================
        # Constraint rules
        # ==================================================================

        def can_tensors_overlap(t1, t2):
            """True if some layer falls inside both lifetimes."""
            for l in range(self.rectangles[t1]["start"], self.rectangles[t1]["end"]):
                if self.rectangles[t2]["start"] <= l < self.rectangles[t2]["end"]:
                    return True
            return False

        def data_alligned_rule(model, t):
            """x_t = N*a_t. Every tensor starts on an N-byte boundary."""
            return model.placement[t] == model.placement_allign[t] * N

        def largest_space_in_memory_rule(model, t):
            """Z >= x_t + z_t. Z is pushed to the highest address any tensor reaches."""
            return model.largest_space_in_memory >= model.placement[t] + self.rectangles[t]["effective_size"]

        def no_spatial_overlap_ascend_rule(model, t1, t2):
            """x_u + z_u <= x_v + M(1 - b_uv). Binds when b_uv = 1: u below v."""
            assert t1 != t2, f"A tensor cannot overlap itself, received t1 = t2 = {t1}"
            assert can_tensors_overlap(t1, t2), f"tensors {t1} and {t2} do not overlap"
            return model.placement[t1] + self.rectangles[t1]["effective_size"] <=  model.placement[t2] + self.SRAM * (1 - model.overlap_switch[(t1, t2)])

        def no_spatial_overlap_descend_rule(model, t1, t2):
            """x_v + z_v <= x_u + M b_uv. Binds when b_uv = 0: v below u."""
            assert t1 != t2, f"A tensor cannot overlap itself, received t1 = t2 = {t1}"
            assert can_tensors_overlap(t1, t2), f"tensors {t1} and {t2} do not overlap"
            return model.placement[t2] + self.rectangles[t2]["effective_size"] <=  model.placement[t1] + self.SRAM * model.overlap_switch[(t1, t2)]

        def inplace_tensor_rule(model, t):
            """
            x_t = x_input(t). An in-place kernel takes its input at the base of
            a buffer and leaves its output at that same base, so the two share
            an address. Delta, the gap by which the input is shifted at run
            time, is transient workspace already carried by z_input(t), not an
            offset between the two placements.
            """
            assert self.rectangles[t]["inplace"], f"tensor {t} is not done inplace"
            return model.placement[self.rectangles[t]["inplace_tensor_idx"]] == model.placement[t]


        # ==================================================================
        # Model
        # ==================================================================

        model = pyo.ConcreteModel()

        # placement parameters
        model.tensors = pyo.Set(initialize=range(len(self.rectangles)))
        # Unordered pairs: one binary and one disjunction per pair, not two.
        model.potential_overlap_tensors = pyo.Set(
            initialize=list((t1, t2) for t1, t2 in itertools.combinations(model.tensors, 2)
            if t1 != t2 and can_tensors_overlap(t1, t2))
        )
        model.inplace_tensors = pyo.Set(
            initialize=[t for t in model.tensors if self.rectangles[t]["inplace"]]
        )

        # placement variables, word alligned
        model.placement = pyo.Var(model.tensors, within=pyo.NonNegativeIntegers, bounds=(0, self.SRAM))
        # a_t. Alignment is expressed as a variable rather than rounded after
        # the solve, since rounding a solved placement would move tensors into
        # each other.
        model.placement_allign = pyo.Var(model.tensors, within=pyo.NonNegativeIntegers, bounds=(0, self.SRAM//N))
        model.largest_space_in_memory = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, self.SRAM))
        # Needed for the spatial overlap if 2 tensors exist while processing a layer
        # 1 if the are arranged in ascending order else 0
        model.overlap_switch = pyo.Var(model.potential_overlap_tensors, within=pyo.Binary)

        model.word_alligned = pyo.Constraint(model.tensors, rule=data_alligned_rule)
        model.largest_space_in_memory_constraint = pyo.Constraint(
            model.tensors, rule=largest_space_in_memory_rule
        )

        # model objective, the largest space in memory touched
        model.objective = pyo.Objective(expr=model.largest_space_in_memory, sense=pyo.minimize)
        model.no_spatial_overlap_ascend_constraint = pyo.Constraint(
            model.potential_overlap_tensors, rule=no_spatial_overlap_ascend_rule
        )
        model.no_spatial_overlap_descend_constraint = pyo.Constraint(
            model.potential_overlap_tensors, rule=no_spatial_overlap_descend_rule
        )
        model.inplace_tensor = pyo.Constraint(model.inplace_tensors, rule=inplace_tensor_rule)

        return model



    def allocate(self):
        """
        Solve and write x_t back to each rectangle's "placement". get_peak()
        then reads Z off those placements.
        """
        model = self.define_model()
        solver = pyo.SolverFactory(SOLVER, executable=os.path.join(SOLVER_PATH, SOLVER))

        results = solver.solve(model, tee=False)
        # Check before reading the variables: an unsolved model leaves every
        # placement at None, which would surface as a TypeError from int(None)
        # rather than the real cause. Infeasible normally means two constraints
        # contradict, e.g. an in-place pair left with overlapping lifetimes.
        condition = results.solver.termination_condition
        if condition != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"{self.__class__.__name__} could not place the {len(self.rectangles)} tensors: "
                f"solver terminated with '{condition}' (status '{results.solver.status}'). "
                f"SRAM limit is {self.SRAM} bytes."
            )

        for t in model.tensors:
            self.rectangles[t]["placement"] = int(model.placement[t].value)
        

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
    in(t)               inputs t may be computed in place over
    out(i)              the output that overwrites i
    N                   address alignment, 4 or 32 bytes
    M                   big-M slack constant, taken as the SRAM size
    x_t                 placement of t, its byte offset         (decision)
    a_t                 x_t counted in N-byte words             (decision)
    Z                   peak address touched                    (decision)
    b_uv                1 if u lies below v, else 0             (decision)
    d_i                 1 if i is the input that gets overwritten (decision)

Model
-----
    min  Z
    s.t. Z            >= x_t + z_t                  for all t
         x_t           = N * a_t                    for all t
         x_u + z_u    <= x_v + M(1 - b_uv)          for overlapping u, v
         x_v + z_v    <= x_u + M b_uv               for overlapping u, v
         sum_{i in in(t)} d_i = 1                   for t with candidates
         x_out(i) - x_i <= M(1 - d_i)               for every candidate i
         x_i - x_out(i) <= M(1 - d_i)               for every candidate i

The two overlap rows form a disjunction: whichever way b_uv is set, one row
binds and the other goes slack, so u lies wholly below v or wholly above it.
An ILP cannot state "or" directly, hence the binary and the slack constant.

Only pairs whose lifetimes intersect are constrained; everything else may
share an address freely. An in-place pair is never among them -- the
scheduler ends an in-place input's lifetime where its output's begins, so
the two only touch. Were the pair also forced apart, that and the aliasing
equality would contradict and the model would be infeasible.

Which input an op overwrites is itself a decision. Most ops have one
candidate, but an Add reads both operands at an index before writing that
index, so either will do -- and the choice sets the peak, because a residual
block that keeps its result in the freshly written buffer leaves the older
one free and settles one slot higher than the block before it. Picking it
here rather than by a rule means it is solved for, not guessed.

The last two rows say x_out(i) = x_i, written as a pair of inequalities
because the equality has to switch off when d_i = 0. Stating it directly as
sum_i x_i * d_i multiplies two decisions, and an integer program is linear.
The sum must be exactly 1, never at most 1: the touching lifetimes above are
what keep the pair out of the disjunction, and that only holds while the two
are pinned together. Left optional, the solver drops unaliased pairs at
overlapping addresses that nothing then forbids, and a kernel overwrites its
own input while still reading it.
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

        def data_alligned_rule(model, tensor_set):
            """x_t = N*a_t. Every tensor starts on an N-byte boundary."""
            constraint_list = list()
            for t in tensor_set:
                constraint_list.append(model.placement[t] == model.placement_allign[t] * N)
            return constraint_list
        
        def largest_space_in_memory_rule(model, tensor_set):
            """Z >= x_t + z_t. Z is pushed to the highest address any tensor reaches."""
            constraint_list = list()
            for t in tensor_set:
                constraint_list.append(model.largest_space_in_memory >= model.placement[t] + self.rectangles[t]["effective_size"])
            return constraint_list

        def no_spatial_overlap_rule(model, tensor_set):
            """x_u + z_u <= x_v + M(1 - b_uv). Binds when b_uv = 1: u below v."""
            constraint_list = list()
            # tensor_set is already the pairs that overlap in time, not the tensors.
            for t1, t2 in tensor_set:
                assert t1 != t2, f"A tensor cannot overlap itself, received t1 = t2 = {t1}"
                assert can_tensors_overlap(t1, t2), f"tensors {t1} and {t2} do not overlap"
                constraint_list.append(
                    model.placement[t1] + self.rectangles[t1]["effective_size"] <=\
                    model.placement[t2] + self.SRAM * (1 - model.overlap_switch[(t1, t2)])
                )
                constraint_list.append(
                    model.placement[t2] + self.rectangles[t2]["effective_size"] <=\
                    model.placement[t1] + self.SRAM * model.overlap_switch[(t1, t2)]
                )
            return constraint_list

        def inplace_tensor_decision_rule(model, tensor_set):
            """sum_i d_i = 1. Exactly one of the inputs of t may be overwritten by t."""
            constraint_list = list()
            for t in tensor_set:
                potential_inplace_inputs = []
                idx = self.rectangles[t]["idx"]
                for i, rec in enumerate(self.rectangles):
                    if rec["inplace"] and rec["inplace_tensor_out_idx"] == idx:
                        potential_inplace_inputs.append(i)
                assert len(potential_inplace_inputs) != 0
                constraint_list.append(
                    pyo.quicksum([
                        model.decision_inplace_tensor_in[i] for i in potential_inplace_inputs
                    ]) == 1
                )
            return constraint_list


        def inplace_tensor_placement_rule(model, tensor_set):
            """
            x_t = x_input(t). An in-place kernel takes its input at the base of
            a buffer and leaves its output at that same base, so the two share
            an address. Delta, the gap by which the input is shifted at run
            time, is transient workspace already carried by z_input(t), not an
            offset between the two placements.

            The buffer is decided by d_i, just on of its inputs gets overwritten.
            """
            constraint_list = list()
            for t in tensor_set:
                potential_inplace_inputs = []
                idx = self.rectangles[t]["idx"]
                for i, rec in enumerate(self.rectangles):
                    if rec["inplace"] and rec["inplace_tensor_out_idx"] == idx:
                        potential_inplace_inputs.append(i)
                assert len(potential_inplace_inputs) != 0
                for i in potential_inplace_inputs:
                    constraint_list.append(
                        model.placement[t] - model.placement[i] <=\
                        self.SRAM * (1 - model.decision_inplace_tensor_in[i])
                    )
                    constraint_list.append(
                        model.placement[i] - model.placement[t] <=\
                        self.SRAM * (1 - model.decision_inplace_tensor_in[i])
                    )
            return constraint_list


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
        model.possible_inplace_tensors_in = pyo.Set(
            initialize=[t for t in model.tensors if self.rectangles[t]["inplace"]]
        )
        model.possible_inplace_tensors_out = pyo.Set(
            initialize=list({
                self.rectangles[t]["inplace_tensor_out_idx"] for t in model.tensors
                if self.rectangles[t]["inplace_tensor_out_idx"] is not None})
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
        # 1 if this input is the one its op overwrites. An Add has two candidates
        # and the solver picks; every other op has one and the row forces it on.
        model.decision_inplace_tensor_in = pyo.Var(model.possible_inplace_tensors_in, within=pyo.Binary)

        # Constraints
        model.word_alligned_constraint = pyo.ConstraintList(rule=data_alligned_rule(model, model.tensors))
        model.largest_space_in_memory_constraint = pyo.ConstraintList(rule=largest_space_in_memory_rule(model, model.tensors))
        model.no_spatial_overlap_constraint = pyo.ConstraintList(rule=no_spatial_overlap_rule(model, model.potential_overlap_tensors))
        model.inplace_tensor_decision_constraint = pyo.ConstraintList(rule=inplace_tensor_decision_rule(model, model.possible_inplace_tensors_out))
        model.inplace_tensor_placement_constraint = pyo.ConstraintList(rule=inplace_tensor_placement_rule(model, model.possible_inplace_tensors_out))

        # model objective, the largest space in memory touched
        model.objective = pyo.Objective(expr=model.largest_space_in_memory, sense=pyo.minimize)

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
        

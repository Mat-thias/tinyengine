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
    n_t                 size, the bytes t holds
    Delta_i             gap, the right-inplace headroom for i
    in(t)               inputs t may be computed in place over
    out(i)              the output that overwrites i
    N                   address alignment, 4 or 32 bytes
    M                   big-M slack constant, the smaller of the SRAM and
                        every rectangle stacked end to end
    C                   memcopy cost per byte moved
    x_t                 placement of t, its byte offset         (decision)
    a_t                 x_t counted in N-byte words             (decision)
    z_t                 bytes t occupies while alive            (decision)
    Z                   peak address touched                    (decision)
    b_uv                1 if u lies below v, else 0             (decision)
    d_i                 1 if i is the input that gets overwritten (decision)
    r_i                 1 if i is shifted rather than placed low  (decision)

Model
-----
    min  Z + sum_i C * n_i * r_i
    s.t. Z             >= x_t + z_t                 for all t
         x_t            = N * a_t                   for all t
         z_t            = n_t                       for t not overwritten
         z_i           >= n_i + Delta_i * r_i       for every candidate i
         x_u + z_u     <= x_v + M(1 - b_uv)         for overlapping u, v
         x_v + z_v     <= x_u + M b_uv              for overlapping u, v
         sum_{i in in(t)} d_i = 1                   for t with candidates
         r_i           <= d_i                       for every candidate i
         x_out(i) - (x_i - Delta_i(1 - r_i)) <= M(1 - d_i)    for candidate i
         (x_i - Delta_i(1 - r_i)) - x_out(i) <= M(1 - d_i)    for candidate i

The two overlap rows form a disjunction: whichever way b_uv is set, one row
binds and the other goes slack, so u lies wholly below v or wholly above it.
An ILP cannot state "or" directly, hence the binary and the slack constant.

M has to be as small as it can be while still switching those rows off. The
SRAM limit is the obvious candidate and the wrong one: it is a user setting, so
it can be fifty times any address the model will ever use, and a slack constant
that loose flattens the relaxation. CBC then returns solutions well off the
optimum and reports lower == upper on them, which reads as proof.


Only pairs whose lifetimes intersect are constrained; everything else may
share an address freely. An in-place pair is never among them -- the
scheduler ends an in-place input's lifetime where its output's begins, so
the two only touch. Were the pair also forced apart, that and the aliasing
equality would contradict and the model would be infeasible.

Touching lifetimes leave the input dead at the very layer its buffer is being
read, which looks like a hole and is not one. Anything live at that layer
either started there, and the only tensor an op produces is the output itself,
or started earlier -- in which case it was live one layer before too, alongside
the input, and the overlap rows already hold it clear of the whole workspace.
The protection is written a layer early, not missing. Layer 0 is the exception
and is safe by exhaustion: only the model input and the first output start
there, and they are the pair.

Which input an op overwrites is itself a decision. Most ops have one candidate,
but an Add reads both operands at an index before writing that index, so either
will do -- and the choice sets the peak, because a residual block that keeps its
result in the freshly written buffer leaves the older one free and settles one
slot higher than the block before it. Picking it here rather than by a rule
means it is solved for, not guessed.

The sum must be exactly 1, never at most 1: the touching lifetimes are what keep
the pair out of the disjunction, and that only holds while the two are pinned
together. Left optional, the solver drops unaliased pairs at overlapping
addresses that nothing then forbids, and a kernel overwrites its own input while
still reading it.

How the pair is separated is the second decision. The output must start Delta
below where the input is read from, and there are two ways to arrange that:
memmove the input up by Delta and write the output from x_i (r_i = 1), or leave
the input alone and start the output at x_i - Delta (r_i = 0). Both give the
kernel the same geometry -- and it needs no telling which, since it shifts only
when handed one pointer for both. The first costs a copy, the second costs the
freedom to place the output. So r_i is priced into the objective at C per byte
moved, and the solver pays for a memmove only where placing the output low
would cost more peak than the copy is worth.

z_i carries Delta only when r_i does, since an unshifted input occupies just its
own bytes; r_i <= d_i keeps a candidate that is not the one being overwritten
from claiming either. Both rows would be products of two decisions written the
obvious way -- x_i * d_i to select an address, Delta * d_i * r_i to size the
workspace -- so each is a big-M pair or an implication instead. An integer
program is linear.
"""

import os
import itertools
import pyomo.environ as pyo

from .base_allocator import BaseAllocator

__all__ = ["MILPAllocator"]

SOLVER = "cbc"
SOLVER_PATH = "/home/matthias/Documents/Research/opt/enterprise-analytics-dev/pai_anop/features/opt/solvers/linux/ampl.linux64/ampl.linux-intel64"

# What a byte of memmove is worth in bytes of peak. At 1e-4 a 154,880 byte copy
# prices at ~15 bytes, so the shift is taken unless placing the output low saves
# more. Raise it to trade peak for fewer copies, lower it for the reverse.
RIGHT_SHIFT_COST = 1e-4


class MILPAllocator(BaseAllocator):

    def __init__(
        self,
        SRAM,
        optimize_right_shift=True,
        RIGHT_SHIFT_COST=RIGHT_SHIFT_COST,
        allign_memory_32=False
    ):
        self.rectangles = []
        self.SRAM = SRAM
        self.optimize_right_shift = optimize_right_shift
        self.RIGHT_SHIFT_COST = RIGHT_SHIFT_COST
        self.allign_memory_32 = allign_memory_32
    
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

        # Every rectangle stacked end to end is itself a schedule, so no address can
        # be higher, and nothing may exceed the SRAM either. See the module docstring
        # for why this is not just the SRAM limit.
        M = min(self.SRAM, sum(rec["size"] + rec.get("gap", 0) for rec in self.rectangles))

        potential_inplace_inputs_register = dict()
        for t in range(len(self.rectangles)):
            potential_inplace_inputs = []
            idx = self.rectangles[t]["idx"]
            for i, rec in enumerate(self.rectangles):
                if rec["inplace"] and rec["inplace_tensor_out_idx"] == idx:
                    potential_inplace_inputs.append(i)
            potential_inplace_inputs_register[t] = potential_inplace_inputs

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
                constraint_list.append(model.largest_space_in_memory >= model.placement[t] + model.effective_size[t])
            return constraint_list

        def no_spatial_overlap_rule(model, tensor_set):
            """x_u + z_u <= x_v + M(1 - b_uv). Binds when b_uv = 1: u below v."""
            constraint_list = list()
            # tensor_set is already the pairs that overlap in time, not the tensors.
            for t1, t2 in tensor_set:
                assert t1 != t2, f"A tensor cannot overlap itself, received t1 = t2 = {t1}"
                assert can_tensors_overlap(t1, t2), f"tensors {t1} and {t2} do not overlap"
                constraint_list.append(
                    model.placement[t1] + model.effective_size[t1] <=\
                    model.placement[t2] + M * (1 - model.overlap_switch[(t1, t2)])
                )
                constraint_list.append(
                    model.placement[t2] + model.effective_size[t2] <=\
                    model.placement[t1] + M * model.overlap_switch[(t1, t2)]
                )
            return constraint_list

        def inplace_tensor_decision_rule(model, tensor_set):
            """sum_i d_i = 1. Exactly one of the inputs of t may be overwritten by t."""
            constraint_list = list()
            for t in tensor_set:
                potential_inplace_inputs = potential_inplace_inputs_register[t]
                assert len(potential_inplace_inputs) != 0
                constraint_list.append(
                    pyo.quicksum([
                        model.decision_inplace_tensor_in[i] for i in potential_inplace_inputs
                    ]) == 1
                )
            return constraint_list

        def inplace_tensor_right_shift_rule(model, tensor_set):
            """s_i <= d_i. An input that is not the one being overwritten is never shifted."""
            return [
                model.decision_right_shift_inplace_tensor_in[i] <= model.decision_inplace_tensor_in[i]
                for i in tensor_set
            ]

        def inplace_tensor_always_right_shift_rule(model, tensor_set):
            """
            sum_i r_i = 1, per output. With r_i <= d_i and sum_i d_i = 1 this pins
            r to d, so the chosen input is shifted and no output is placed low.
            """
            return [
                pyo.quicksum([
                    model.decision_right_shift_inplace_tensor_in[i]
                    for i in potential_inplace_inputs_register[t]
                ]) == 1
                for t in tensor_set
            ]


        def inplace_tensor_placement_rule(model, tensor_set):
            """
            x_out(i) = x_i - Delta_i(1 - r_i), as a big-M pair so it switches off
            for the candidates that are not chosen.

            The output has to start Delta below where its input is read from. A
            shifted input is memmoved up to x_i + Delta and the output takes the
            base, so the two placements coincide; an unshifted one stays put and
            the output starts Delta lower instead. Both leave the kernel the same
            geometry, which is why it can tell them apart on pointers alone.
            """
            constraint_list = list()
            for t in tensor_set:
                potential_inplace_inputs = potential_inplace_inputs_register[t]
                assert len(potential_inplace_inputs) != 0
                for i in potential_inplace_inputs:
                    output_start_referenced_to_input_right_shift_decision =\
                        model.placement[i] - self.rectangles[i]["gap"] * (1 - model.decision_right_shift_inplace_tensor_in[i])
                    constraint_list.append(
                        model.placement[t] - output_start_referenced_to_input_right_shift_decision <=\
                        M * (1 - model.decision_inplace_tensor_in[i])
                    )
                    constraint_list.append(
                        output_start_referenced_to_input_right_shift_decision - model.placement[t] <=\
                        M * (1 - model.decision_inplace_tensor_in[i])
                    )
            return constraint_list

        def effective_size_rule(model, tensor_set):
            """
            z_i >= n_i + Delta_i * r_i, and z_t = n_t for everything else. Only a
            shifted input reaches Delta past its own bytes; an unshifted one has
            its output below it instead, covered by the output's own rectangle.
            """
            constraint_list = list()
            for t in tensor_set:
                if t in model.possible_inplace_tensors_in:
                    constraint_list.append(
                        model.effective_size[t] >= (
                            self.rectangles[t]["size"] +
                            self.rectangles[t]["gap"] *
                            model.decision_right_shift_inplace_tensor_in[t]
                        )
                    )
                else:
                    constraint_list.append(
                        model.effective_size[t] == self.rectangles[t]["size"]
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
        model.placement = pyo.Var(model.tensors, within=pyo.NonNegativeIntegers, bounds=(0, M))
        # a_t. Alignment is expressed as a variable rather than rounded after
        # the solve, since rounding a solved placement would move tensors into
        # each other.
        model.placement_allign = pyo.Var(model.tensors, within=pyo.NonNegativeIntegers, bounds=(0, M//N))
        model.largest_space_in_memory = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, M))
        # z_t. the effective size, basically max(size, size+gap) if t is inplace else size
        model.effective_size = pyo.Var(model.tensors, within=pyo.NonNegativeIntegers, bounds=(0, M))
        # Needed for the spatial overlap if 2 tensors exist while processing a layer
        # 1 if the are arranged in ascending order else 0
        model.overlap_switch = pyo.Var(model.potential_overlap_tensors, within=pyo.Binary)
        # 1 if this input is the one its op overwrites. An Add has two candidates
        # and the solver picks; every other op has one and the row forces it on.
        model.decision_inplace_tensor_in = pyo.Var(model.possible_inplace_tensors_in, within=pyo.Binary)
        # 1 to right+shift this input up by its gap, 0 to start the output below it instead.
        # Chooses to pay the cost of time with moving or memory, optimizer decides.
        model.decision_right_shift_inplace_tensor_in = pyo.Var(model.possible_inplace_tensors_in, within=pyo.Binary)

        # Constraints
        model.word_alligned_constraint = pyo.ConstraintList(rule=data_alligned_rule(model, model.tensors))
        model.largest_space_in_memory_constraint = pyo.ConstraintList(rule=largest_space_in_memory_rule(model, model.tensors))
        model.effective_size_constraint = pyo.ConstraintList(rule=effective_size_rule(model, model.tensors))
        model.inplace_tensor_right_shift_constraint = pyo.ConstraintList(rule=inplace_tensor_right_shift_rule(model, model.possible_inplace_tensors_in))
        if not self.optimize_right_shift:
            model.inplace_tensor_always_right_shift_constraint = pyo.ConstraintList(rule=inplace_tensor_always_right_shift_rule(model, model.possible_inplace_tensors_out))
        model.no_spatial_overlap_constraint = pyo.ConstraintList(rule=no_spatial_overlap_rule(model, model.potential_overlap_tensors))
        model.inplace_tensor_decision_constraint = pyo.ConstraintList(rule=inplace_tensor_decision_rule(model, model.possible_inplace_tensors_out))
        model.inplace_tensor_placement_constraint = pyo.ConstraintList(rule=inplace_tensor_placement_rule(model, model.possible_inplace_tensors_out))

        # model objective, the largest space in memory touched
        inplace_tensor_right_shift_cost = pyo.quicksum([
            self.RIGHT_SHIFT_COST * model.decision_right_shift_inplace_tensor_in[i]
            * self.rectangles[i]["size"]
            for i in model.possible_inplace_tensors_in
        ])
        objective_fun = model.largest_space_in_memory + inplace_tensor_right_shift_cost
        model.objective = pyo.Objective(expr=objective_fun, sense=pyo.minimize)

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

        def solved_int(var):
            """A solver returns 0.9999999 as often as 1.0, and int() of that is 0."""
            return int(round(pyo.value(var)))

        for t in model.tensors:
            rec = self.rectangles[t]
            rec["placement"] = solved_int(pyo.value(model.placement[t]))

            if rec["inplace"]:
                # the input is shift if its the one select of all the input its output depends
                # on to be overwritten and the optimizer found it best to right shift it for
                # objective
                right_shift = bool(solved_int(model.decision_right_shift_inplace_tensor_in[t]))
                if not right_shift:
                    # If not right_shift we set the gap to 0
                    rec["gap"] = 0

                rec["effective_size"] = rec["size"] + rec["gap"]

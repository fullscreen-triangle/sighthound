"""
Core model for the buffer-processor rail-network-yield paper.

Instantiates the definitions of Sections 2-8:
  - Section network (Def 2.1), geography (Def 2.2)
  - Block-second quantum beta_0 (Def 2.3)
  - Vehicles with convex energy cost (Def 2.6)
  - Items with private targets and residual distance (Def 4.5)
  - Network transport yield (Def 3.2)
  - Separation cost of a section (Def 4.1)
  - Deterministic closure via local reassignment (Def 5.2)
  - Clearing price system (Def 6.1)

Everything here is small-scale and exact: networks are kept small enough that
the global yield optimum can be found by brute-force enumeration, so each
theorem check compares an independently-computed quantity against ground truth.

Dependencies: numpy only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np


# ----------------------------------------------------------------------
# Infrastructure
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """A track section (processor). Def 2.1."""
    name: str
    tail: str          # tail station (buffer) name
    head: str          # head station name
    length_km: float   # ell(sigma)
    vmax_kmh: float    # civil speed limit v-bar(sigma)


@dataclass(frozen=True)
class Vehicle:
    """A vehicle. Def 2.6. Energy cost g_u(v) = a + b * v^2 (strictly convex)."""
    name: str
    vmax_kmh: float        # capability speed
    capacity: int          # item-slots c(u)
    energy_a: float = 0.0   # constant term of g_u
    energy_b: float = 0.0   # quadratic coefficient (>0 => strictly convex)

    def g(self, v_kmh: float) -> float:
        """Energy-wear cost per unit length at speed v (km/h)."""
        return self.energy_a + self.energy_b * (v_kmh ** 2)


@dataclass(frozen=True)
class Item:
    """A unit of transport demand with a private target. Def 4.5."""
    name: str
    loc: str      # current station
    target: str   # target station


@dataclass
class Network:
    """Section network + geography. Def 2.1, Def 2.2."""
    stations: list[str]
    sections: list[Section]
    coords: dict[str, float]   # 1-D geographic coordinate per station (km along corridor)
    beta0_s: float             # resolution floor / block headway (seconds)

    def rho(self, a: str, b: str) -> float:
        """Residual (geographic) distance metric. Def 2.2."""
        return abs(self.coords[a] - self.coords[b])

    def sections_from(self, station: str) -> list[Section]:
        return [s for s in self.sections if s.tail == station]

    def gradient_connected(self) -> bool:
        """Strong (Def 2.2) gradient-connectedness: from every station one can
        strictly reduce residual distance to every other *reachable* station.

        Note: a strictly one-way corridor is not gradient-connected in this
        universal sense (you cannot descend from the terminus back to the
        origin). For validation of routing/liveness we use the target-relative
        condition below, which is what the theorems actually require.
        """
        for b in self.stations:
            for bp in self.stations:
                if b == bp:
                    continue
                # only require descent toward stations that are actually reachable
                if not self._reachable(b, bp):
                    continue
                improve = any(
                    self.rho(s.head, bp) < self.rho(b, bp)
                    for s in self.sections_from(b)
                )
                if not improve:
                    return False
        return True

    def _reachable(self, src: str, dst: str) -> bool:
        """Is dst reachable from src by following directed sections?"""
        seen, stack = {src}, [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for s in self.sections_from(cur):
                if s.head not in seen:
                    seen.add(s.head)
                    stack.append(s.head)
        return False

    def target_descendable(self, items) -> bool:
        """Target-relative gradient condition (what Thm 8.6 requires): every
        live item, from every station on a path to its target, can strictly
        reduce residual distance to *its own* target. Checked over stations from
        which the target is reachable."""
        for x in items:
            for b in self.stations:
                if b == x.target or not self._reachable(b, x.target):
                    continue
                improve = any(
                    self.rho(s.head, x.target) < self.rho(b, x.target)
                    for s in self.sections_from(b)
                )
                if not improve:
                    return False
        return True


# ----------------------------------------------------------------------
# Block-second accounting
# ----------------------------------------------------------------------

def blockseconds(section: Section, v_kmh: float, beta0_s: float) -> int:
    """n(sigma, v) = ceil( ell / (v * beta_0) ). Def 3.2.

    Length in km, v in km/h, beta_0 in seconds. Convert consistently:
    distance covered per block-second = v(km/h) * beta_0(s) / 3600 (km).
    """
    km_per_block = v_kmh * beta0_s / 3600.0
    return math.ceil(section.length_km / km_per_block)


def useful_displacement(net: Network, section: Section, items: list[Item]) -> float:
    """Delta: target-ward residual-distance reduction of carried items. Def 3.1/Eq 3.2.

    Only the positive (target-ward) component counts.
    """
    total = 0.0
    for x in items:
        reduction = net.rho(section.tail, x.target) - net.rho(section.head, x.target)
        total += max(reduction, 0.0)
    return total


def blocksecond_cost(section: Section, vehicle: Vehicle, v_kmh: float,
                     beta0_s: float, lam: float) -> float:
    """Denominator contribution of one traversal: n + lambda * ell * g_u(v). Def 3.2."""
    n = blockseconds(section, v_kmh, beta0_s)
    return n + lam * section.length_km * vehicle.g(v_kmh)


# ----------------------------------------------------------------------
# Assignments and configurations
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Assignment:
    """A non-idle block-second assignment (u, v, S) on a section. Def 2.7."""
    section: str
    vehicle: str
    v_kmh: float
    items: tuple[str, ...]   # names of carried items


@dataclass
class Configuration:
    """A set of assignments over a (implicit, single-window) horizon. Def 2.7.

    We work with a single scheduling window per section for the equivalence and
    speed/sorting experiments (the theorems are per-block-second statements), so
    a configuration is a choice of one assignment (or idle) per section.
    """
    assignments: dict[str, Assignment | None]   # section name -> assignment or None (idle)


def capability_ceiling(vehicle: Vehicle, section: Section) -> float:
    """nu(u, sigma) = min(vmax(u), vbar(sigma)). Def 2.5."""
    return min(vehicle.vmax_kmh, section.vmax_kmh)


# ----------------------------------------------------------------------
# Yield of a configuration
# ----------------------------------------------------------------------

def configuration_yield(net: Network, config: Configuration,
                        vehicles: dict[str, Vehicle], items: dict[str, Item],
                        lam: float, h_hold: float = 0.0,
                        waiting_item_blockseconds: float = 0.0) -> float:
    """Network transport yield Y(C) = sum Delta / (sum cost + buffer holding). Def 3.2, Ass 3.4.

    Returns 0.0 for an all-idle configuration (no useful work, no cost).
    """
    num = 0.0
    den = 0.0
    section_by_name = {s.name: s for s in net.sections}
    for sec_name, a in config.assignments.items():
        if a is None:
            continue
        sec = section_by_name[sec_name]
        veh = vehicles[a.vehicle]
        carried = [items[nm] for nm in a.items]
        num += useful_displacement(net, sec, carried)
        den += blocksecond_cost(sec, veh, a.v_kmh, net.beta0_s, lam)
    den += h_hold * waiting_item_blockseconds
    if den == 0.0:
        return 0.0
    return num / den


# ----------------------------------------------------------------------
# Section-optimal speed (Def 7.1) and yield density (Def 7.4)
# ----------------------------------------------------------------------

def section_optimal_speed(section: Section, vehicle: Vehicle, delta: float,
                          beta0_s: float, lam: float,
                          grid: int = 2000) -> tuple[float, float]:
    """v*(u, sigma) = argmax delta / (n(sigma,v) + lambda*ell*g_u(v)). Def 7.1.

    Returns (v_star_kmh, best_ratio). Fine speed grid up to the capability ceiling.
    """
    ceiling = capability_ceiling(vehicle, section)
    best_v, best_ratio = None, -np.inf
    for i in range(1, grid + 1):
        v = ceiling * i / grid
        den = blocksecond_cost(section, vehicle, v, beta0_s, lam)
        ratio = delta / den if den > 0 else -np.inf
        if ratio > best_ratio:
            best_ratio, best_v = ratio, v
    return best_v, best_ratio


def yield_density(net: Network, section: Section, vehicle: Vehicle,
                  carried: list[Item], beta0_s: float, lam: float) -> float:
    """D(u, sigma) at the vehicle's section-optimal speed. Def 7.4."""
    delta = useful_displacement(net, section, carried)
    if delta == 0.0:
        return 0.0
    _, ratio = section_optimal_speed(section, vehicle, delta, beta0_s, lam)
    return ratio


# ----------------------------------------------------------------------
# Separation cost (Def 4.1) by brute-force over configurations
# ----------------------------------------------------------------------

def enumerate_configurations(net: Network, vehicles: dict[str, Vehicle],
                             items: dict[str, Item], speed_grid: int = 6):
    """Yield an iterable of all candidate configurations for a small network.

    For each section, the candidate assignments are: idle, or (vehicle, speed, carried-set)
    for each vehicle that can serve it, a coarse speed grid, and each admissible carried set
    (items currently at the section tail whose move is target-ward), respecting capacity.
    Small networks only.
    """
    section_choices = []  # list over sections of list of candidate (Assignment|None)

    for sec in net.sections:
        choices: list[Assignment | None] = [None]  # idle
        # An item is eligible to be carried on `sec` if its origin can reach the
        # section's tail (possibly via earlier hops) and the section is
        # target-ward for it. This lets a single-window configuration represent
        # a multi-hop journey (origin -> ... -> sec.tail -> sec.head -> ...).
        tail_items = [x for x in items.values()
                      if (x.loc == sec.tail or net._reachable(x.loc, sec.tail))]
        for veh in vehicles.values():
            ceiling = capability_ceiling(veh, sec)
            speeds = [ceiling * (i + 1) / speed_grid for i in range(speed_grid)]
            # admissible carried sets: subsets of tail items up to capacity, target-ward
            targetward = [x for x in tail_items
                          if net.rho(sec.head, x.target) < net.rho(sec.tail, x.target)]
            # enumerate subsets up to capacity (small counts only)
            subsets = _subsets_upto(targetward, veh.capacity)
            for v in speeds:
                for sub in subsets:
                    if not sub:
                        continue  # carrying nothing produces no yield; skip empty
                    choices.append(Assignment(sec.name, veh.name, v,
                                              tuple(x.name for x in sub)))
        section_choices.append(choices)

    for combo in product(*section_choices):
        yield Configuration({net.sections[i].name: combo[i]
                             for i in range(len(net.sections))})


def _subsets_upto(elems, k):
    """All subsets of elems of size 0..k (k small)."""
    from itertools import combinations
    out = [()]
    for size in range(1, min(k, len(elems)) + 1):
        out.extend(combinations(elems, size))
    return [list(s) for s in out]


def brute_force_optimum(net: Network, vehicles: dict[str, Vehicle],
                        items: dict[str, Item], lam: float, speed_grid: int = 6):
    """Global yield optimum Y* and an argmax configuration. Def 3.2 / Thm 6.4(i)."""
    best_y, best_c = -np.inf, None
    for c in enumerate_configurations(net, vehicles, items, speed_grid):
        y = configuration_yield(net, c, vehicles, items, lam)
        if y > best_y:
            best_y, best_c = y, c
    return best_y, best_c


def _delivers_all(net: Network, config: Configuration, items: dict[str, Item],
                  target_station: str) -> bool:
    """Does this configuration move every item to `target_station`?

    We trace each item: it starts at its loc; a section carrying it advances it
    to that section's head. With one window per section, an item is delivered if
    there is a carried chain from its origin to the target. For the small
    two-hop corridors used here we check reachability via carried sections.
    """
    # Build carried adjacency: item -> set of sections carrying it
    carried_sections = {nm: [] for nm in items}
    for sec_name, a in config.assignments.items():
        if a is None:
            continue
        for it in a.items:
            carried_sections[it].append(sec_name)
    sec_by_name = {s.name: s for s in net.sections}
    for nm, x in items.items():
        # trace from origin following carried sections toward target
        loc = x.loc
        advanced = True
        guard = 0
        while loc != target_station and advanced and guard < len(net.sections) + 2:
            advanced = False
            guard += 1
            for sc in carried_sections[nm]:
                s = sec_by_name[sc]
                if s.tail == loc and net.rho(s.head, target_station) < net.rho(loc, target_station):
                    loc = s.head
                    advanced = True
                    break
        if loc != target_station:
            return False
    return True


def brute_force_optimum_delivering(net: Network, vehicles: dict[str, Vehicle],
                                   items: dict[str, Item], lam: float,
                                   target_station: str, speed_grid: int = 6):
    """Yield-optimal configuration among those that deliver all items to target.

    This is the optimum for a fixed transport task (Def 3.2 restricted to
    task-completing configurations), avoiding the degenerate single-block-second
    ratio optimum.
    """
    best_y, best_c = -np.inf, None
    for c in enumerate_configurations(net, vehicles, items, speed_grid):
        if not _delivers_all(net, c, items, target_station):
            continue
        y = configuration_yield(net, c, vehicles, items, lam)
        if y > best_y:
            best_y, best_c = y, c
    return best_y, best_c


def separation_cost(net: Network, vehicles: dict[str, Vehicle],
                    items: dict[str, Item], lam: float, section_name: str,
                    y_star: float, speed_grid: int = 6,
                    target_station: str | None = None) -> float:
    """varsigma(sigma) = Y* - Y(C_{-sigma}). Def 4.1.

    Y(C_{-sigma}) is the best yield achievable without using the section. If
    `target_station` is given, the avoiding configuration must still deliver all
    items to target (separation cost of the task, not of an arbitrary config);
    if no delivering config avoids the section, the section is essential and its
    separation cost is Y* (avoiding yield 0).
    """
    best_avoid = -np.inf
    for c in enumerate_configurations(net, vehicles, items, speed_grid):
        if c.assignments.get(section_name) is not None:
            continue  # must avoid the section
        if target_station is not None and not _delivers_all(net, c, items, target_station):
            continue
        y = configuration_yield(net, c, vehicles, items, lam)
        if y > best_avoid:
            best_avoid = y
    if best_avoid == -np.inf:
        best_avoid = 0.0  # no (delivering) config avoids it; section is essential
    return y_star - best_avoid

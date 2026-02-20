from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from collections import defaultdict
from typing import MutableMapping, Collection

import torch

class Constraint(Flag):
    MIN_POLY = auto()
    MAX_POLY = auto()
    MIN_DUR = auto()
    MAX_DUR = auto()
    MIN_TIME = auto()
    MAX_TIME = auto()

_default_penalty = {
    Constraint.MIN_POLY:1,
    Constraint.MAX_POLY:3,
    Constraint.MIN_DUR:2,
    Constraint.MAX_DUR:2,
    Constraint.MIN_TIME:10,
    Constraint.MAX_TIME:0.5,
}

# map from instrument to pitch set
# ugly because Python invariant typing is annoying?
# NoteMap = MutableMapping[int,Collection[int]] | dict[int,set[int]]
NoteMap = dict[int,set[int]]|dict[int,Collection[int]]

@dataclass
class EventConstraints:
    note_on_map:NoteMap|None = None # inst:[pitch]
    note_off_map:NoteMap|None = None
    min_time:float = -torch.inf
    max_time:float = torch.inf
    min_vel:float = -torch.inf
    max_vel:float = torch.inf
    min_polyphony:MutableMapping[int,int]|int = 0 
    max_polyphony:MutableMapping[int,int]|int = 128
    min_duration:MutableMapping[int,float]|float = 0 
    max_duration:MutableMapping[int,float]|float = float('inf') 
    # penalty specifies the relative importance of different constraints
    penalty:MutableMapping[Constraint,float] = field(
        default_factory=_default_penalty.copy)
    # steering
    pitch_temp:float = 1.0
    rhythm_temp:float = 1.0 
    timing_temp:float = 1.0
    truncate_quantile_time:tuple[float,float] = (0.,1.)
    truncate_quantile_pitch:tuple[float,float] = (0.,1.)
    truncate_quantile_vel:tuple[float,float] = (0.,1.)
    steer_density:float = 0.5
    inst_weights:MutableMapping[int,float] = field(default_factory=dict)
    # set of externally controlled instruments which may not obey constraints
    external:set[int] = field(default_factory=set)

    def get(self, a:str, i:int):
        """get from fields which may be scalar or instrument dict"""
        field = getattr(self, a)
        if isinstance(field, dict):
            return field.get(i, EventConstraints.__dataclass_fields__[a].default)
        else:
            return field


@dataclass
class SupportRange:
    lo:float = -torch.inf
    hi:float = torch.inf
    weight:float = 1.0
    # def __init__(self, lo:float=None, hi:float=None):
        # self.lo = float('-inf') if lo is None else lo
        # self.hi = float('inf') if hi is None else hi

    def __repr__(self):
        return f'SupportRange({self.lo}, {self.hi})'
    
    def __contains__(self, item): 
        return item<self.hi and item>=self.lo
    
    def __and__(self, other):
        return SupportRange(max(self.lo, other.lo), min(self.hi, other.hi))
    
    def __or__(self, other):
        # return SupportRange(min(self.lo, other.lo), max(self.hi, other.hi))
        return SupportMultiRange([self, other])
    
    def size(self):
        return self.hi - self.lo
    
    def bounds(self):
        return self.lo, self.hi
    
    def copy(self):
        return SupportRange(self.lo, self.hi, self.weight)
    
    def value(self):
        if self.lo==self.hi:
            return self.lo
        return None
    
class SupportMultiRange:
    def __init__(self, ranges):
        self.ranges = ranges
        self.normalize()

    def normalize(self):
        """merge any overlapping ranges"""
        complete = []
        working = self.ranges
        remaining = []
        while len(working):
            r = working.pop()
            others = working.copy()
            while len(others):
                r_other = others.pop()
                # if they overlap, merge into r
                if (r & r_other).size():
                    r = SupportRange(min(r.lo, r_other.lo), max(r.hi, r_other.hi))
                else:
                    remaining.append(r_other)
            complete.append(r)
            working = remaining
        self.ranges = complete

    def __or__(self, other):
        if isinstance(other, SupportMultiRange):
            return SupportMultiRange(self.ranges + other.ranges)
        elif isinstance(other, SupportRange):
            return SupportMultiRange(self.ranges + [other])
        else: raise TypeError
    
    def __contains__(self, item) -> bool:
        return any(item in r for r in self.ranges)
    
    def empty(self) -> bool:
        return not any(r.size()>=0 for r in self.ranges)
    
    def value(self) -> float|None:
        values = []
        for r in self.ranges:
            v = r.value()
            if v is not None:
                values.append(v)
        if len(values)==1:
            return next(iter(values))
        return None


@dataclass 
class SupportAtom:
    inst: set
    pitch: set
    time: SupportRange
    vel: SupportRange
    penalty: float = 0
    constraint: Constraint = Constraint(0)

class Support:
    def __init__(self):
        self.atoms: list[SupportAtom] = []

    def add(self, atom:SupportAtom):
        self.atoms.append(atom)

    def empty(self):
        return not len(self.atoms)

    # when sampling, atoms need to be filtered
    # based on whether they match the sample
    def apply_inst(self, i):
        self.atoms = list(filter(lambda a: i in a.inst, self.atoms))

    def apply_pitch(self, p):
        self.atoms = list(filter(lambda a: p in a.pitch, self.atoms))

    def apply_time(self, t):
        self.atoms = list(filter(lambda a: t in a.time, self.atoms))

    def apply_vel(self, v):
        self.atoms = list(filter(lambda a: v in a.vel, self.atoms))

    def penalize_later(self, t, penalty, constraint):
        new_atoms = []
        for atom in self.atoms:
            # TODO look at closed/open interval here...
            if t > atom.time.lo:
                new_atoms.append(SupportAtom(
                    atom.inst, atom.pitch, 
                    SupportRange(atom.time.lo, min(t, atom.time.hi)), 
                    atom.vel, atom.penalty, atom.constraint))
            if t < atom.time.hi:
                new_atoms.append(SupportAtom(
                    set(atom.inst), set(atom.pitch), 
                    SupportRange(max(t, atom.time.lo), atom.time.hi), 
                    atom.vel.copy(), 
                    atom.penalty+penalty, atom.constraint|constraint))
        self.atoms = new_atoms

    def penalize_earlier_noteoff(self, t, penalty, pitch, inst):
        new_atoms = []
        for atom in self.atoms:
            # leave any non-matching atoms alone
            if atom.vel.lo>=0.5 or pitch not in atom.pitch or inst not in atom.inst:
                new_atoms.append(atom)
                continue

            # the old atom splits into four parts,
            # two which are left alone, 
            # two with just the target inst,pitch split again on time
            
            # with target pitch removed
            atom.pitch.remove(pitch)
            if len(atom.pitch):
                new_atoms.append(SupportAtom(
                        atom.inst, atom.pitch,#-{pitch},
                        atom.time, atom.vel,
                        atom.penalty, atom.constraint
                    ))
            # preserving that pitch on other instruments
            # note this disappears when instruments are already separate
            preserve_inst = atom.inst-{inst}
            if len(preserve_inst):
                new_atoms.append(SupportAtom(
                    preserve_inst, {pitch},
                    atom.time.copy(), atom.vel.copy(),
                    atom.penalty, atom.constraint
                ))

            # now split on time
            if t > atom.time.lo:
                new_atoms.append(SupportAtom(
                    {inst}, {pitch},
                    SupportRange(atom.time.lo, min(t, atom.time.hi)), 
                    atom.vel.copy(), 
                    atom.penalty+penalty, atom.constraint|Constraint.MIN_DUR))
            if t < atom.time.hi:
                new_atoms.append(SupportAtom(
                    {inst}, {pitch},
                    SupportRange(max(t, atom.time.lo), atom.time.hi), 
                    atom.vel.copy(), atom.penalty, atom.constraint))
        self.atoms = new_atoms

    def remove_off(self):
        # assuming on and off are already disjoint!
        self.atoms = list(filter(lambda a: a.vel.lo>=0.5, self.atoms))

    def penalize_off(self, inst, penalty=1.):
        # assuming on and off are already disjoint
        # assuming instruments are already disjoint
        for a in self.atoms:
            if inst in a.inst and a.vel.hi <= 0.5:
                a.penalty += penalty
                a.constraint |= Constraint.MIN_POLY

    def penalize_on(self, inst, penalty=1.):
        # assuming on and off are already disjoint
        # assuming instruments are already disjoint
        for a in self.atoms:
            if inst in a.inst and a.vel.lo >= 0.5:
                a.penalty += penalty
                a.constraint |= Constraint.MAX_POLY

    def __str__(self):
        s = []
        for a in self.atoms:
            on = a.vel.lo >= 0.5
            ps = len(a.pitch) if len(a.pitch) > 3 else a.pitch
            s .append(f'{a.inst} ({ps} {on}) {a.time} {a.constraint} {a.penalty}')
        return '\n'.join(s)

    def stratify(self):
        if len(self.atoms):
            stratum = min(a.penalty for a in self.atoms)
            if stratum:
                print(self)
                print(f'breaking constraints {stratum=} {set(a.constraint for a in self.atoms)}')
            self.atoms = list(filter(lambda a: a.penalty==stratum, self.atoms))

    def marginal_inst(self):
        if not len(self.atoms):
            return set()
        m = set(self.atoms[0].inst)
        for a in self.atoms[1:]:
            m |= a.inst
        return m
    
    def marginal_pitch(self):
        if not len(self.atoms):
            return set()
        m = set(self.atoms[0].pitch)
        for a in self.atoms[1:]:
            m |= a.pitch 
        return m
    
    def marginal_time(self):
        if not len(self.atoms):
            return SupportMultiRange([])
        m = SupportMultiRange([self.atoms[0].time])
        for a in self.atoms[1:]:
            m = m | a.time 
        return m
    
    def marginal_vel(self):
        if not len(self.atoms):
            return SupportMultiRange([])
        m = SupportMultiRange([self.atoms[0].vel])
        for a in self.atoms[1:]:
            m = m | a.vel 
        return m
    
    
@dataclass
class NotochordEvent():
    inst: int|None = None
    pitch: int|None = None
    time: float|None = None
    vel: float|None = None
    support: Support = field(default_factory=Support)

    def is_complete(self):
        return (
            self.inst is not None and 
            self.pitch is not None and
            self.time is not None and 
            self.vel is not None)
        
    def modality(self, m):
        if m=='i': return self.inst
        elif m=='p': return self.pitch
        elif m=='t': return self.time
        elif m=='v': return self.vel
        raise ValueError(f'unknown modality {m}')
    
    def set(self, m, value):
        """set the value of a modality and remove any incompatible support"""
        if m=='i': 
            self.inst = value
            self.support.apply_inst(value)
        elif m=='p': 
            self.pitch = value
            self.support.apply_pitch(value)
        elif m=='t': 
            self.time = value
            self.support.apply_time(value)
        elif m=='v': 
            self.vel = value
            self.support.apply_vel(value)
        else:
            raise ValueError
        
    def autoset(self):
        """check if support has been reduced to a single possibility for any modality, and set it"""
        # this while/continue/break structure
        # ensures that if one modality gets set,
        # the others get re-checked
        while True:
            if self.inst is None:
                i = self.support.marginal_inst()
                if len(i)==1:
                    self.set('i', next(iter(i)))
                    print(f'inst determined by constraint for {self}')
                    continue
            if self.pitch is None:
                p = self.support.marginal_pitch()
                if len(p)==1:
                    self.set('p', next(iter(p)))
                    print(f'pitch determined by constraint for {self}')
                    continue
            if self.time is None:
                t = self.support.marginal_time()
                val = t.value()
                if val is not None:
                    self.set('t', val)
                    print(f'time determined by constraint for {self}')
                    continue
            if self.vel is None:
                v = self.support.marginal_vel()
                val = v.value()
                if all(r.hi <= 0.5 for r in v.ranges):
                    val = 0
                if val is not None:
                    self.set('v', val)
                    print(f'vel determined by constraint for {self}')
                    continue
            break
            
        
    # compatibility with dict-style event
    # def __contains__(self, k):
    #     return getattr(self, k, None) is not None
        
    # def __getitem__(self, k):
    #     if k in self:
    #         return getattr(self, k)
    #     raise KeyError
    
    # def get(self, k, default=None):
    #     if k not in self:
    #         return self[k]
    #     return default



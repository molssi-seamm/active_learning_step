# -*- coding: utf-8 -*-

"""Covalent radii (Å) as molsystem uses them for bond perception.

Generated from ``mendeleev`` (version 1.3.0): ``covalent_radius_pyykko``, falling
back to ``covalent_radius``, divided by 100 (pm to Å), exactly as
``molsystem.topology.TopologyMixin._covalent_radii``. Kept as a table so that
:mod:`active_learning_step.molecules` needs neither mendeleev nor molsystem.
"""

#: Atomic number -> covalent radius in Å (``None`` when mendeleev has none).
COVALENT_RADII = {
    1: 0.32,  # H
    2: 0.46,  # He
    3: 1.33,  # Li
    4: 1.02,  # Be
    5: 0.85,  # B
    6: 0.75,  # C
    7: 0.71,  # N
    8: 0.63,  # O
    9: 0.64,  # F
    10: 0.67,  # Ne
    11: 1.55,  # Na
    12: 1.39,  # Mg
    13: 1.26,  # Al
    14: 1.16,  # Si
    15: 1.11,  # P
    16: 1.03,  # S
    17: 0.99,  # Cl
    18: 0.96,  # Ar
    19: 1.96,  # K
    20: 1.71,  # Ca
    21: 1.48,  # Sc
    22: 1.36,  # Ti
    23: 1.34,  # V
    24: 1.22,  # Cr
    25: 1.19,  # Mn
    26: 1.16,  # Fe
    27: 1.11,  # Co
    28: 1.1,  # Ni
    29: 1.12,  # Cu
    30: 1.18,  # Zn
    31: 1.24,  # Ga
    32: 1.21,  # Ge
    33: 1.21,  # As
    34: 1.16,  # Se
    35: 1.14,  # Br
    36: 1.17,  # Kr
    37: 2.1,  # Rb
    38: 1.85,  # Sr
    39: 1.63,  # Y
    40: 1.54,  # Zr
    41: 1.47,  # Nb
    42: 1.38,  # Mo
    43: 1.28,  # Tc
    44: 1.25,  # Ru
    45: 1.25,  # Rh
    46: 1.2,  # Pd
    47: 1.28,  # Ag
    48: 1.36,  # Cd
    49: 1.42,  # In
    50: 1.4,  # Sn
    51: 1.4,  # Sb
    52: 1.36,  # Te
    53: 1.33,  # I
    54: 1.31,  # Xe
    55: 2.32,  # Cs
    56: 1.96,  # Ba
    57: 1.8,  # La
    58: 1.63,  # Ce
    59: 1.76,  # Pr
    60: 1.74,  # Nd
    61: 1.73,  # Pm
    62: 1.72,  # Sm
    63: 1.68,  # Eu
    64: 1.69,  # Gd
    65: 1.68,  # Tb
    66: 1.67,  # Dy
    67: 1.66,  # Ho
    68: 1.65,  # Er
    69: 1.64,  # Tm
    70: 1.7,  # Yb
    71: 1.62,  # Lu
    72: 1.52,  # Hf
    73: 1.46,  # Ta
    74: 1.37,  # W
    75: 1.31,  # Re
    76: 1.29,  # Os
    77: 1.22,  # Ir
    78: 1.23,  # Pt
    79: 1.24,  # Au
    80: 1.33,  # Hg
    81: 1.44,  # Tl
    82: 1.44,  # Pb
    83: 1.51,  # Bi
    84: 1.45,  # Po
    85: 1.47,  # At
    86: 1.42,  # Rn
    87: 2.23,  # Fr
    88: 2.01,  # Ra
    89: 1.86,  # Ac
    90: 1.75,  # Th
    91: 1.69,  # Pa
    92: 1.7,  # U
    93: 1.71,  # Np
    94: 1.72,  # Pu
    95: 1.66,  # Am
    96: 1.66,  # Cm
    97: 1.68,  # Bk
    98: 1.68,  # Cf
    99: 1.65,  # Es
    100: 1.67,  # Fm
    101: 1.73,  # Md
    102: 1.76,  # No
    103: 1.61,  # Lr
    104: 1.57,  # Rf
    105: 1.49,  # Db
    106: 1.43,  # Sg
    107: 1.41,  # Bh
    108: 1.34,  # Hs
    109: 1.29,  # Mt
    110: 1.28,  # Ds
    111: 1.21,  # Rg
    112: 1.22,  # Cn
    113: 1.36,  # Nh
    114: 1.43,  # Fl
    115: 1.62,  # Mc
    116: 1.75,  # Lv
    117: 1.65,  # Ts
    118: 1.57,  # Og
}

import numpy as np
import pandas as pd


def material_compliance(material, s):
    data = {'material': ['Si', 'Ge', 'C'],
            'lattice spacing (m)': [0.00000000054309402, 0.000000000565735, 0.000000000356679],
            's11': [0.768, 0.964, 0.0949],
            's12': [-0.214, -0.26, -0.00978],
            's44': [1.26, 1.49, 0.17]}
    data = pd.DataFrame(data)
    data = data.set_index('material')
    s = data.loc[material, s]
    return s


def normalize_vector(v):
    v_sum = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return np.array([v[0] / v_sum, v[1] / v_sum, v[2] / v_sum])


def cross_vector(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.cross(v1, v2)


def calculator(material, reflection_hkl, bending_hkl, chi=0):
    """
    References: Wortman, J. J., & Evans, R. A. (1965). Young’s modulus, shear modulus, and poisson’s ratio in silicon
    and germanium. Journal of Applied Physics, 36(1), 153–156. https://doi.org/10.1063/1.1713863

    Parameters
    ----------
    material: 'Si', 'Ge', 'C'
    reflection_hkl: [h, k, l]
    bending_hkl: [h, k, l]
    chi: Degree.  Asymmetric angle from reflection plane to surface normal (counterclockwise positive)

    Returns
    -------
    Poisson's ratio on ZX direction.
    """
    chi = np.radians(chi)
    # XR, YR, ZR for Reflection related dimensions
    ZR = np.array(reflection_hkl)
    Y = np.array(bending_hkl)
    XR = np.cross(Y, ZR)

    # **n for NORMALIZED vector
    XRn = normalize_vector(XR)
    # Yn = normalize_vector(Y)
    ZRn = normalize_vector(ZR)

    # X, Y, Z for crystal orientation related dimensions
    X = XRn * np.cos(chi) - ZRn * np.sin(chi)
    Z = np.cross(X, Y)

    # *n for NORMALIZED vector
    Xn = normalize_vector(X)
    Yn = normalize_vector(Y)
    Zn = normalize_vector(Z)

    # sc** for Compliance coefficient
    sc11 = material_compliance(material, 's11')
    sc12 = material_compliance(material, 's12')
    sc44 = material_compliance(material, 's44')
    sc = sc11 - sc12 - 0.5 * sc44

    s13 = sc12 + sc * (Xn[0] ** 2 * Zn[0] ** 2 + Xn[1] ** 2 * Zn[1] ** 2 + Xn[2] ** 2 * Zn[2] ** 2)
    s33 = sc11 + sc * (Zn[0] ** 4 + Zn[1] ** 4 + Zn[2] ** 4 - 1)
    sp31 = s13 / s33

    c_factor = 1.0
    nu = -sp31 * c_factor  # (\nu ZX)
    return nu


if __name__ == '__main__':
    calculator('Si', [1, 1, -1], [1, -1, 0], chi=0)

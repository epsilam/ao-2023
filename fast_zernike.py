# MIT License

# Copyright (c) 2020 Timothy Gebhard

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################

# This code implements fast versions of the derivatives of Zernike polynomials 
# in Cartesian coordinates. It has been shamelessly ripped from 
# https://github.com/timothygebhard/hswfs

from typing import no_type_check, Union, Tuple

import numpy as np
from numpy import ndarray, ones, sqrt, sin, cos, arctan2 as atan2
from sympy import Symbol

def j_to_mn(j: int) -> Tuple[int, int]:
    r"""
    Map the index :math:`j` of the single-indexing scheme to the
    corresponding indices :math:`m, n` from the double-indexing scheme.

    Mathematically, the mapping is given by:

    .. math::
        n = \left\lceil (-3 + \sqrt{9 + 8 j})\, /\, 2 \right\rceil
        \quad \text{and} \quad
        m = 2 j - n \cdot (n + 2)

    Args:
        j: Index :math:`j` of :math:`Z_j`.

    Returns:
        The pair of indices :math:`m, n` which correspond to :math:`j`.
    """

    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = int(2 * j - n * (n + 2))

    return m, n

@no_type_check
def zernike_derivative_cartesian(
    m: int,
    n: int,
    x: Union[float, ndarray],
    y: Union[float, ndarray],
    wrt: Union[str, Symbol],
) -> Union[float, ndarray]:
    r"""
    Evaluate the Cartesian derivative of a the Zernike polynomial
    :math:`Z^m_n` at the position(s) `x`, `y`. Fast.

    Args:
        m: Index :math:`m` of :math:`Z^m_n`.
        n: Index :math:`n` of :math:`Z^m_n`.
            Default maximum value is :math:`n_\text{max} = 15`.
        x: The x-coordinate(s) at which to evaluate the derivative.
        y: The y-coordinate(s) at which to evaluate the derivative.
        wrt: A string or `sy.Symbol` that specifies with respect to
            which variable the derivative is taken: `"x"` or `"y"`.

    Returns:
        The value(s) of the derivative of :math:`Z^m_n` with
        respect to `wrt` at the given position(s) `x`, `y`.
    """

    # Derivatives for j = 0
    if m == 0 and n == 0 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == 0 and n == 0 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 1
    if m == -1 and n == 1 and wrt == "x":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0
    if m == -1 and n == 1 and wrt == "y":
        if isinstance(x, ndarray):
            return 2 * ones(x.shape)
        return 2

    # Derivatives for j = 2
    if m == 1 and n == 1 and wrt == "x":
        if isinstance(x, ndarray):
            return 2 * ones(x.shape)
        return 2
    if m == 1 and n == 1 and wrt == "y":
        if isinstance(x, ndarray):
            return 0 * ones(x.shape)
        return 0

    # Derivatives for j = 3
    if m == -2 and n == 2 and wrt == "x":
        return 2 * sqrt(6) * y
    if m == -2 and n == 2 and wrt == "y":
        return 2 * sqrt(6) * x

    # Derivatives for j = 4
    if m == 0 and n == 2 and wrt == "x":
        return 4 * sqrt(3) * x
    if m == 0 and n == 2 and wrt == "y":
        return 4 * sqrt(3) * y

    # Derivatives for j = 5
    if m == 2 and n == 2 and wrt == "x":
        return 2 * sqrt(6) * x
    if m == 2 and n == 2 and wrt == "y":
        return -2 * sqrt(6) * y

    # Derivatives for j = 6
    if m == -3 and n == 3 and wrt == "x":
        return (
            6
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (x * sin(3 * atan2(y, x)) - y * cos(3 * atan2(y, x)))
        )
    if m == -3 and n == 3 and wrt == "y":
        return (
            6
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (x * cos(3 * atan2(y, x)) + y * sin(3 * atan2(y, x)))
        )

    # Derivatives for j = 7
    if m == -1 and n == 3 and wrt == "x":
        return 12 * sqrt(2) * x * y
    if m == -1 and n == 3 and wrt == "y":
        return sqrt(2) * (6 * x**2 + 18 * y**2 - 4)

    # Derivatives for j = 8
    if m == 1 and n == 3 and wrt == "x":
        return sqrt(2) * (18 * x**2 + 6 * y**2 - 4)
    if m == 1 and n == 3 and wrt == "y":
        return 12 * sqrt(2) * x * y

    # Derivatives for j = 9
    if m == 3 and n == 3 and wrt == "x":
        return (
            6
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (x * cos(3 * atan2(y, x)) + y * sin(3 * atan2(y, x)))
        )
    if m == 3 and n == 3 and wrt == "y":
        return (
            6
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (-x * sin(3 * atan2(y, x)) + y * cos(3 * atan2(y, x)))
        )

    # Derivatives for j = 10
    if m == -4 and n == 4 and wrt == "x":
        return 4 * sqrt(10) * y * (3 * x**2 - y**2)
    if m == -4 and n == 4 and wrt == "y":
        return 4 * sqrt(10) * x * (x**2 - 3 * y**2)

    # Derivatives for j = 11
    if m == -2 and n == 4 and wrt == "x":
        return 2 * sqrt(10) * y * (12 * x**2 + 4 * y**2 - 3)
    if m == -2 and n == 4 and wrt == "y":
        return 2 * sqrt(10) * x * (4 * x**2 + 12 * y**2 - 3)

    # Derivatives for j = 12
    if m == 0 and n == 4 and wrt == "x":
        return 12 * sqrt(5) * x * (2 * x**2 + 2 * y**2 - 1)
    if m == 0 and n == 4 and wrt == "y":
        return 12 * sqrt(5) * y * (2 * x**2 + 2 * y**2 - 1)

    # Derivatives for j = 13
    if m == 2 and n == 4 and wrt == "x":
        return 2 * sqrt(10) * x * (8 * x**2 - 3)
    if m == 2 and n == 4 and wrt == "y":
        return 2 * sqrt(10) * y * (3 - 8 * y**2)

    # Derivatives for j = 14
    if m == 4 and n == 4 and wrt == "x":
        return 4 * sqrt(10) * x * (x**2 - 3 * y**2)
    if m == 4 and n == 4 and wrt == "y":
        return 4 * sqrt(10) * y * (-3 * x**2 + y**2)

    # Derivatives for j = 15
    if m == -5 and n == 5 and wrt == "x":
        return (
            10
            * sqrt(3)
            * (x**2 + y**2) ** (3 / 2)
            * (x * sin(5 * atan2(y, x)) - y * cos(5 * atan2(y, x)))
        )
    if m == -5 and n == 5 and wrt == "y":
        return (
            10
            * sqrt(3)
            * (x**2 + y**2) ** (3 / 2)
            * (x * cos(5 * atan2(y, x)) + y * sin(5 * atan2(y, x)))
        )

    # Derivatives for j = 16
    if m == -3 and n == 5 and wrt == "x":
        return sqrt(3 * x**2 + 3 * y**2) * (
            2 * x * (25 * x**2 + 25 * y**2 - 12) * sin(3 * atan2(y, x))
            - 6 * y * (5 * x**2 + 5 * y**2 - 4) * cos(3 * atan2(y, x))
        )
    if m == -3 and n == 5 and wrt == "y":
        return sqrt(3 * x**2 + 3 * y**2) * (
            6 * x * (5 * x**2 + 5 * y**2 - 4) * cos(3 * atan2(y, x))
            + 2 * y * (25 * x**2 + 25 * y**2 - 12) * sin(3 * atan2(y, x))
        )

    # Derivatives for j = 17
    if m == -1 and n == 5 and wrt == "x":
        return 16 * sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    if m == -1 and n == 5 and wrt == "y":
        return sqrt(3) * (
            20 * x**4
            + 120 * x**2 * y**2
            - 24 * x**2
            + 100 * y**4
            - 72 * y**2
            + 6
        )

    # Derivatives for j = 18
    if m == 1 and n == 5 and wrt == "x":
        return sqrt(3) * (
            100 * x**4
            + 120 * x**2 * y**2
            - 72 * x**2
            + 20 * y**4
            - 24 * y**2
            + 6
        )
    if m == 1 and n == 5 and wrt == "y":
        return 16 * sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)

    # Derivatives for j = 19
    if m == 3 and n == 5 and wrt == "x":
        return sqrt(3 * x**2 + 3 * y**2) * (
            2 * x * (25 * x**2 + 25 * y**2 - 12) * cos(3 * atan2(y, x))
            + 6 * y * (5 * x**2 + 5 * y**2 - 4) * sin(3 * atan2(y, x))
        )
    if m == 3 and n == 5 and wrt == "y":
        return sqrt(3 * x**2 + 3 * y**2) * (
            -6 * x * (5 * x**2 + 5 * y**2 - 4) * sin(3 * atan2(y, x))
            + 2 * y * (25 * x**2 + 25 * y**2 - 12) * cos(3 * atan2(y, x))
        )

    # Derivatives for j = 20
    if m == 5 and n == 5 and wrt == "x":
        return (
            10
            * sqrt(3)
            * (x**2 + y**2) ** (3 / 2)
            * (x * cos(5 * atan2(y, x)) + y * sin(5 * atan2(y, x)))
        )
    if m == 5 and n == 5 and wrt == "y":
        return (
            10
            * sqrt(3)
            * (x**2 + y**2) ** (3 / 2)
            * (-x * sin(5 * atan2(y, x)) + y * cos(5 * atan2(y, x)))
        )

    # Derivatives for j = 21
    if m == -6 and n == 6 and wrt == "x":
        return (
            6
            * sqrt(14)
            * (x**2 + y**2) ** 2
            * (x * sin(6 * atan2(y, x)) - y * cos(6 * atan2(y, x)))
        )
    if m == -6 and n == 6 and wrt == "y":
        return (
            6
            * sqrt(14)
            * (x**2 + y**2) ** 2
            * (x * cos(6 * atan2(y, x)) + y * sin(6 * atan2(y, x)))
        )

    # Derivatives for j = 22
    if m == -4 and n == 6 and wrt == "x":
        return (
            4
            * sqrt(14)
            * y
            * (30 * x**4 - 15 * x**2 - 6 * y**4 + 5 * y**2)
        )
    if m == -4 and n == 6 and wrt == "y":
        return (
            4
            * sqrt(14)
            * x
            * (6 * x**4 - 5 * x**2 - 30 * y**4 + 15 * y**2)
        )

    # Derivatives for j = 23
    if m == -2 and n == 6 and wrt == "x":
        return (
            2
            * sqrt(14)
            * y
            * (
                75 * x**4
                + 90 * x**2 * y**2
                - 60 * x**2
                + 15 * y**4
                - 20 * y**2
                + 6
            )
        )
    if m == -2 and n == 6 and wrt == "y":
        return (
            2
            * sqrt(14)
            * x
            * (
                15 * x**4
                + 90 * x**2 * y**2
                - 20 * x**2
                + 75 * y**4
                - 60 * y**2
                + 6
            )
        )

    # Derivatives for j = 24
    if m == 0 and n == 6 and wrt == "x":
        return (
            24
            * sqrt(7)
            * x
            * (-5 * x**2 - 5 * y**2 + 5 * (x**2 + y**2) ** 2 + 1)
        )
    if m == 0 and n == 6 and wrt == "y":
        return (
            24
            * sqrt(7)
            * y
            * (-5 * x**2 - 5 * y**2 + 5 * (x**2 + y**2) ** 2 + 1)
        )

    # Derivatives for j = 25
    if m == 2 and n == 6 and wrt == "x":
        return (
            2
            * sqrt(14)
            * x
            * (
                45 * x**4
                + 30 * x**2 * y**2
                - 40 * x**2
                - 15 * y**4
                + 6
            )
        )
    if m == 2 and n == 6 and wrt == "y":
        return (
            2
            * sqrt(14)
            * y
            * (
                15 * x**4
                - 30 * x**2 * y**2
                - 45 * y**4
                + 40 * y**2
                - 6
            )
        )

    # Derivatives for j = 26
    if m == 4 and n == 6 and wrt == "x":
        return (
            4
            * sqrt(14)
            * x
            * (
                9 * x**4
                - 30 * x**2 * y**2
                - 5 * x**2
                - 15 * y**4
                + 15 * y**2
            )
        )
    if m == 4 and n == 6 and wrt == "y":
        return (
            4
            * sqrt(14)
            * y
            * (
                -15 * x**4
                - 30 * x**2 * y**2
                + 15 * x**2
                + 9 * y**4
                - 5 * y**2
            )
        )

    # Derivatives for j = 27
    if m == 6 and n == 6 and wrt == "x":
        return (
            6
            * sqrt(14)
            * (x**2 + y**2) ** 2
            * (x * cos(6 * atan2(y, x)) + y * sin(6 * atan2(y, x)))
        )
    if m == 6 and n == 6 and wrt == "y":
        return (
            6
            * sqrt(14)
            * (x**2 + y**2) ** 2
            * (-x * sin(6 * atan2(y, x)) + y * cos(6 * atan2(y, x)))
        )

    # Derivatives for j = 28
    if m == -7 and n == 7 and wrt == "x":
        return (
            28
            * (x**2 + y**2) ** (5 / 2)
            * (x * sin(7 * atan2(y, x)) - y * cos(7 * atan2(y, x)))
        )
    if m == -7 and n == 7 and wrt == "y":
        return (
            28
            * (x**2 + y**2) ** (5 / 2)
            * (x * cos(7 * atan2(y, x)) + y * sin(7 * atan2(y, x)))
        )

    # Derivatives for j = 29
    if m == -5 and n == 7 and wrt == "x":
        return (
            4
            * (x**2 + y**2) ** (3 / 2)
            * (
                x * (49 * x**2 + 49 * y**2 - 30) * sin(5 * atan2(y, x))
                - 5 * y * (7 * x**2 + 7 * y**2 - 6) * cos(5 * atan2(y, x))
            )
        )
    if m == -5 and n == 7 and wrt == "y":
        return (
            4
            * (x**2 + y**2) ** (3 / 2)
            * (
                5 * x * (7 * x**2 + 7 * y**2 - 6) * cos(5 * atan2(y, x))
                + y * (49 * x**2 + 49 * y**2 - 30) * sin(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 30
    if m == -3 and n == 7 and wrt == "x":
        return (
            12
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -50 * x**2
                    - 50 * y**2
                    + 49 * (x**2 + y**2) ** 2
                    + 10
                )
                * sin(3 * atan2(y, x))
                - y
                * (
                    -30 * x**2
                    - 30 * y**2
                    + 21 * (x**2 + y**2) ** 2
                    + 10
                )
                * cos(3 * atan2(y, x))
            )
        )
    if m == -3 and n == 7 and wrt == "y":
        return (
            12
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -30 * x**2
                    - 30 * y**2
                    + 21 * (x**2 + y**2) ** 2
                    + 10
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    -50 * x**2
                    - 50 * y**2
                    + 49 * (x**2 + y**2) ** 2
                    + 10
                )
                * sin(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 31
    if m == -1 and n == 7 and wrt == "x":
        return (
            120
            * x
            * y
            * (
                7 * x**4
                + 14 * x**2 * y**2
                - 8 * x**2
                + 7 * y**4
                - 8 * y**2
                + 2
            )
        )
    if m == -1 and n == 7 and wrt == "y":
        return (
            140 * x**6
            + 1260 * x**4 * y**2
            - 240 * x**4
            + 2100 * x**2 * y**4
            - 1440 * x**2 * y**2
            + 120 * x**2
            + 980 * y**6
            - 1200 * y**4
            + 360 * y**2
            - 16
        )

    # Derivatives for j = 32
    if m == 1 and n == 7 and wrt == "x":
        return (
            980 * x**6
            + 2100 * x**4 * y**2
            - 1200 * x**4
            + 1260 * x**2 * y**4
            - 1440 * x**2 * y**2
            + 360 * x**2
            + 140 * y**6
            - 240 * y**4
            + 120 * y**2
            - 16
        )
    if m == 1 and n == 7 and wrt == "y":
        return (
            120
            * x
            * y
            * (
                7 * x**4
                + 14 * x**2 * y**2
                - 8 * x**2
                + 7 * y**4
                - 8 * y**2
                + 2
            )
        )

    # Derivatives for j = 33
    if m == 3 and n == 7 and wrt == "x":
        return (
            12
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -50 * x**2
                    - 50 * y**2
                    + 49 * (x**2 + y**2) ** 2
                    + 10
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    -30 * x**2
                    - 30 * y**2
                    + 21 * (x**2 + y**2) ** 2
                    + 10
                )
                * sin(3 * atan2(y, x))
            )
        )
    if m == 3 and n == 7 and wrt == "y":
        return (
            12
            * sqrt(x**2 + y**2)
            * (
                -x
                * (
                    -30 * x**2
                    - 30 * y**2
                    + 21 * (x**2 + y**2) ** 2
                    + 10
                )
                * sin(3 * atan2(y, x))
                + y
                * (
                    -50 * x**2
                    - 50 * y**2
                    + 49 * (x**2 + y**2) ** 2
                    + 10
                )
                * cos(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 34
    if m == 5 and n == 7 and wrt == "x":
        return (
            4
            * (x**2 + y**2) ** (3 / 2)
            * (
                x * (49 * x**2 + 49 * y**2 - 30) * cos(5 * atan2(y, x))
                + 5 * y * (7 * x**2 + 7 * y**2 - 6) * sin(5 * atan2(y, x))
            )
        )
    if m == 5 and n == 7 and wrt == "y":
        return (
            4
            * (x**2 + y**2) ** (3 / 2)
            * (
                -5 * x * (7 * x**2 + 7 * y**2 - 6) * sin(5 * atan2(y, x))
                + y * (49 * x**2 + 49 * y**2 - 30) * cos(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 35
    if m == 7 and n == 7 and wrt == "x":
        return (
            28
            * (x**2 + y**2) ** (5 / 2)
            * (x * cos(7 * atan2(y, x)) + y * sin(7 * atan2(y, x)))
        )
    if m == 7 and n == 7 and wrt == "y":
        return (
            28
            * (x**2 + y**2) ** (5 / 2)
            * (-x * sin(7 * atan2(y, x)) + y * cos(7 * atan2(y, x)))
        )

    # Derivatives for j = 36
    if m == -8 and n == 8 and wrt == "x":
        return (
            24
            * sqrt(2)
            * y
            * (
                7 * x**6
                - 35 * x**4 * y**2
                + 21 * x**2 * y**4
                - y**6
            )
        )
    if m == -8 and n == 8 and wrt == "y":
        return (
            24
            * sqrt(2)
            * x
            * (
                x**6
                - 21 * x**4 * y**2
                + 35 * x**2 * y**4
                - 7 * y**6
            )
        )

    # Derivatives for j = 37
    if m == -6 and n == 8 and wrt == "x":
        return (
            6
            * sqrt(2)
            * (x**2 + y**2) ** 2
            * (
                x * (32 * x**2 + 32 * y**2 - 21) * sin(6 * atan2(y, x))
                - 3 * y * (8 * x**2 + 8 * y**2 - 7) * cos(6 * atan2(y, x))
            )
        )
    if m == -6 and n == 8 and wrt == "y":
        return (
            6
            * sqrt(2)
            * (x**2 + y**2) ** 2
            * (
                3 * x * (8 * x**2 + 8 * y**2 - 7) * cos(6 * atan2(y, x))
                + y * (32 * x**2 + 32 * y**2 - 21) * sin(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 38
    if m == -4 and n == 8 and wrt == "x":
        return (
            12
            * sqrt(2)
            * y
            * (
                196 * x**6
                + 140 * x**4 * y**2
                - 210 * x**4
                - 84 * x**2 * y**4
                + 45 * x**2
                - 28 * y**6
                + 42 * y**4
                - 15 * y**2
            )
        )
    if m == -4 and n == 8 and wrt == "y":
        return (
            12
            * sqrt(2)
            * x
            * (
                28 * x**6
                + 84 * x**4 * y**2
                - 42 * x**4
                - 140 * x**2 * y**4
                + 15 * x**2
                - 196 * y**6
                + 210 * y**4
                - 45 * y**2
            )
        )

    # Derivatives for j = 39
    if m == -2 and n == 8 and wrt == "x":
        return (
            6
            * sqrt(2)
            * y
            * (
                2
                * x**2
                * (
                    120 * x**2
                    + 120 * y**2
                    + 224 * (x**2 + y**2) ** 3
                    - 315 * (x**2 + y**2) ** 2
                    - 10
                )
                + (x**2 - y**2)
                * (
                    10 * x**2
                    + 10 * y**2
                    - 56 * (x**2 + y**2) ** 4
                    + 105 * (x**2 + y**2) ** 3
                    - 60 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )
    if m == -2 and n == 8 and wrt == "y":
        return (
            6
            * sqrt(2)
            * x
            * (
                2
                * y**2
                * (
                    120 * x**2
                    + 120 * y**2
                    + 224 * (x**2 + y**2) ** 3
                    - 315 * (x**2 + y**2) ** 2
                    - 10
                )
                - (x**2 - y**2)
                * (
                    10 * x**2
                    + 10 * y**2
                    - 56 * (x**2 + y**2) ** 4
                    + 105 * (x**2 + y**2) ** 3
                    - 60 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 40
    if m == 0 and n == 8 and wrt == "x":
        return (
            120
            * x
            * (
                9 * x**2
                + 9 * y**2
                + 14 * (x**2 + y**2) ** 3
                - 21 * (x**2 + y**2) ** 2
                - 1
            )
        )
    if m == 0 and n == 8 and wrt == "y":
        return (
            120
            * y
            * (
                9 * x**2
                + 9 * y**2
                + 14 * (x**2 + y**2) ** 3
                - 21 * (x**2 + y**2) ** 2
                - 1
            )
        )

    # Derivatives for j = 41
    if m == 2 and n == 8 and wrt == "x":
        return (
            6
            * sqrt(2)
            * x
            * (
                224 * x**6
                + 336 * x**4 * y**2
                - 315 * x**4
                - 210 * x**2 * y**2
                + 120 * x**2
                - 112 * y**6
                + 105 * y**4
                - 10
            )
        )
    if m == 2 and n == 8 and wrt == "y":
        return (
            6
            * sqrt(2)
            * y
            * (
                112 * x**6
                - 105 * x**4
                - 336 * x**2 * y**4
                + 210 * x**2 * y**2
                - 224 * y**6
                + 315 * y**4
                - 120 * y**2
                + 10
            )
        )

    # Derivatives for j = 42
    if m == 4 and n == 8 and wrt == "x":
        return (
            12
            * sqrt(2)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    -42 * x**2
                    - 42 * y**2
                    + 28 * (x**2 + y**2) ** 2
                    + 15
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -63 * x**2
                    - 63 * y**2
                    + 56 * (x**2 + y**2) ** 2
                    + 15
                )
            )
            / (x**2 + y**2)
        )
    if m == 4 and n == 8 and wrt == "y":
        return (
            12
            * sqrt(2)
            * y
            * (
                -4
                * x**2
                * (x**2 - y**2)
                * (
                    -42 * x**2
                    - 42 * y**2
                    + 28 * (x**2 + y**2) ** 2
                    + 15
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -63 * x**2
                    - 63 * y**2
                    + 56 * (x**2 + y**2) ** 2
                    + 15
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 43
    if m == 6 and n == 8 and wrt == "x":
        return (
            6
            * sqrt(2)
            * (x**2 + y**2) ** 2
            * (
                x * (32 * x**2 + 32 * y**2 - 21) * cos(6 * atan2(y, x))
                + 3 * y * (8 * x**2 + 8 * y**2 - 7) * sin(6 * atan2(y, x))
            )
        )
    if m == 6 and n == 8 and wrt == "y":
        return (
            6
            * sqrt(2)
            * (x**2 + y**2) ** 2
            * (
                -3 * x * (8 * x**2 + 8 * y**2 - 7) * sin(6 * atan2(y, x))
                + y * (32 * x**2 + 32 * y**2 - 21) * cos(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 44
    if m == 8 and n == 8 and wrt == "x":
        return (
            24
            * sqrt(2)
            * x
            * (
                x**6
                - 21 * x**4 * y**2
                + 35 * x**2 * y**4
                - 7 * y**6
            )
        )
    if m == 8 and n == 8 and wrt == "y":
        return (
            24
            * sqrt(2)
            * y
            * (
                -7 * x**6
                + 35 * x**4 * y**2
                - 21 * x**2 * y**4
                + y**6
            )
        )

    # Derivatives for j = 45
    if m == -9 and n == 9 and wrt == "x":
        return (
            18
            * sqrt(5)
            * (x**2 + y**2) ** (7 / 2)
            * (x * sin(9 * atan2(y, x)) - y * cos(9 * atan2(y, x)))
        )
    if m == -9 and n == 9 and wrt == "y":
        return (
            18
            * sqrt(5)
            * (x**2 + y**2) ** (7 / 2)
            * (x * cos(9 * atan2(y, x)) + y * sin(9 * atan2(y, x)))
        )

    # Derivatives for j = 46
    if m == -7 and n == 9 and wrt == "x":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x * (81 * x**2 + 81 * y**2 - 56) * sin(7 * atan2(y, x))
                - 7 * y * (9 * x**2 + 9 * y**2 - 8) * cos(7 * atan2(y, x))
            )
        )
    if m == -7 and n == 9 and wrt == "y":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (5 / 2)
            * (
                7 * x * (9 * x**2 + 9 * y**2 - 8) * cos(7 * atan2(y, x))
                + y * (81 * x**2 + 81 * y**2 - 56) * sin(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 47
    if m == -5 and n == 9 and wrt == "x":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    -392 * x**2
                    - 392 * y**2
                    + 324 * (x**2 + y**2) ** 2
                    + 105
                )
                * sin(5 * atan2(y, x))
                - 5
                * y
                * (
                    -56 * x**2
                    - 56 * y**2
                    + 36 * (x**2 + y**2) ** 2
                    + 21
                )
                * cos(5 * atan2(y, x))
            )
        )
    if m == -5 and n == 9 and wrt == "y":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (3 / 2)
            * (
                5
                * x
                * (
                    -56 * x**2
                    - 56 * y**2
                    + 36 * (x**2 + y**2) ** 2
                    + 21
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    -392 * x**2
                    - 392 * y**2
                    + 324 * (x**2 + y**2) ** 2
                    + 105
                )
                * sin(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 48
    if m == -3 and n == 9 and wrt == "x":
        return (
            6
            * sqrt(5)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    175 * x**2
                    + 175 * y**2
                    + 252 * (x**2 + y**2) ** 3
                    - 392 * (x**2 + y**2) ** 2
                    - 20
                )
                * sin(3 * atan2(y, x))
                - y
                * (
                    105 * x**2
                    + 105 * y**2
                    + 84 * (x**2 + y**2) ** 3
                    - 168 * (x**2 + y**2) ** 2
                    - 20
                )
                * cos(3 * atan2(y, x))
            )
        )
    if m == -3 and n == 9 and wrt == "y":
        return (
            6
            * sqrt(5)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    105 * x**2
                    + 105 * y**2
                    + 84 * (x**2 + y**2) ** 3
                    - 168 * (x**2 + y**2) ** 2
                    - 20
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    175 * x**2
                    + 175 * y**2
                    + 252 * (x**2 + y**2) ** 3
                    - 392 * (x**2 + y**2) ** 2
                    - 20
                )
                * sin(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 49
    if m == -1 and n == 9 and wrt == "x":
        return (
            48
            * sqrt(5)
            * x
            * y
            * (
                42 * x**6
                + 126 * x**4 * y**2
                - 70 * x**4
                + 126 * x**2 * y**4
                - 140 * x**2 * y**2
                + 35 * x**2
                + 42 * y**6
                - 70 * y**4
                + 35 * y**2
                - 5
            )
        )
    if m == -1 and n == 9 and wrt == "y":
        return sqrt(5) * (
            252 * x**8
            + 3024 * x**6 * y**2
            - 560 * x**6
            + 7560 * x**4 * y**4
            - 5040 * x**4 * y**2
            + 420 * x**4
            + 7056 * x**2 * y**6
            - 8400 * x**2 * y**4
            + 2520 * x**2 * y**2
            - 120 * x**2
            + 2268 * y**8
            - 3920 * y**6
            + 2100 * y**4
            - 360 * y**2
            + 10
        )

    # Derivatives for j = 50
    if m == 1 and n == 9 and wrt == "x":
        return sqrt(5) * (
            2268 * x**8
            + 7056 * x**6 * y**2
            - 3920 * x**6
            + 7560 * x**4 * y**4
            - 8400 * x**4 * y**2
            + 2100 * x**4
            + 3024 * x**2 * y**6
            - 5040 * x**2 * y**4
            + 2520 * x**2 * y**2
            - 360 * x**2
            + 252 * y**8
            - 560 * y**6
            + 420 * y**4
            - 120 * y**2
            + 10
        )
    if m == 1 and n == 9 and wrt == "y":
        return (
            48
            * sqrt(5)
            * x
            * y
            * (
                42 * x**6
                + 126 * x**4 * y**2
                - 70 * x**4
                + 126 * x**2 * y**4
                - 140 * x**2 * y**2
                + 35 * x**2
                + 42 * y**6
                - 70 * y**4
                + 35 * y**2
                - 5
            )
        )

    # Derivatives for j = 51
    if m == 3 and n == 9 and wrt == "x":
        return (
            6
            * sqrt(5)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    175 * x**2
                    + 175 * y**2
                    + 252 * (x**2 + y**2) ** 3
                    - 392 * (x**2 + y**2) ** 2
                    - 20
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    105 * x**2
                    + 105 * y**2
                    + 84 * (x**2 + y**2) ** 3
                    - 168 * (x**2 + y**2) ** 2
                    - 20
                )
                * sin(3 * atan2(y, x))
            )
        )
    if m == 3 and n == 9 and wrt == "y":
        return (
            6
            * sqrt(5)
            * sqrt(x**2 + y**2)
            * (
                -x
                * (
                    105 * x**2
                    + 105 * y**2
                    + 84 * (x**2 + y**2) ** 3
                    - 168 * (x**2 + y**2) ** 2
                    - 20
                )
                * sin(3 * atan2(y, x))
                + y
                * (
                    175 * x**2
                    + 175 * y**2
                    + 252 * (x**2 + y**2) ** 3
                    - 392 * (x**2 + y**2) ** 2
                    - 20
                )
                * cos(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 52
    if m == 5 and n == 9 and wrt == "x":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    -392 * x**2
                    - 392 * y**2
                    + 324 * (x**2 + y**2) ** 2
                    + 105
                )
                * cos(5 * atan2(y, x))
                + 5
                * y
                * (
                    -56 * x**2
                    - 56 * y**2
                    + 36 * (x**2 + y**2) ** 2
                    + 21
                )
                * sin(5 * atan2(y, x))
            )
        )
    if m == 5 and n == 9 and wrt == "y":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (3 / 2)
            * (
                -5
                * x
                * (
                    -56 * x**2
                    - 56 * y**2
                    + 36 * (x**2 + y**2) ** 2
                    + 21
                )
                * sin(5 * atan2(y, x))
                + y
                * (
                    -392 * x**2
                    - 392 * y**2
                    + 324 * (x**2 + y**2) ** 2
                    + 105
                )
                * cos(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 53
    if m == 7 and n == 9 and wrt == "x":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x * (81 * x**2 + 81 * y**2 - 56) * cos(7 * atan2(y, x))
                + 7 * y * (9 * x**2 + 9 * y**2 - 8) * sin(7 * atan2(y, x))
            )
        )
    if m == 7 and n == 9 and wrt == "y":
        return (
            2
            * sqrt(5)
            * (x**2 + y**2) ** (5 / 2)
            * (
                -7 * x * (9 * x**2 + 9 * y**2 - 8) * sin(7 * atan2(y, x))
                + y * (81 * x**2 + 81 * y**2 - 56) * cos(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 54
    if m == 9 and n == 9 and wrt == "x":
        return (
            18
            * sqrt(5)
            * (x**2 + y**2) ** (7 / 2)
            * (x * cos(9 * atan2(y, x)) + y * sin(9 * atan2(y, x)))
        )
    if m == 9 and n == 9 and wrt == "y":
        return (
            18
            * sqrt(5)
            * (x**2 + y**2) ** (7 / 2)
            * (-x * sin(9 * atan2(y, x)) + y * cos(9 * atan2(y, x)))
        )

    # Derivatives for j = 55
    if m == -10 and n == 10 and wrt == "x":
        return (
            10
            * sqrt(22)
            * (x**2 + y**2) ** 4
            * (x * sin(10 * atan2(y, x)) - y * cos(10 * atan2(y, x)))
        )
    if m == -10 and n == 10 and wrt == "y":
        return (
            10
            * sqrt(22)
            * (x**2 + y**2) ** 4
            * (x * cos(10 * atan2(y, x)) + y * sin(10 * atan2(y, x)))
        )

    # Derivatives for j = 56
    if m == -8 and n == 10 and wrt == "x":
        return (
            8
            * sqrt(22)
            * y
            * (
                90 * x**8
                - 420 * x**6 * y**2
                - 63 * x**6
                + 315 * x**4 * y**2
                + 180 * x**2 * y**6
                - 189 * x**2 * y**4
                - 10 * y**8
                + 9 * y**6
            )
        )
    if m == -8 and n == 10 and wrt == "y":
        return (
            8
            * sqrt(22)
            * x
            * (
                10 * x**8
                - 180 * x**6 * y**2
                - 9 * x**6
                + 189 * x**4 * y**2
                + 420 * x**2 * y**6
                - 315 * x**2 * y**4
                - 90 * y**8
                + 63 * y**6
            )
        )

    # Derivatives for j = 57
    if m == -6 and n == 10 and wrt == "x":
        return (
            6
            * sqrt(22)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    -96 * x**2
                    - 96 * y**2
                    + 75 * (x**2 + y**2) ** 2
                    + 28
                )
                * sin(6 * atan2(y, x))
                - y
                * (
                    -72 * x**2
                    - 72 * y**2
                    + 45 * (x**2 + y**2) ** 2
                    + 28
                )
                * cos(6 * atan2(y, x))
            )
        )
    if m == -6 and n == 10 and wrt == "y":
        return (
            6
            * sqrt(22)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    -72 * x**2
                    - 72 * y**2
                    + 45 * (x**2 + y**2) ** 2
                    + 28
                )
                * cos(6 * atan2(y, x))
                + y
                * (
                    -96 * x**2
                    - 96 * y**2
                    + 75 * (x**2 + y**2) ** 2
                    + 28
                )
                * sin(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 58
    if m == -4 and n == 10 and wrt == "x":
        return (
            4
            * sqrt(22)
            * y
            * (
                4
                * x**2
                * (x**2 - y**2)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 300 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                    - 35
                )
                - (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    168 * x**2
                    + 168 * y**2
                    + 120 * (x**2 + y**2) ** 3
                    - 252 * (x**2 + y**2) ** 2
                    - 35
                )
            )
            / (x**2 + y**2)
        )
    if m == -4 and n == 10 and wrt == "y":
        return (
            4
            * sqrt(22)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 300 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                    - 35
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    168 * x**2
                    + 168 * y**2
                    + 120 * (x**2 + y**2) ** 3
                    - 252 * (x**2 + y**2) ** 2
                    - 35
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 59
    if m == -2 and n == 10 and wrt == "x":
        return (
            2
            * sqrt(22)
            * y
            * (
                2
                * x**2
                * (
                    -280 * x**2
                    - 280 * y**2
                    + 1050 * (x**2 + y**2) ** 4
                    - 2016 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 15
                )
                - (x**2 - y**2)
                * (
                    15 * x**2
                    + 15 * y**2
                    + 210 * (x**2 + y**2) ** 5
                    - 504 * (x**2 + y**2) ** 4
                    + 420 * (x**2 + y**2) ** 3
                    - 140 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )
    if m == -2 and n == 10 and wrt == "y":
        return (
            2
            * sqrt(22)
            * x
            * (
                2
                * y**2
                * (
                    -280 * x**2
                    - 280 * y**2
                    + 1050 * (x**2 + y**2) ** 4
                    - 2016 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 15
                )
                + (x**2 - y**2)
                * (
                    15 * x**2
                    + 15 * y**2
                    + 210 * (x**2 + y**2) ** 5
                    - 504 * (x**2 + y**2) ** 4
                    + 420 * (x**2 + y**2) ** 3
                    - 140 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 60
    if m == 0 and n == 10 and wrt == "x":
        return (
            60
            * sqrt(11)
            * x
            * (
                -14 * x**2
                - 14 * y**2
                + 42 * (x**2 + y**2) ** 4
                - 84 * (x**2 + y**2) ** 3
                + 56 * (x**2 + y**2) ** 2
                + 1
            )
        )
    if m == 0 and n == 10 and wrt == "y":
        return (
            60
            * sqrt(11)
            * y
            * (
                -14 * x**2
                - 14 * y**2
                + 42 * (x**2 + y**2) ** 4
                - 84 * (x**2 + y**2) ** 3
                + 56 * (x**2 + y**2) ** 2
                + 1
            )
        )

    # Derivatives for j = 61
    if m == 2 and n == 10 and wrt == "x":
        return (
            2
            * sqrt(22)
            * x
            * (
                2
                * y**2
                * (
                    15 * x**2
                    + 15 * y**2
                    + 210 * (x**2 + y**2) ** 5
                    - 504 * (x**2 + y**2) ** 4
                    + 420 * (x**2 + y**2) ** 3
                    - 140 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    -280 * x**2
                    - 280 * y**2
                    + 1050 * (x**2 + y**2) ** 4
                    - 2016 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 15
                )
            )
            / (x**2 + y**2)
        )
    if m == 2 and n == 10 and wrt == "y":
        return (
            2
            * sqrt(22)
            * y
            * (
                -2
                * x**2
                * (
                    15 * x**2
                    + 15 * y**2
                    + 210 * (x**2 + y**2) ** 5
                    - 504 * (x**2 + y**2) ** 4
                    + 420 * (x**2 + y**2) ** 3
                    - 140 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    -280 * x**2
                    - 280 * y**2
                    + 1050 * (x**2 + y**2) ** 4
                    - 2016 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 15
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 62
    if m == 4 and n == 10 and wrt == "x":
        return (
            4
            * sqrt(22)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    168 * x**2
                    + 168 * y**2
                    + 120 * (x**2 + y**2) ** 3
                    - 252 * (x**2 + y**2) ** 2
                    - 35
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 300 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                    - 35
                )
            )
            / (x**2 + y**2)
        )
    if m == 4 and n == 10 and wrt == "y":
        return (
            4
            * sqrt(22)
            * y
            * (
                -4
                * x**2
                * (x**2 - y**2)
                * (
                    168 * x**2
                    + 168 * y**2
                    + 120 * (x**2 + y**2) ** 3
                    - 252 * (x**2 + y**2) ** 2
                    - 35
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 300 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                    - 35
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 63
    if m == 6 and n == 10 and wrt == "x":
        return (
            6
            * sqrt(22)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    -96 * x**2
                    - 96 * y**2
                    + 75 * (x**2 + y**2) ** 2
                    + 28
                )
                * cos(6 * atan2(y, x))
                + y
                * (
                    -72 * x**2
                    - 72 * y**2
                    + 45 * (x**2 + y**2) ** 2
                    + 28
                )
                * sin(6 * atan2(y, x))
            )
        )
    if m == 6 and n == 10 and wrt == "y":
        return (
            6
            * sqrt(22)
            * (x**2 + y**2) ** 2
            * (
                -x
                * (
                    -72 * x**2
                    - 72 * y**2
                    + 45 * (x**2 + y**2) ** 2
                    + 28
                )
                * sin(6 * atan2(y, x))
                + y
                * (
                    -96 * x**2
                    - 96 * y**2
                    + 75 * (x**2 + y**2) ** 2
                    + 28
                )
                * cos(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 64
    if m == 8 and n == 10 and wrt == "x":
        return (
            4
            * sqrt(22)
            * x
            * (
                25 * x**8
                - 540 * x**6 * y**2
                - 18 * x**6
                + 630 * x**4 * y**4
                + 378 * x**4 * y**2
                + 420 * x**2 * y**6
                - 630 * x**2 * y**4
                - 135 * y**8
                + 126 * y**6
            )
        )
    if m == 8 and n == 10 and wrt == "y":
        return (
            4
            * sqrt(22)
            * y
            * (
                -135 * x**8
                + 420 * x**6 * y**2
                + 126 * x**6
                + 630 * x**4 * y**4
                - 630 * x**4 * y**2
                - 540 * x**2 * y**6
                + 378 * x**2 * y**4
                + 25 * y**8
                - 18 * y**6
            )
        )

    # Derivatives for j = 65
    if m == 10 and n == 10 and wrt == "x":
        return (
            10
            * sqrt(22)
            * (x**2 + y**2) ** 4
            * (x * cos(10 * atan2(y, x)) + y * sin(10 * atan2(y, x)))
        )
    if m == 10 and n == 10 and wrt == "y":
        return (
            10
            * sqrt(22)
            * (x**2 + y**2) ** 4
            * (-x * sin(10 * atan2(y, x)) + y * cos(10 * atan2(y, x)))
        )

    # Derivatives for j = 66
    if m == -11 and n == 11 and wrt == "x":
        return (
            22
            * sqrt(6)
            * (x**2 + y**2) ** (9 / 2)
            * (x * sin(11 * atan2(y, x)) - y * cos(11 * atan2(y, x)))
        )
    if m == -11 and n == 11 and wrt == "y":
        return (
            22
            * sqrt(6)
            * (x**2 + y**2) ** (9 / 2)
            * (x * cos(11 * atan2(y, x)) + y * sin(11 * atan2(y, x)))
        )

    # Derivatives for j = 67
    if m == -9 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x * (121 * x**2 + 121 * y**2 - 90) * sin(9 * atan2(y, x))
                - 9
                * y
                * (11 * x**2 + 11 * y**2 - 10)
                * cos(9 * atan2(y, x))
            )
        )
    if m == -9 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (7 / 2)
            * (
                9 * x * (11 * x**2 + 11 * y**2 - 10) * cos(9 * atan2(y, x))
                + y * (121 * x**2 + 121 * y**2 - 90) * sin(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 68
    if m == -7 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    -810 * x**2
                    - 810 * y**2
                    + 605 * (x**2 + y**2) ** 2
                    + 252
                )
                * sin(7 * atan2(y, x))
                - 7
                * y
                * (
                    -90 * x**2
                    - 90 * y**2
                    + 55 * (x**2 + y**2) ** 2
                    + 36
                )
                * cos(7 * atan2(y, x))
            )
        )
    if m == -7 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (5 / 2)
            * (
                7
                * x
                * (
                    -90 * x**2
                    - 90 * y**2
                    + 55 * (x**2 + y**2) ** 2
                    + 36
                )
                * cos(7 * atan2(y, x))
                + y
                * (
                    -810 * x**2
                    - 810 * y**2
                    + 605 * (x**2 + y**2) ** 2
                    + 252
                )
                * sin(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 69
    if m == -5 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    1764 * x**2
                    + 1764 * y**2
                    + 1815 * (x**2 + y**2) ** 3
                    - 3240 * (x**2 + y**2) ** 2
                    - 280
                )
                * sin(5 * atan2(y, x))
                - 5
                * y
                * (
                    252 * x**2
                    + 252 * y**2
                    + 165 * (x**2 + y**2) ** 3
                    - 360 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(5 * atan2(y, x))
            )
        )
    if m == -5 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (3 / 2)
            * (
                5
                * x
                * (
                    252 * x**2
                    + 252 * y**2
                    + 165 * (x**2 + y**2) ** 3
                    - 360 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    1764 * x**2
                    + 1764 * y**2
                    + 1815 * (x**2 + y**2) ** 3
                    - 3240 * (x**2 + y**2) ** 2
                    - 280
                )
                * sin(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 70
    if m == -3 and n == 11 and wrt == "x":
        return sqrt(6 * x**2 + 6 * y**2) * (
            2
            * x
            * (
                -1400 * x**2
                - 1400 * y**2
                + 3630 * (x**2 + y**2) ** 4
                - 7560 * (x**2 + y**2) ** 3
                + 5292 * (x**2 + y**2) ** 2
                + 105
            )
            * sin(3 * atan2(y, x))
            - 6
            * y
            * (
                -280 * x**2
                - 280 * y**2
                + 330 * (x**2 + y**2) ** 4
                - 840 * (x**2 + y**2) ** 3
                + 756 * (x**2 + y**2) ** 2
                + 35
            )
            * cos(3 * atan2(y, x))
        )
    if m == -3 and n == 11 and wrt == "y":
        return sqrt(6 * x**2 + 6 * y**2) * (
            6
            * x
            * (
                -280 * x**2
                - 280 * y**2
                + 330 * (x**2 + y**2) ** 4
                - 840 * (x**2 + y**2) ** 3
                + 756 * (x**2 + y**2) ** 2
                + 35
            )
            * cos(3 * atan2(y, x))
            + 2
            * y
            * (
                -1400 * x**2
                - 1400 * y**2
                + 3630 * (x**2 + y**2) ** 4
                - 7560 * (x**2 + y**2) ** 3
                + 5292 * (x**2 + y**2) ** 2
                + 105
            )
            * sin(3 * atan2(y, x))
        )

    # Derivatives for j = 71
    if m == -1 and n == 11 and wrt == "x":
        return (
            140
            * sqrt(6)
            * x
            * y
            * (
                66 * x**8
                + 264 * x**6 * y**2
                - 144 * x**6
                + 396 * x**4 * y**4
                - 432 * x**4 * y**2
                + 108 * x**4
                + 264 * x**2 * y**6
                - 432 * x**2 * y**4
                + 216 * x**2 * y**2
                - 32 * x**2
                + 66 * y**8
                - 144 * y**6
                + 108 * y**4
                - 32 * y**2
                + 3
            )
        )
    if m == -1 and n == 11 and wrt == "y":
        return sqrt(6) * (
            924 * x**10
            + 13860 * x**8 * y**2
            - 2520 * x**8
            + 46200 * x**6 * y**4
            - 30240 * x**6 * y**2
            + 2520 * x**6
            + 64680 * x**4 * y**6
            - 75600 * x**4 * y**4
            + 22680 * x**4 * y**2
            - 1120 * x**4
            + 41580 * x**2 * y**8
            - 70560 * x**2 * y**6
            + 37800 * x**2 * y**4
            - 6720 * x**2 * y**2
            + 210 * x**2
            + 10164 * y**10
            - 22680 * y**8
            + 17640 * y**6
            - 5600 * y**4
            + 630 * y**2
            - 12
        )

    # Derivatives for j = 72
    if m == 1 and n == 11 and wrt == "x":
        return sqrt(6) * (
            10164 * x**10
            + 41580 * x**8 * y**2
            - 22680 * x**8
            + 64680 * x**6 * y**4
            - 70560 * x**6 * y**2
            + 17640 * x**6
            + 46200 * x**4 * y**6
            - 75600 * x**4 * y**4
            + 37800 * x**4 * y**2
            - 5600 * x**4
            + 13860 * x**2 * y**8
            - 30240 * x**2 * y**6
            + 22680 * x**2 * y**4
            - 6720 * x**2 * y**2
            + 630 * x**2
            + 924 * y**10
            - 2520 * y**8
            + 2520 * y**6
            - 1120 * y**4
            + 210 * y**2
            - 12
        )
    if m == 1 and n == 11 and wrt == "y":
        return (
            140
            * sqrt(6)
            * x
            * y
            * (
                66 * x**8
                + 264 * x**6 * y**2
                - 144 * x**6
                + 396 * x**4 * y**4
                - 432 * x**4 * y**2
                + 108 * x**4
                + 264 * x**2 * y**6
                - 432 * x**2 * y**4
                + 216 * x**2 * y**2
                - 32 * x**2
                + 66 * y**8
                - 144 * y**6
                + 108 * y**4
                - 32 * y**2
                + 3
            )
        )

    # Derivatives for j = 73
    if m == 3 and n == 11 and wrt == "x":
        return sqrt(6 * x**2 + 6 * y**2) * (
            2
            * x
            * (
                -1400 * x**2
                - 1400 * y**2
                + 3630 * (x**2 + y**2) ** 4
                - 7560 * (x**2 + y**2) ** 3
                + 5292 * (x**2 + y**2) ** 2
                + 105
            )
            * cos(3 * atan2(y, x))
            + 6
            * y
            * (
                -280 * x**2
                - 280 * y**2
                + 330 * (x**2 + y**2) ** 4
                - 840 * (x**2 + y**2) ** 3
                + 756 * (x**2 + y**2) ** 2
                + 35
            )
            * sin(3 * atan2(y, x))
        )
    if m == 3 and n == 11 and wrt == "y":
        return sqrt(6 * x**2 + 6 * y**2) * (
            -6
            * x
            * (
                -280 * x**2
                - 280 * y**2
                + 330 * (x**2 + y**2) ** 4
                - 840 * (x**2 + y**2) ** 3
                + 756 * (x**2 + y**2) ** 2
                + 35
            )
            * sin(3 * atan2(y, x))
            + 2
            * y
            * (
                -1400 * x**2
                - 1400 * y**2
                + 3630 * (x**2 + y**2) ** 4
                - 7560 * (x**2 + y**2) ** 3
                + 5292 * (x**2 + y**2) ** 2
                + 105
            )
            * cos(3 * atan2(y, x))
        )

    # Derivatives for j = 74
    if m == 5 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    1764 * x**2
                    + 1764 * y**2
                    + 1815 * (x**2 + y**2) ** 3
                    - 3240 * (x**2 + y**2) ** 2
                    - 280
                )
                * cos(5 * atan2(y, x))
                + 5
                * y
                * (
                    252 * x**2
                    + 252 * y**2
                    + 165 * (x**2 + y**2) ** 3
                    - 360 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(5 * atan2(y, x))
            )
        )
    if m == 5 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (3 / 2)
            * (
                -5
                * x
                * (
                    252 * x**2
                    + 252 * y**2
                    + 165 * (x**2 + y**2) ** 3
                    - 360 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(5 * atan2(y, x))
                + y
                * (
                    1764 * x**2
                    + 1764 * y**2
                    + 1815 * (x**2 + y**2) ** 3
                    - 3240 * (x**2 + y**2) ** 2
                    - 280
                )
                * cos(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 75
    if m == 7 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    -810 * x**2
                    - 810 * y**2
                    + 605 * (x**2 + y**2) ** 2
                    + 252
                )
                * cos(7 * atan2(y, x))
                + 7
                * y
                * (
                    -90 * x**2
                    - 90 * y**2
                    + 55 * (x**2 + y**2) ** 2
                    + 36
                )
                * sin(7 * atan2(y, x))
            )
        )
    if m == 7 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (5 / 2)
            * (
                -7
                * x
                * (
                    -90 * x**2
                    - 90 * y**2
                    + 55 * (x**2 + y**2) ** 2
                    + 36
                )
                * sin(7 * atan2(y, x))
                + y
                * (
                    -810 * x**2
                    - 810 * y**2
                    + 605 * (x**2 + y**2) ** 2
                    + 252
                )
                * cos(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 76
    if m == 9 and n == 11 and wrt == "x":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x * (121 * x**2 + 121 * y**2 - 90) * cos(9 * atan2(y, x))
                + 9
                * y
                * (11 * x**2 + 11 * y**2 - 10)
                * sin(9 * atan2(y, x))
            )
        )
    if m == 9 and n == 11 and wrt == "y":
        return (
            2
            * sqrt(6)
            * (x**2 + y**2) ** (7 / 2)
            * (
                -9
                * x
                * (11 * x**2 + 11 * y**2 - 10)
                * sin(9 * atan2(y, x))
                + y * (121 * x**2 + 121 * y**2 - 90) * cos(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 77
    if m == 11 and n == 11 and wrt == "x":
        return (
            22
            * sqrt(6)
            * (x**2 + y**2) ** (9 / 2)
            * (x * cos(11 * atan2(y, x)) + y * sin(11 * atan2(y, x)))
        )
    if m == 11 and n == 11 and wrt == "y":
        return (
            22
            * sqrt(6)
            * (x**2 + y**2) ** (9 / 2)
            * (-x * sin(11 * atan2(y, x)) + y * cos(11 * atan2(y, x)))
        )

    # Derivatives for j = 78
    if m == -12 and n == 12 and wrt == "x":
        return (
            12
            * sqrt(26)
            * (x**2 + y**2) ** 5
            * (x * sin(12 * atan2(y, x)) - y * cos(12 * atan2(y, x)))
        )
    if m == -12 and n == 12 and wrt == "y":
        return (
            12
            * sqrt(26)
            * (x**2 + y**2) ** 5
            * (x * cos(12 * atan2(y, x)) + y * sin(12 * atan2(y, x)))
        )

    # Derivatives for j = 79
    if m == -10 and n == 12 and wrt == "x":
        return (
            2
            * sqrt(26)
            * (x**2 + y**2) ** 4
            * (
                x * (72 * x**2 + 72 * y**2 - 55) * sin(10 * atan2(y, x))
                - 5
                * y
                * (12 * x**2 + 12 * y**2 - 11)
                * cos(10 * atan2(y, x))
            )
        )
    if m == -10 and n == 12 and wrt == "y":
        return (
            2
            * sqrt(26)
            * (x**2 + y**2) ** 4
            * (
                5
                * x
                * (12 * x**2 + 12 * y**2 - 11)
                * cos(10 * atan2(y, x))
                + y * (72 * x**2 + 72 * y**2 - 55) * sin(10 * atan2(y, x))
            )
        )

    # Derivatives for j = 80
    if m == -8 and n == 12 and wrt == "x":
        return (
            8
            * sqrt(26)
            * y
            * (
                4
                * x**2
                * (
                    -275 * x**2
                    - 275 * y**2
                    + 198 * (x**2 + y**2) ** 2
                    + 90
                )
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                - (
                    -110 * x**2
                    - 110 * y**2
                    + 66 * (x**2 + y**2) ** 2
                    + 45
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )
    if m == -8 and n == 12 and wrt == "y":
        return (
            8
            * sqrt(26)
            * x
            * (
                4
                * y**2
                * (
                    -275 * x**2
                    - 275 * y**2
                    + 198 * (x**2 + y**2) ** 2
                    + 90
                )
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                + (
                    -110 * x**2
                    - 110 * y**2
                    + 66 * (x**2 + y**2) ** 2
                    + 45
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 81
    if m == -6 and n == 12 and wrt == "x":
        return (
            6
            * sqrt(26)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    480 * x**2
                    + 480 * y**2
                    + 440 * (x**2 + y**2) ** 3
                    - 825 * (x**2 + y**2) ** 2
                    - 84
                )
                * sin(6 * atan2(y, x))
                - y
                * (
                    360 * x**2
                    + 360 * y**2
                    + 220 * (x**2 + y**2) ** 3
                    - 495 * (x**2 + y**2) ** 2
                    - 84
                )
                * cos(6 * atan2(y, x))
            )
        )
    if m == -6 and n == 12 and wrt == "y":
        return (
            6
            * sqrt(26)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    360 * x**2
                    + 360 * y**2
                    + 220 * (x**2 + y**2) ** 3
                    - 495 * (x**2 + y**2) ** 2
                    - 84
                )
                * cos(6 * atan2(y, x))
                + y
                * (
                    480 * x**2
                    + 480 * y**2
                    + 440 * (x**2 + y**2) ** 3
                    - 825 * (x**2 + y**2) ** 2
                    - 84
                )
                * sin(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 82
    if m == -4 and n == 12 and wrt == "x":
        return (
            4
            * sqrt(26)
            * y
            * (
                4
                * x**2
                * (x**2 - y**2)
                * (
                    -756 * x**2
                    - 756 * y**2
                    + 1485 * (x**2 + y**2) ** 4
                    - 3300 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 70
                )
                - (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -504 * x**2
                    - 504 * y**2
                    + 495 * (x**2 + y**2) ** 4
                    - 1320 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 70
                )
            )
            / (x**2 + y**2)
        )
    if m == -4 and n == 12 and wrt == "y":
        return (
            4
            * sqrt(26)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    -756 * x**2
                    - 756 * y**2
                    + 1485 * (x**2 + y**2) ** 4
                    - 3300 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 70
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -504 * x**2
                    - 504 * y**2
                    + 495 * (x**2 + y**2) ** 4
                    - 1320 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 70
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 83
    if m == -2 and n == 12 and wrt == "x":
        return (
            2
            * sqrt(26)
            * y
            * (
                2
                * x**2
                * (
                    560 * x**2
                    + 560 * y**2
                    + 4752 * (x**2 + y**2) ** 5
                    - 11550 * (x**2 + y**2) ** 4
                    + 10080 * (x**2 + y**2) ** 3
                    - 3780 * (x**2 + y**2) ** 2
                    - 21
                )
                + (x**2 - y**2)
                * (
                    21 * x**2
                    + 21 * y**2
                    - 792 * (x**2 + y**2) ** 6
                    + 2310 * (x**2 + y**2) ** 5
                    - 2520 * (x**2 + y**2) ** 4
                    + 1260 * (x**2 + y**2) ** 3
                    - 280 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )
    if m == -2 and n == 12 and wrt == "y":
        return (
            2
            * sqrt(26)
            * x
            * (
                2
                * y**2
                * (
                    560 * x**2
                    + 560 * y**2
                    + 4752 * (x**2 + y**2) ** 5
                    - 11550 * (x**2 + y**2) ** 4
                    + 10080 * (x**2 + y**2) ** 3
                    - 3780 * (x**2 + y**2) ** 2
                    - 21
                )
                - (x**2 - y**2)
                * (
                    21 * x**2
                    + 21 * y**2
                    - 792 * (x**2 + y**2) ** 6
                    + 2310 * (x**2 + y**2) ** 5
                    - 2520 * (x**2 + y**2) ** 4
                    + 1260 * (x**2 + y**2) ** 3
                    - 280 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 84
    if m == 0 and n == 12 and wrt == "x":
        return (
            84
            * sqrt(13)
            * x
            * (
                20 * x**2
                + 20 * y**2
                + 132 * (x**2 + y**2) ** 5
                - 330 * (x**2 + y**2) ** 4
                + 300 * (x**2 + y**2) ** 3
                - 120 * (x**2 + y**2) ** 2
                - 1
            )
        )
    if m == 0 and n == 12 and wrt == "y":
        return (
            84
            * sqrt(13)
            * y
            * (
                20 * x**2
                + 20 * y**2
                + 132 * (x**2 + y**2) ** 5
                - 330 * (x**2 + y**2) ** 4
                + 300 * (x**2 + y**2) ** 3
                - 120 * (x**2 + y**2) ** 2
                - 1
            )
        )

    # Derivatives for j = 85
    if m == 2 and n == 12 and wrt == "x":
        return (
            2
            * sqrt(26)
            * x
            * (
                -2
                * y**2
                * (
                    21 * x**2
                    + 21 * y**2
                    - 792 * (x**2 + y**2) ** 6
                    + 2310 * (x**2 + y**2) ** 5
                    - 2520 * (x**2 + y**2) ** 4
                    + 1260 * (x**2 + y**2) ** 3
                    - 280 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    560 * x**2
                    + 560 * y**2
                    + 4752 * (x**2 + y**2) ** 5
                    - 11550 * (x**2 + y**2) ** 4
                    + 10080 * (x**2 + y**2) ** 3
                    - 3780 * (x**2 + y**2) ** 2
                    - 21
                )
            )
            / (x**2 + y**2)
        )
    if m == 2 and n == 12 and wrt == "y":
        return (
            2
            * sqrt(26)
            * y
            * (
                2
                * x**2
                * (
                    21 * x**2
                    + 21 * y**2
                    - 792 * (x**2 + y**2) ** 6
                    + 2310 * (x**2 + y**2) ** 5
                    - 2520 * (x**2 + y**2) ** 4
                    + 1260 * (x**2 + y**2) ** 3
                    - 280 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    560 * x**2
                    + 560 * y**2
                    + 4752 * (x**2 + y**2) ** 5
                    - 11550 * (x**2 + y**2) ** 4
                    + 10080 * (x**2 + y**2) ** 3
                    - 3780 * (x**2 + y**2) ** 2
                    - 21
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 86
    if m == 4 and n == 12 and wrt == "x":
        return (
            4
            * sqrt(26)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    -504 * x**2
                    - 504 * y**2
                    + 495 * (x**2 + y**2) ** 4
                    - 1320 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 70
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -756 * x**2
                    - 756 * y**2
                    + 1485 * (x**2 + y**2) ** 4
                    - 3300 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 70
                )
            )
            / (x**2 + y**2)
        )
    if m == 4 and n == 12 and wrt == "y":
        return (
            4
            * sqrt(26)
            * y
            * (
                -4
                * x**2
                * (x**2 - y**2)
                * (
                    -504 * x**2
                    - 504 * y**2
                    + 495 * (x**2 + y**2) ** 4
                    - 1320 * (x**2 + y**2) ** 3
                    + 1260 * (x**2 + y**2) ** 2
                    + 70
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    -756 * x**2
                    - 756 * y**2
                    + 1485 * (x**2 + y**2) ** 4
                    - 3300 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 70
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 87
    if m == 6 and n == 12 and wrt == "x":
        return (
            6
            * sqrt(26)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    480 * x**2
                    + 480 * y**2
                    + 440 * (x**2 + y**2) ** 3
                    - 825 * (x**2 + y**2) ** 2
                    - 84
                )
                * cos(6 * atan2(y, x))
                + y
                * (
                    360 * x**2
                    + 360 * y**2
                    + 220 * (x**2 + y**2) ** 3
                    - 495 * (x**2 + y**2) ** 2
                    - 84
                )
                * sin(6 * atan2(y, x))
            )
        )
    if m == 6 and n == 12 and wrt == "y":
        return (
            6
            * sqrt(26)
            * (x**2 + y**2) ** 2
            * (
                -x
                * (
                    360 * x**2
                    + 360 * y**2
                    + 220 * (x**2 + y**2) ** 3
                    - 495 * (x**2 + y**2) ** 2
                    - 84
                )
                * sin(6 * atan2(y, x))
                + y
                * (
                    480 * x**2
                    + 480 * y**2
                    + 440 * (x**2 + y**2) ** 3
                    - 825 * (x**2 + y**2) ** 2
                    - 84
                )
                * cos(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 88
    if m == 8 and n == 12 and wrt == "x":
        return (
            4
            * sqrt(26)
            * x
            * (
                16
                * y**2
                * (
                    -110 * x**2
                    - 110 * y**2
                    + 66 * (x**2 + y**2) ** 2
                    + 45
                )
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                + (
                    -275 * x**2
                    - 275 * y**2
                    + 198 * (x**2 + y**2) ** 2
                    + 90
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )
    if m == 8 and n == 12 and wrt == "y":
        return (
            4
            * sqrt(26)
            * y
            * (
                -16
                * x**2
                * (
                    -110 * x**2
                    - 110 * y**2
                    + 66 * (x**2 + y**2) ** 2
                    + 45
                )
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                + (
                    -275 * x**2
                    - 275 * y**2
                    + 198 * (x**2 + y**2) ** 2
                    + 90
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 89
    if m == 10 and n == 12 and wrt == "x":
        return (
            2
            * sqrt(26)
            * (x**2 + y**2) ** 4
            * (
                x * (72 * x**2 + 72 * y**2 - 55) * cos(10 * atan2(y, x))
                + 5
                * y
                * (12 * x**2 + 12 * y**2 - 11)
                * sin(10 * atan2(y, x))
            )
        )
    if m == 10 and n == 12 and wrt == "y":
        return (
            2
            * sqrt(26)
            * (x**2 + y**2) ** 4
            * (
                -5
                * x
                * (12 * x**2 + 12 * y**2 - 11)
                * sin(10 * atan2(y, x))
                + y * (72 * x**2 + 72 * y**2 - 55) * cos(10 * atan2(y, x))
            )
        )

    # Derivatives for j = 90
    if m == 12 and n == 12 and wrt == "x":
        return (
            12
            * sqrt(26)
            * (x**2 + y**2) ** 5
            * (x * cos(12 * atan2(y, x)) + y * sin(12 * atan2(y, x)))
        )
    if m == 12 and n == 12 and wrt == "y":
        return (
            12
            * sqrt(26)
            * (x**2 + y**2) ** 5
            * (-x * sin(12 * atan2(y, x)) + y * cos(12 * atan2(y, x)))
        )

    # Derivatives for j = 91
    if m == -13 and n == 13 and wrt == "x":
        return (
            26
            * sqrt(7)
            * (x**2 + y**2) ** (11 / 2)
            * (x * sin(13 * atan2(y, x)) - y * cos(13 * atan2(y, x)))
        )
    if m == -13 and n == 13 and wrt == "y":
        return (
            26
            * sqrt(7)
            * (x**2 + y**2) ** (11 / 2)
            * (x * cos(13 * atan2(y, x)) + y * sin(13 * atan2(y, x)))
        )

    # Derivatives for j = 92
    if m == -11 and n == 13 and wrt == "x":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (9 / 2)
            * (
                x * (169 * x**2 + 169 * y**2 - 132) * sin(11 * atan2(y, x))
                - 11
                * y
                * (13 * x**2 + 13 * y**2 - 12)
                * cos(11 * atan2(y, x))
            )
        )
    if m == -11 and n == 13 and wrt == "y":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (9 / 2)
            * (
                11
                * x
                * (13 * x**2 + 13 * y**2 - 12)
                * cos(11 * atan2(y, x))
                + y
                * (169 * x**2 + 169 * y**2 - 132)
                * sin(11 * atan2(y, x))
            )
        )

    # Derivatives for j = 93
    if m == -9 and n == 13 and wrt == "x":
        return (
            6
            * sqrt(7)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x
                * (
                    -484 * x**2
                    - 484 * y**2
                    + 338 * (x**2 + y**2) ** 2
                    + 165
                )
                * sin(9 * atan2(y, x))
                - 3
                * y
                * (
                    -132 * x**2
                    - 132 * y**2
                    + 78 * (x**2 + y**2) ** 2
                    + 55
                )
                * cos(9 * atan2(y, x))
            )
        )
    if m == -9 and n == 13 and wrt == "y":
        return (
            6
            * sqrt(7)
            * (x**2 + y**2) ** (7 / 2)
            * (
                3
                * x
                * (
                    -132 * x**2
                    - 132 * y**2
                    + 78 * (x**2 + y**2) ** 2
                    + 55
                )
                * cos(9 * atan2(y, x))
                + y
                * (
                    -484 * x**2
                    - 484 * y**2
                    + 338 * (x**2 + y**2) ** 2
                    + 165
                )
                * sin(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 94
    if m == -7 and n == 13 and wrt == "x":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    4455 * x**2
                    + 4455 * y**2
                    + 3718 * (x**2 + y**2) ** 3
                    - 7260 * (x**2 + y**2) ** 2
                    - 840
                )
                * sin(7 * atan2(y, x))
                - 7
                * y
                * (
                    495 * x**2
                    + 495 * y**2
                    + 286 * (x**2 + y**2) ** 3
                    - 660 * (x**2 + y**2) ** 2
                    - 120
                )
                * cos(7 * atan2(y, x))
            )
        )
    if m == -7 and n == 13 and wrt == "y":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (5 / 2)
            * (
                7
                * x
                * (
                    495 * x**2
                    + 495 * y**2
                    + 286 * (x**2 + y**2) ** 3
                    - 660 * (x**2 + y**2) ** 2
                    - 120
                )
                * cos(7 * atan2(y, x))
                + y
                * (
                    4455 * x**2
                    + 4455 * y**2
                    + 3718 * (x**2 + y**2) ** 3
                    - 7260 * (x**2 + y**2) ** 2
                    - 840
                )
                * sin(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 95
    if m == -5 and n == 13 and wrt == "x":
        return (
            10
            * sqrt(7)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    -1176 * x**2
                    - 1176 * y**2
                    + 1859 * (x**2 + y**2) ** 4
                    - 4356 * (x**2 + y**2) ** 3
                    + 3564 * (x**2 + y**2) ** 2
                    + 126
                )
                * sin(5 * atan2(y, x))
                - y
                * (
                    -840 * x**2
                    - 840 * y**2
                    + 715 * (x**2 + y**2) ** 4
                    - 1980 * (x**2 + y**2) ** 3
                    + 1980 * (x**2 + y**2) ** 2
                    + 126
                )
                * cos(5 * atan2(y, x))
            )
        )
    if m == -5 and n == 13 and wrt == "y":
        return (
            10
            * sqrt(7)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    -840 * x**2
                    - 840 * y**2
                    + 715 * (x**2 + y**2) ** 4
                    - 1980 * (x**2 + y**2) ** 3
                    + 1980 * (x**2 + y**2) ** 2
                    + 126
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    -1176 * x**2
                    - 1176 * y**2
                    + 1859 * (x**2 + y**2) ** 4
                    - 4356 * (x**2 + y**2) ** 3
                    + 3564 * (x**2 + y**2) ** 2
                    + 126
                )
                * sin(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 96
    if m == -3 and n == 13 and wrt == "x":
        return (
            6
            * sqrt(7)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 5577 * (x**2 + y**2) ** 5
                    - 14520 * (x**2 + y**2) ** 4
                    + 13860 * (x**2 + y**2) ** 3
                    - 5880 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(3 * atan2(y, x))
                - y
                * (
                    630 * x**2
                    + 630 * y**2
                    + 1287 * (x**2 + y**2) ** 5
                    - 3960 * (x**2 + y**2) ** 4
                    + 4620 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(3 * atan2(y, x))
            )
        )
    if m == -3 and n == 13 and wrt == "y":
        return (
            6
            * sqrt(7)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    630 * x**2
                    + 630 * y**2
                    + 1287 * (x**2 + y**2) ** 5
                    - 3960 * (x**2 + y**2) ** 4
                    + 4620 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 5577 * (x**2 + y**2) ** 5
                    - 14520 * (x**2 + y**2) ** 4
                    + 13860 * (x**2 + y**2) ** 3
                    - 5880 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 97
    if m == -1 and n == 13 and wrt == "x":
        return (
            12
            * sqrt(7)
            * x
            * y
            * (
                28 * x**2
                + 28 * y**2
                - 286 * (x**2 + y**2) ** 6
                + 924 * (x**2 + y**2) ** 5
                - 1155 * (x**2 + y**2) ** 4
                + 700 * (x**2 + y**2) ** 3
                - 210 * (x**2 + y**2) ** 2
                + (x**2 + y**2)
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 3718 * (x**2 + y**2) ** 5
                    - 10164 * (x**2 + y**2) ** 4
                    + 10395 * (x**2 + y**2) ** 3
                    - 4900 * (x**2 + y**2) ** 2
                    - 84
                )
            )
            / (x**2 + y**2)
        )
    if m == -1 and n == 13 and wrt == "y":
        return sqrt(7) * (
            3432 * x**12
            + 61776 * x**10 * y**2
            - 11088 * x**10
            + 257400 * x**8 * y**4
            - 166320 * x**8 * y**2
            + 13860 * x**8
            + 480480 * x**6 * y**6
            - 554400 * x**6 * y**4
            + 166320 * x**6 * y**2
            - 8400 * x**6
            + 463320 * x**4 * y**8
            - 776160 * x**4 * y**6
            + 415800 * x**4 * y**4
            - 75600 * x**4 * y**2
            + 2520 * x**4
            + 226512 * x**2 * y**10
            - 498960 * x**2 * y**8
            + 388080 * x**2 * y**6
            - 126000 * x**2 * y**4
            + 15120 * x**2 * y**2
            - 336 * x**2
            + 44616 * y**12
            - 121968 * y**10
            + 124740 * y**8
            - 58800 * y**6
            + 12600 * y**4
            - 1008 * y**2
            + 14
        )

    # Derivatives for j = 98
    if m == 1 and n == 13 and wrt == "x":
        return sqrt(7) * (
            44616 * x**12
            + 226512 * x**10 * y**2
            - 121968 * x**10
            + 463320 * x**8 * y**4
            - 498960 * x**8 * y**2
            + 124740 * x**8
            + 480480 * x**6 * y**6
            - 776160 * x**6 * y**4
            + 388080 * x**6 * y**2
            - 58800 * x**6
            + 257400 * x**4 * y**8
            - 554400 * x**4 * y**6
            + 415800 * x**4 * y**4
            - 126000 * x**4 * y**2
            + 12600 * x**4
            + 61776 * x**2 * y**10
            - 166320 * x**2 * y**8
            + 166320 * x**2 * y**6
            - 75600 * x**2 * y**4
            + 15120 * x**2 * y**2
            - 1008 * x**2
            + 3432 * y**12
            - 11088 * y**10
            + 13860 * y**8
            - 8400 * y**6
            + 2520 * y**4
            - 336 * y**2
            + 14
        )
    if m == 1 and n == 13 and wrt == "y":
        return (
            12
            * sqrt(7)
            * x
            * y
            * (
                28 * x**2
                + 28 * y**2
                - 286 * (x**2 + y**2) ** 6
                + 924 * (x**2 + y**2) ** 5
                - 1155 * (x**2 + y**2) ** 4
                + 700 * (x**2 + y**2) ** 3
                - 210 * (x**2 + y**2) ** 2
                + (x**2 + y**2)
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 3718 * (x**2 + y**2) ** 5
                    - 10164 * (x**2 + y**2) ** 4
                    + 10395 * (x**2 + y**2) ** 3
                    - 4900 * (x**2 + y**2) ** 2
                    - 84
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 99
    if m == 3 and n == 13 and wrt == "x":
        return (
            6
            * sqrt(7)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 5577 * (x**2 + y**2) ** 5
                    - 14520 * (x**2 + y**2) ** 4
                    + 13860 * (x**2 + y**2) ** 3
                    - 5880 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    630 * x**2
                    + 630 * y**2
                    + 1287 * (x**2 + y**2) ** 5
                    - 3960 * (x**2 + y**2) ** 4
                    + 4620 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(3 * atan2(y, x))
            )
        )
    if m == 3 and n == 13 and wrt == "y":
        return (
            6
            * sqrt(7)
            * sqrt(x**2 + y**2)
            * (
                -x
                * (
                    630 * x**2
                    + 630 * y**2
                    + 1287 * (x**2 + y**2) ** 5
                    - 3960 * (x**2 + y**2) ** 4
                    + 4620 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 56
                )
                * sin(3 * atan2(y, x))
                + y
                * (
                    1050 * x**2
                    + 1050 * y**2
                    + 5577 * (x**2 + y**2) ** 5
                    - 14520 * (x**2 + y**2) ** 4
                    + 13860 * (x**2 + y**2) ** 3
                    - 5880 * (x**2 + y**2) ** 2
                    - 56
                )
                * cos(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 100
    if m == 5 and n == 13 and wrt == "x":
        return (
            10
            * sqrt(7)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    -1176 * x**2
                    - 1176 * y**2
                    + 1859 * (x**2 + y**2) ** 4
                    - 4356 * (x**2 + y**2) ** 3
                    + 3564 * (x**2 + y**2) ** 2
                    + 126
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    -840 * x**2
                    - 840 * y**2
                    + 715 * (x**2 + y**2) ** 4
                    - 1980 * (x**2 + y**2) ** 3
                    + 1980 * (x**2 + y**2) ** 2
                    + 126
                )
                * sin(5 * atan2(y, x))
            )
        )
    if m == 5 and n == 13 and wrt == "y":
        return (
            10
            * sqrt(7)
            * (x**2 + y**2) ** (3 / 2)
            * (
                -x
                * (
                    -840 * x**2
                    - 840 * y**2
                    + 715 * (x**2 + y**2) ** 4
                    - 1980 * (x**2 + y**2) ** 3
                    + 1980 * (x**2 + y**2) ** 2
                    + 126
                )
                * sin(5 * atan2(y, x))
                + y
                * (
                    -1176 * x**2
                    - 1176 * y**2
                    + 1859 * (x**2 + y**2) ** 4
                    - 4356 * (x**2 + y**2) ** 3
                    + 3564 * (x**2 + y**2) ** 2
                    + 126
                )
                * cos(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 101
    if m == 7 and n == 13 and wrt == "x":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    4455 * x**2
                    + 4455 * y**2
                    + 3718 * (x**2 + y**2) ** 3
                    - 7260 * (x**2 + y**2) ** 2
                    - 840
                )
                * cos(7 * atan2(y, x))
                + 7
                * y
                * (
                    495 * x**2
                    + 495 * y**2
                    + 286 * (x**2 + y**2) ** 3
                    - 660 * (x**2 + y**2) ** 2
                    - 120
                )
                * sin(7 * atan2(y, x))
            )
        )
    if m == 7 and n == 13 and wrt == "y":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (5 / 2)
            * (
                -7
                * x
                * (
                    495 * x**2
                    + 495 * y**2
                    + 286 * (x**2 + y**2) ** 3
                    - 660 * (x**2 + y**2) ** 2
                    - 120
                )
                * sin(7 * atan2(y, x))
                + y
                * (
                    4455 * x**2
                    + 4455 * y**2
                    + 3718 * (x**2 + y**2) ** 3
                    - 7260 * (x**2 + y**2) ** 2
                    - 840
                )
                * cos(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 102
    if m == 9 and n == 13 and wrt == "x":
        return (
            6
            * sqrt(7)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x
                * (
                    -484 * x**2
                    - 484 * y**2
                    + 338 * (x**2 + y**2) ** 2
                    + 165
                )
                * cos(9 * atan2(y, x))
                + 3
                * y
                * (
                    -132 * x**2
                    - 132 * y**2
                    + 78 * (x**2 + y**2) ** 2
                    + 55
                )
                * sin(9 * atan2(y, x))
            )
        )
    if m == 9 and n == 13 and wrt == "y":
        return (
            6
            * sqrt(7)
            * (x**2 + y**2) ** (7 / 2)
            * (
                -3
                * x
                * (
                    -132 * x**2
                    - 132 * y**2
                    + 78 * (x**2 + y**2) ** 2
                    + 55
                )
                * sin(9 * atan2(y, x))
                + y
                * (
                    -484 * x**2
                    - 484 * y**2
                    + 338 * (x**2 + y**2) ** 2
                    + 165
                )
                * cos(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 103
    if m == 11 and n == 13 and wrt == "x":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (9 / 2)
            * (
                x * (169 * x**2 + 169 * y**2 - 132) * cos(11 * atan2(y, x))
                + 11
                * y
                * (13 * x**2 + 13 * y**2 - 12)
                * sin(11 * atan2(y, x))
            )
        )
    if m == 11 and n == 13 and wrt == "y":
        return (
            2
            * sqrt(7)
            * (x**2 + y**2) ** (9 / 2)
            * (
                -11
                * x
                * (13 * x**2 + 13 * y**2 - 12)
                * sin(11 * atan2(y, x))
                + y
                * (169 * x**2 + 169 * y**2 - 132)
                * cos(11 * atan2(y, x))
            )
        )

    # Derivatives for j = 104
    if m == 13 and n == 13 and wrt == "x":
        return (
            26
            * sqrt(7)
            * (x**2 + y**2) ** (11 / 2)
            * (x * cos(13 * atan2(y, x)) + y * sin(13 * atan2(y, x)))
        )
    if m == 13 and n == 13 and wrt == "y":
        return (
            26
            * sqrt(7)
            * (x**2 + y**2) ** (11 / 2)
            * (-x * sin(13 * atan2(y, x)) + y * cos(13 * atan2(y, x)))
        )

    # Derivatives for j = 105
    if m == -14 and n == 14 and wrt == "x":
        return (
            14
            * sqrt(30)
            * (x**2 + y**2) ** 6
            * (x * sin(14 * atan2(y, x)) - y * cos(14 * atan2(y, x)))
        )
    if m == -14 and n == 14 and wrt == "y":
        return (
            14
            * sqrt(30)
            * (x**2 + y**2) ** 6
            * (x * cos(14 * atan2(y, x)) + y * sin(14 * atan2(y, x)))
        )

    # Derivatives for j = 106
    if m == -12 and n == 14 and wrt == "x":
        return (
            4
            * sqrt(30)
            * (x**2 + y**2) ** 5
            * (
                x * (49 * x**2 + 49 * y**2 - 39) * sin(12 * atan2(y, x))
                - 3
                * y
                * (14 * x**2 + 14 * y**2 - 13)
                * cos(12 * atan2(y, x))
            )
        )
    if m == -12 and n == 14 and wrt == "y":
        return (
            4
            * sqrt(30)
            * (x**2 + y**2) ** 5
            * (
                3
                * x
                * (14 * x**2 + 14 * y**2 - 13)
                * cos(12 * atan2(y, x))
                + y * (49 * x**2 + 49 * y**2 - 39) * sin(12 * atan2(y, x))
            )
        )

    # Derivatives for j = 107
    if m == -10 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 4
            * (
                x
                * (
                    -936 * x**2
                    - 936 * y**2
                    + 637 * (x**2 + y**2) ** 2
                    + 330
                )
                * sin(10 * atan2(y, x))
                - 5
                * y
                * (
                    -156 * x**2
                    - 156 * y**2
                    + 91 * (x**2 + y**2) ** 2
                    + 66
                )
                * cos(10 * atan2(y, x))
            )
        )
    if m == -10 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 4
            * (
                5
                * x
                * (
                    -156 * x**2
                    - 156 * y**2
                    + 91 * (x**2 + y**2) ** 2
                    + 66
                )
                * cos(10 * atan2(y, x))
                + y
                * (
                    -936 * x**2
                    - 936 * y**2
                    + 637 * (x**2 + y**2) ** 2
                    + 330
                )
                * sin(10 * atan2(y, x))
            )
        )

    # Derivatives for j = 108
    if m == -8 and n == 14 and wrt == "x":
        return (
            8
            * sqrt(30)
            * y
            * (
                8
                * x**2
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                * (
                    825 * x**2
                    + 825 * y**2
                    + 637 * (x**2 + y**2) ** 3
                    - 1287 * (x**2 + y**2) ** 2
                    - 165
                )
                - (
                    660 * x**2
                    + 660 * y**2
                    + 364 * (x**2 + y**2) ** 3
                    - 858 * (x**2 + y**2) ** 2
                    - 165
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )
    if m == -8 and n == 14 and wrt == "y":
        return (
            8
            * sqrt(30)
            * x
            * (
                8
                * y**2
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                * (
                    825 * x**2
                    + 825 * y**2
                    + 637 * (x**2 + y**2) ** 3
                    - 1287 * (x**2 + y**2) ** 2
                    - 165
                )
                + (
                    660 * x**2
                    + 660 * y**2
                    + 364 * (x**2 + y**2) ** 3
                    - 858 * (x**2 + y**2) ** 2
                    - 165
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 109
    if m == -6 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    -5280 * x**2
                    - 5280 * y**2
                    + 7007 * (x**2 + y**2) ** 4
                    - 17160 * (x**2 + y**2) ** 3
                    + 14850 * (x**2 + y**2) ** 2
                    + 630
                )
                * sin(6 * atan2(y, x))
                - 3
                * y
                * (
                    -1320 * x**2
                    - 1320 * y**2
                    + 1001 * (x**2 + y**2) ** 4
                    - 2860 * (x**2 + y**2) ** 3
                    + 2970 * (x**2 + y**2) ** 2
                    + 210
                )
                * cos(6 * atan2(y, x))
            )
        )
    if m == -6 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 2
            * (
                3
                * x
                * (
                    -1320 * x**2
                    - 1320 * y**2
                    + 1001 * (x**2 + y**2) ** 4
                    - 2860 * (x**2 + y**2) ** 3
                    + 2970 * (x**2 + y**2) ** 2
                    + 210
                )
                * cos(6 * atan2(y, x))
                + y
                * (
                    -5280 * x**2
                    - 5280 * y**2
                    + 7007 * (x**2 + y**2) ** 4
                    - 17160 * (x**2 + y**2) ** 3
                    + 14850 * (x**2 + y**2) ** 2
                    + 630
                )
                * sin(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 110
    if m == -4 and n == 14 and wrt == "x":
        return (
            4
            * sqrt(30)
            * y
            * (
                4
                * x**2
                * (x**2 - y**2)
                * (
                    1890 * x**2
                    + 1890 * y**2
                    + 7007 * (x**2 + y**2) ** 5
                    - 19305 * (x**2 + y**2) ** 4
                    + 19800 * (x**2 + y**2) ** 3
                    - 9240 * (x**2 + y**2) ** 2
                    - 126
                )
                - (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    1260 * x**2
                    + 1260 * y**2
                    + 2002 * (x**2 + y**2) ** 5
                    - 6435 * (x**2 + y**2) ** 4
                    + 7920 * (x**2 + y**2) ** 3
                    - 4620 * (x**2 + y**2) ** 2
                    - 126
                )
            )
            / (x**2 + y**2)
        )
    if m == -4 and n == 14 and wrt == "y":
        return (
            4
            * sqrt(30)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    1890 * x**2
                    + 1890 * y**2
                    + 7007 * (x**2 + y**2) ** 5
                    - 19305 * (x**2 + y**2) ** 4
                    + 19800 * (x**2 + y**2) ** 3
                    - 9240 * (x**2 + y**2) ** 2
                    - 126
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    1260 * x**2
                    + 1260 * y**2
                    + 2002 * (x**2 + y**2) ** 5
                    - 6435 * (x**2 + y**2) ** 4
                    + 7920 * (x**2 + y**2) ** 3
                    - 4620 * (x**2 + y**2) ** 2
                    - 126
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 111
    if m == -2 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * y
            * (
                2
                * x**2
                * (
                    -1008 * x**2
                    - 1008 * y**2
                    + 21021 * (x**2 + y**2) ** 6
                    - 61776 * (x**2 + y**2) ** 5
                    + 69300 * (x**2 + y**2) ** 4
                    - 36960 * (x**2 + y**2) ** 3
                    + 9450 * (x**2 + y**2) ** 2
                    + 28
                )
                - (x**2 - y**2)
                * (
                    28 * x**2
                    + 28 * y**2
                    + 3003 * (x**2 + y**2) ** 7
                    - 10296 * (x**2 + y**2) ** 6
                    + 13860 * (x**2 + y**2) ** 5
                    - 9240 * (x**2 + y**2) ** 4
                    + 3150 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )
    if m == -2 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * x
            * (
                2
                * y**2
                * (
                    -1008 * x**2
                    - 1008 * y**2
                    + 21021 * (x**2 + y**2) ** 6
                    - 61776 * (x**2 + y**2) ** 5
                    + 69300 * (x**2 + y**2) ** 4
                    - 36960 * (x**2 + y**2) ** 3
                    + 9450 * (x**2 + y**2) ** 2
                    + 28
                )
                + (x**2 - y**2)
                * (
                    28 * x**2
                    + 28 * y**2
                    + 3003 * (x**2 + y**2) ** 7
                    - 10296 * (x**2 + y**2) ** 6
                    + 13860 * (x**2 + y**2) ** 5
                    - 9240 * (x**2 + y**2) ** 4
                    + 3150 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 112
    if m == 0 and n == 14 and wrt == "x":
        return (
            112
            * sqrt(15)
            * x
            * (
                -27 * x**2
                - 27 * y**2
                + 429 * (x**2 + y**2) ** 6
                - 1287 * (x**2 + y**2) ** 5
                + 1485 * (x**2 + y**2) ** 4
                - 825 * (x**2 + y**2) ** 3
                + 225 * (x**2 + y**2) ** 2
                + 1
            )
        )
    if m == 0 and n == 14 and wrt == "y":
        return (
            112
            * sqrt(15)
            * y
            * (
                -27 * x**2
                - 27 * y**2
                + 429 * (x**2 + y**2) ** 6
                - 1287 * (x**2 + y**2) ** 5
                + 1485 * (x**2 + y**2) ** 4
                - 825 * (x**2 + y**2) ** 3
                + 225 * (x**2 + y**2) ** 2
                + 1
            )
        )

    # Derivatives for j = 113
    if m == 2 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * x
            * (
                2
                * y**2
                * (
                    28 * x**2
                    + 28 * y**2
                    + 3003 * (x**2 + y**2) ** 7
                    - 10296 * (x**2 + y**2) ** 6
                    + 13860 * (x**2 + y**2) ** 5
                    - 9240 * (x**2 + y**2) ** 4
                    + 3150 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    -1008 * x**2
                    - 1008 * y**2
                    + 21021 * (x**2 + y**2) ** 6
                    - 61776 * (x**2 + y**2) ** 5
                    + 69300 * (x**2 + y**2) ** 4
                    - 36960 * (x**2 + y**2) ** 3
                    + 9450 * (x**2 + y**2) ** 2
                    + 28
                )
            )
            / (x**2 + y**2)
        )
    if m == 2 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * y
            * (
                -2
                * x**2
                * (
                    28 * x**2
                    + 28 * y**2
                    + 3003 * (x**2 + y**2) ** 7
                    - 10296 * (x**2 + y**2) ** 6
                    + 13860 * (x**2 + y**2) ** 5
                    - 9240 * (x**2 + y**2) ** 4
                    + 3150 * (x**2 + y**2) ** 3
                    - 504 * (x**2 + y**2) ** 2
                )
                / (x**2 + y**2)
                + (x**2 - y**2)
                * (
                    -1008 * x**2
                    - 1008 * y**2
                    + 21021 * (x**2 + y**2) ** 6
                    - 61776 * (x**2 + y**2) ** 5
                    + 69300 * (x**2 + y**2) ** 4
                    - 36960 * (x**2 + y**2) ** 3
                    + 9450 * (x**2 + y**2) ** 2
                    + 28
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 114
    if m == 4 and n == 14 and wrt == "x":
        return (
            4
            * sqrt(30)
            * x
            * (
                4
                * y**2
                * (x**2 - y**2)
                * (
                    1260 * x**2
                    + 1260 * y**2
                    + 2002 * (x**2 + y**2) ** 5
                    - 6435 * (x**2 + y**2) ** 4
                    + 7920 * (x**2 + y**2) ** 3
                    - 4620 * (x**2 + y**2) ** 2
                    - 126
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    1890 * x**2
                    + 1890 * y**2
                    + 7007 * (x**2 + y**2) ** 5
                    - 19305 * (x**2 + y**2) ** 4
                    + 19800 * (x**2 + y**2) ** 3
                    - 9240 * (x**2 + y**2) ** 2
                    - 126
                )
            )
            / (x**2 + y**2)
        )
    if m == 4 and n == 14 and wrt == "y":
        return (
            4
            * sqrt(30)
            * y
            * (
                -4
                * x**2
                * (x**2 - y**2)
                * (
                    1260 * x**2
                    + 1260 * y**2
                    + 2002 * (x**2 + y**2) ** 5
                    - 6435 * (x**2 + y**2) ** 4
                    + 7920 * (x**2 + y**2) ** 3
                    - 4620 * (x**2 + y**2) ** 2
                    - 126
                )
                + (x**4 - 6 * x**2 * y**2 + y**4)
                * (
                    1890 * x**2
                    + 1890 * y**2
                    + 7007 * (x**2 + y**2) ** 5
                    - 19305 * (x**2 + y**2) ** 4
                    + 19800 * (x**2 + y**2) ** 3
                    - 9240 * (x**2 + y**2) ** 2
                    - 126
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 115
    if m == 6 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 2
            * (
                x
                * (
                    -5280 * x**2
                    - 5280 * y**2
                    + 7007 * (x**2 + y**2) ** 4
                    - 17160 * (x**2 + y**2) ** 3
                    + 14850 * (x**2 + y**2) ** 2
                    + 630
                )
                * cos(6 * atan2(y, x))
                + 3
                * y
                * (
                    -1320 * x**2
                    - 1320 * y**2
                    + 1001 * (x**2 + y**2) ** 4
                    - 2860 * (x**2 + y**2) ** 3
                    + 2970 * (x**2 + y**2) ** 2
                    + 210
                )
                * sin(6 * atan2(y, x))
            )
        )
    if m == 6 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 2
            * (
                -3
                * x
                * (
                    -1320 * x**2
                    - 1320 * y**2
                    + 1001 * (x**2 + y**2) ** 4
                    - 2860 * (x**2 + y**2) ** 3
                    + 2970 * (x**2 + y**2) ** 2
                    + 210
                )
                * sin(6 * atan2(y, x))
                + y
                * (
                    -5280 * x**2
                    - 5280 * y**2
                    + 7007 * (x**2 + y**2) ** 4
                    - 17160 * (x**2 + y**2) ** 3
                    + 14850 * (x**2 + y**2) ** 2
                    + 630
                )
                * cos(6 * atan2(y, x))
            )
        )

    # Derivatives for j = 116
    if m == 8 and n == 14 and wrt == "x":
        return (
            8
            * sqrt(30)
            * x
            * (
                8
                * y**2
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                * (
                    660 * x**2
                    + 660 * y**2
                    + 364 * (x**2 + y**2) ** 3
                    - 858 * (x**2 + y**2) ** 2
                    - 165
                )
                + (
                    825 * x**2
                    + 825 * y**2
                    + 637 * (x**2 + y**2) ** 3
                    - 1287 * (x**2 + y**2) ** 2
                    - 165
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )
    if m == 8 and n == 14 and wrt == "y":
        return (
            8
            * sqrt(30)
            * y
            * (
                -8
                * x**2
                * (x**6 - 7 * x**4 * y**2 + 7 * x**2 * y**4 - y**6)
                * (
                    660 * x**2
                    + 660 * y**2
                    + 364 * (x**2 + y**2) ** 3
                    - 858 * (x**2 + y**2) ** 2
                    - 165
                )
                + (
                    825 * x**2
                    + 825 * y**2
                    + 637 * (x**2 + y**2) ** 3
                    - 1287 * (x**2 + y**2) ** 2
                    - 165
                )
                * (
                    x**8
                    - 28 * x**6 * y**2
                    + 70 * x**4 * y**4
                    - 28 * x**2 * y**6
                    + y**8
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 117
    if m == 10 and n == 14 and wrt == "x":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 4
            * (
                x
                * (
                    -936 * x**2
                    - 936 * y**2
                    + 637 * (x**2 + y**2) ** 2
                    + 330
                )
                * cos(10 * atan2(y, x))
                + 5
                * y
                * (
                    -156 * x**2
                    - 156 * y**2
                    + 91 * (x**2 + y**2) ** 2
                    + 66
                )
                * sin(10 * atan2(y, x))
            )
        )
    if m == 10 and n == 14 and wrt == "y":
        return (
            2
            * sqrt(30)
            * (x**2 + y**2) ** 4
            * (
                -5
                * x
                * (
                    -156 * x**2
                    - 156 * y**2
                    + 91 * (x**2 + y**2) ** 2
                    + 66
                )
                * sin(10 * atan2(y, x))
                + y
                * (
                    -936 * x**2
                    - 936 * y**2
                    + 637 * (x**2 + y**2) ** 2
                    + 330
                )
                * cos(10 * atan2(y, x))
            )
        )

    # Derivatives for j = 118
    if m == 12 and n == 14 and wrt == "x":
        return (
            4
            * sqrt(30)
            * (x**2 + y**2) ** 5
            * (
                x * (49 * x**2 + 49 * y**2 - 39) * cos(12 * atan2(y, x))
                + 3
                * y
                * (14 * x**2 + 14 * y**2 - 13)
                * sin(12 * atan2(y, x))
            )
        )
    if m == 12 and n == 14 and wrt == "y":
        return (
            4
            * sqrt(30)
            * (x**2 + y**2) ** 5
            * (
                -3
                * x
                * (14 * x**2 + 14 * y**2 - 13)
                * sin(12 * atan2(y, x))
                + y * (49 * x**2 + 49 * y**2 - 39) * cos(12 * atan2(y, x))
            )
        )

    # Derivatives for j = 119
    if m == 14 and n == 14 and wrt == "x":
        return (
            14
            * sqrt(30)
            * (x**2 + y**2) ** 6
            * (x * cos(14 * atan2(y, x)) + y * sin(14 * atan2(y, x)))
        )
    if m == 14 and n == 14 and wrt == "y":
        return (
            14
            * sqrt(30)
            * (x**2 + y**2) ** 6
            * (-x * sin(14 * atan2(y, x)) + y * cos(14 * atan2(y, x)))
        )

    # Derivatives for j = 120
    if m == -15 and n == 15 and wrt == "x":
        return (
            60
            * sqrt(2)
            * (x**2 + y**2) ** (13 / 2)
            * (x * sin(15 * atan2(y, x)) - y * cos(15 * atan2(y, x)))
        )
    if m == -15 and n == 15 and wrt == "y":
        return (
            60
            * sqrt(2)
            * (x**2 + y**2) ** (13 / 2)
            * (x * cos(15 * atan2(y, x)) + y * sin(15 * atan2(y, x)))
        )

    # Derivatives for j = 121
    if m == -13 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (11 / 2)
            * (
                x * (225 * x**2 + 225 * y**2 - 182) * sin(13 * atan2(y, x))
                - 13
                * y
                * (15 * x**2 + 15 * y**2 - 14)
                * cos(13 * atan2(y, x))
            )
        )
    if m == -13 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (11 / 2)
            * (
                13
                * x
                * (15 * x**2 + 15 * y**2 - 14)
                * cos(13 * atan2(y, x))
                + y
                * (225 * x**2 + 225 * y**2 - 182)
                * sin(13 * atan2(y, x))
            )
        )

    # Derivatives for j = 122
    if m == -11 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (9 / 2)
            * (
                x
                * (
                    -2366 * x**2
                    - 2366 * y**2
                    + 1575 * (x**2 + y**2) ** 2
                    + 858
                )
                * sin(11 * atan2(y, x))
                - 11
                * y
                * (
                    -182 * x**2
                    - 182 * y**2
                    + 105 * (x**2 + y**2) ** 2
                    + 78
                )
                * cos(11 * atan2(y, x))
            )
        )
    if m == -11 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (9 / 2)
            * (
                11
                * x
                * (
                    -182 * x**2
                    - 182 * y**2
                    + 105 * (x**2 + y**2) ** 2
                    + 78
                )
                * cos(11 * atan2(y, x))
                + y
                * (
                    -2366 * x**2
                    - 2366 * y**2
                    + 1575 * (x**2 + y**2) ** 2
                    + 858
                )
                * sin(11 * atan2(y, x))
            )
        )

    # Derivatives for j = 123
    if m == -9 and n == 15 and wrt == "x":
        return (
            12
            * sqrt(2)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x
                * (
                    3146 * x**2
                    + 3146 * y**2
                    + 2275 * (x**2 + y**2) ** 3
                    - 4732 * (x**2 + y**2) ** 2
                    - 660
                )
                * sin(9 * atan2(y, x))
                - 3
                * y
                * (
                    858 * x**2
                    + 858 * y**2
                    + 455 * (x**2 + y**2) ** 3
                    - 1092 * (x**2 + y**2) ** 2
                    - 220
                )
                * cos(9 * atan2(y, x))
            )
        )
    if m == -9 and n == 15 and wrt == "y":
        return (
            12
            * sqrt(2)
            * (x**2 + y**2) ** (7 / 2)
            * (
                3
                * x
                * (
                    858 * x**2
                    + 858 * y**2
                    + 455 * (x**2 + y**2) ** 3
                    - 1092 * (x**2 + y**2) ** 2
                    - 220
                )
                * cos(9 * atan2(y, x))
                + y
                * (
                    3146 * x**2
                    + 3146 * y**2
                    + 2275 * (x**2 + y**2) ** 3
                    - 4732 * (x**2 + y**2) ** 2
                    - 660
                )
                * sin(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 124
    if m == -7 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    -17820 * x**2
                    - 17820 * y**2
                    + 20475 * (x**2 + y**2) ** 4
                    - 52052 * (x**2 + y**2) ** 3
                    + 47190 * (x**2 + y**2) ** 2
                    + 2310
                )
                * sin(7 * atan2(y, x))
                - 7
                * y
                * (
                    -1980 * x**2
                    - 1980 * y**2
                    + 1365 * (x**2 + y**2) ** 4
                    - 4004 * (x**2 + y**2) ** 3
                    + 4290 * (x**2 + y**2) ** 2
                    + 330
                )
                * cos(7 * atan2(y, x))
            )
        )
    if m == -7 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (5 / 2)
            * (
                7
                * x
                * (
                    -1980 * x**2
                    - 1980 * y**2
                    + 1365 * (x**2 + y**2) ** 4
                    - 4004 * (x**2 + y**2) ** 3
                    + 4290 * (x**2 + y**2) ** 2
                    + 330
                )
                * cos(7 * atan2(y, x))
                + y
                * (
                    -17820 * x**2
                    - 17820 * y**2
                    + 20475 * (x**2 + y**2) ** 4
                    - 52052 * (x**2 + y**2) ** 3
                    + 47190 * (x**2 + y**2) ** 2
                    + 2310
                )
                * sin(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 125
    if m == -5 and n == 15 and wrt == "x":
        return (
            20
            * sqrt(2)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    3234 * x**2
                    + 3234 * y**2
                    + 9009 * (x**2 + y**2) ** 5
                    - 26026 * (x**2 + y**2) ** 4
                    + 28314 * (x**2 + y**2) ** 3
                    - 14256 * (x**2 + y**2) ** 2
                    - 252
                )
                * sin(5 * atan2(y, x))
                - y
                * (
                    2310 * x**2
                    + 2310 * y**2
                    + 3003 * (x**2 + y**2) ** 5
                    - 10010 * (x**2 + y**2) ** 4
                    + 12870 * (x**2 + y**2) ** 3
                    - 7920 * (x**2 + y**2) ** 2
                    - 252
                )
                * cos(5 * atan2(y, x))
            )
        )
    if m == -5 and n == 15 and wrt == "y":
        return (
            20
            * sqrt(2)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    2310 * x**2
                    + 2310 * y**2
                    + 3003 * (x**2 + y**2) ** 5
                    - 10010 * (x**2 + y**2) ** 4
                    + 12870 * (x**2 + y**2) ** 3
                    - 7920 * (x**2 + y**2) ** 2
                    - 252
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    3234 * x**2
                    + 3234 * y**2
                    + 9009 * (x**2 + y**2) ** 5
                    - 26026 * (x**2 + y**2) ** 4
                    + 28314 * (x**2 + y**2) ** 3
                    - 14256 * (x**2 + y**2) ** 2
                    - 252
                )
                * sin(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 126
    if m == -3 and n == 15 and wrt == "x":
        return (
            12
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -2100 * x**2
                    - 2100 * y**2
                    + 25025 * (x**2 + y**2) ** 6
                    - 78078 * (x**2 + y**2) ** 5
                    + 94380 * (x**2 + y**2) ** 4
                    - 55440 * (x**2 + y**2) ** 3
                    + 16170 * (x**2 + y**2) ** 2
                    + 84
                )
                * sin(3 * atan2(y, x))
                - y
                * (
                    -1260 * x**2
                    - 1260 * y**2
                    + 5005 * (x**2 + y**2) ** 6
                    - 18018 * (x**2 + y**2) ** 5
                    + 25740 * (x**2 + y**2) ** 4
                    - 18480 * (x**2 + y**2) ** 3
                    + 6930 * (x**2 + y**2) ** 2
                    + 84
                )
                * cos(3 * atan2(y, x))
            )
        )
    if m == -3 and n == 15 and wrt == "y":
        return (
            12
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -1260 * x**2
                    - 1260 * y**2
                    + 5005 * (x**2 + y**2) ** 6
                    - 18018 * (x**2 + y**2) ** 5
                    + 25740 * (x**2 + y**2) ** 4
                    - 18480 * (x**2 + y**2) ** 3
                    + 6930 * (x**2 + y**2) ** 2
                    + 84
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    -2100 * x**2
                    - 2100 * y**2
                    + 25025 * (x**2 + y**2) ** 6
                    - 78078 * (x**2 + y**2) ** 5
                    + 94380 * (x**2 + y**2) ** 4
                    - 55440 * (x**2 + y**2) ** 3
                    + 16170 * (x**2 + y**2) ** 2
                    + 84
                )
                * sin(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 127
    if m == -1 and n == 15 and wrt == "x":
        return (
            12
            * sqrt(2)
            * x
            * y
            * (
                -84 * x**2
                - 84 * y**2
                - 2145 * (x**2 + y**2) ** 7
                + 8008 * (x**2 + y**2) ** 6
                - 12012 * (x**2 + y**2) ** 5
                + 9240 * (x**2 + y**2) ** 4
                - 3850 * (x**2 + y**2) ** 3
                + 840 * (x**2 + y**2) ** 2
                + (x**2 + y**2)
                * (
                    -4200 * x**2
                    - 4200 * y**2
                    + 32175 * (x**2 + y**2) ** 6
                    - 104104 * (x**2 + y**2) ** 5
                    + 132132 * (x**2 + y**2) ** 4
                    - 83160 * (x**2 + y**2) ** 3
                    + 26950 * (x**2 + y**2) ** 2
                    + 252
                )
            )
            / (x**2 + y**2)
        )
    if m == -1 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (
                y**2
                * (
                    3
                    * (x**2 + y**2)
                    * (
                        -4200 * x**2
                        - 4200 * y**2
                        + 32175 * (x**2 + y**2) ** 6
                        - 104104 * (x**2 + y**2) ** 5
                        + 132132 * (x**2 + y**2) ** 4
                        - 83160 * (x**2 + y**2) ** 3
                        + 26950 * (x**2 + y**2) ** 2
                        + 252
                    )
                    - 8
                )
                + y**2
                * (
                    -252 * x**2
                    - 252 * y**2
                    - 6435 * (x**2 + y**2) ** 7
                    + 24024 * (x**2 + y**2) ** 6
                    - 36036 * (x**2 + y**2) ** 5
                    + 27720 * (x**2 + y**2) ** 4
                    - 11550 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 8
                )
                + (x**2 + y**2)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 6435 * (x**2 + y**2) ** 7
                    - 24024 * (x**2 + y**2) ** 6
                    + 36036 * (x**2 + y**2) ** 5
                    - 27720 * (x**2 + y**2) ** 4
                    + 11550 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 8
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 128
    if m == 1 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (
                x**2
                * (
                    3
                    * (x**2 + y**2)
                    * (
                        -4200 * x**2
                        - 4200 * y**2
                        + 32175 * (x**2 + y**2) ** 6
                        - 104104 * (x**2 + y**2) ** 5
                        + 132132 * (x**2 + y**2) ** 4
                        - 83160 * (x**2 + y**2) ** 3
                        + 26950 * (x**2 + y**2) ** 2
                        + 252
                    )
                    - 8
                )
                + x**2
                * (
                    -252 * x**2
                    - 252 * y**2
                    - 6435 * (x**2 + y**2) ** 7
                    + 24024 * (x**2 + y**2) ** 6
                    - 36036 * (x**2 + y**2) ** 5
                    + 27720 * (x**2 + y**2) ** 4
                    - 11550 * (x**2 + y**2) ** 3
                    + 2520 * (x**2 + y**2) ** 2
                    + 8
                )
                + (x**2 + y**2)
                * (
                    252 * x**2
                    + 252 * y**2
                    + 6435 * (x**2 + y**2) ** 7
                    - 24024 * (x**2 + y**2) ** 6
                    + 36036 * (x**2 + y**2) ** 5
                    - 27720 * (x**2 + y**2) ** 4
                    + 11550 * (x**2 + y**2) ** 3
                    - 2520 * (x**2 + y**2) ** 2
                    - 8
                )
            )
            / (x**2 + y**2)
        )
    if m == 1 and n == 15 and wrt == "y":
        return (
            12
            * sqrt(2)
            * x
            * y
            * (
                -84 * x**2
                - 84 * y**2
                - 2145 * (x**2 + y**2) ** 7
                + 8008 * (x**2 + y**2) ** 6
                - 12012 * (x**2 + y**2) ** 5
                + 9240 * (x**2 + y**2) ** 4
                - 3850 * (x**2 + y**2) ** 3
                + 840 * (x**2 + y**2) ** 2
                + (x**2 + y**2)
                * (
                    -4200 * x**2
                    - 4200 * y**2
                    + 32175 * (x**2 + y**2) ** 6
                    - 104104 * (x**2 + y**2) ** 5
                    + 132132 * (x**2 + y**2) ** 4
                    - 83160 * (x**2 + y**2) ** 3
                    + 26950 * (x**2 + y**2) ** 2
                    + 252
                )
            )
            / (x**2 + y**2)
        )

    # Derivatives for j = 129
    if m == 3 and n == 15 and wrt == "x":
        return (
            12
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (
                x
                * (
                    -2100 * x**2
                    - 2100 * y**2
                    + 25025 * (x**2 + y**2) ** 6
                    - 78078 * (x**2 + y**2) ** 5
                    + 94380 * (x**2 + y**2) ** 4
                    - 55440 * (x**2 + y**2) ** 3
                    + 16170 * (x**2 + y**2) ** 2
                    + 84
                )
                * cos(3 * atan2(y, x))
                + y
                * (
                    -1260 * x**2
                    - 1260 * y**2
                    + 5005 * (x**2 + y**2) ** 6
                    - 18018 * (x**2 + y**2) ** 5
                    + 25740 * (x**2 + y**2) ** 4
                    - 18480 * (x**2 + y**2) ** 3
                    + 6930 * (x**2 + y**2) ** 2
                    + 84
                )
                * sin(3 * atan2(y, x))
            )
        )
    if m == 3 and n == 15 and wrt == "y":
        return (
            12
            * sqrt(2)
            * sqrt(x**2 + y**2)
            * (
                -x
                * (
                    -1260 * x**2
                    - 1260 * y**2
                    + 5005 * (x**2 + y**2) ** 6
                    - 18018 * (x**2 + y**2) ** 5
                    + 25740 * (x**2 + y**2) ** 4
                    - 18480 * (x**2 + y**2) ** 3
                    + 6930 * (x**2 + y**2) ** 2
                    + 84
                )
                * sin(3 * atan2(y, x))
                + y
                * (
                    -2100 * x**2
                    - 2100 * y**2
                    + 25025 * (x**2 + y**2) ** 6
                    - 78078 * (x**2 + y**2) ** 5
                    + 94380 * (x**2 + y**2) ** 4
                    - 55440 * (x**2 + y**2) ** 3
                    + 16170 * (x**2 + y**2) ** 2
                    + 84
                )
                * cos(3 * atan2(y, x))
            )
        )

    # Derivatives for j = 130
    if m == 5 and n == 15 and wrt == "x":
        return (
            20
            * sqrt(2)
            * (x**2 + y**2) ** (3 / 2)
            * (
                x
                * (
                    3234 * x**2
                    + 3234 * y**2
                    + 9009 * (x**2 + y**2) ** 5
                    - 26026 * (x**2 + y**2) ** 4
                    + 28314 * (x**2 + y**2) ** 3
                    - 14256 * (x**2 + y**2) ** 2
                    - 252
                )
                * cos(5 * atan2(y, x))
                + y
                * (
                    2310 * x**2
                    + 2310 * y**2
                    + 3003 * (x**2 + y**2) ** 5
                    - 10010 * (x**2 + y**2) ** 4
                    + 12870 * (x**2 + y**2) ** 3
                    - 7920 * (x**2 + y**2) ** 2
                    - 252
                )
                * sin(5 * atan2(y, x))
            )
        )
    if m == 5 and n == 15 and wrt == "y":
        return (
            20
            * sqrt(2)
            * (x**2 + y**2) ** (3 / 2)
            * (
                -x
                * (
                    2310 * x**2
                    + 2310 * y**2
                    + 3003 * (x**2 + y**2) ** 5
                    - 10010 * (x**2 + y**2) ** 4
                    + 12870 * (x**2 + y**2) ** 3
                    - 7920 * (x**2 + y**2) ** 2
                    - 252
                )
                * sin(5 * atan2(y, x))
                + y
                * (
                    3234 * x**2
                    + 3234 * y**2
                    + 9009 * (x**2 + y**2) ** 5
                    - 26026 * (x**2 + y**2) ** 4
                    + 28314 * (x**2 + y**2) ** 3
                    - 14256 * (x**2 + y**2) ** 2
                    - 252
                )
                * cos(5 * atan2(y, x))
            )
        )

    # Derivatives for j = 131
    if m == 7 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (5 / 2)
            * (
                x
                * (
                    -17820 * x**2
                    - 17820 * y**2
                    + 20475 * (x**2 + y**2) ** 4
                    - 52052 * (x**2 + y**2) ** 3
                    + 47190 * (x**2 + y**2) ** 2
                    + 2310
                )
                * cos(7 * atan2(y, x))
                + 7
                * y
                * (
                    -1980 * x**2
                    - 1980 * y**2
                    + 1365 * (x**2 + y**2) ** 4
                    - 4004 * (x**2 + y**2) ** 3
                    + 4290 * (x**2 + y**2) ** 2
                    + 330
                )
                * sin(7 * atan2(y, x))
            )
        )
    if m == 7 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (5 / 2)
            * (
                -7
                * x
                * (
                    -1980 * x**2
                    - 1980 * y**2
                    + 1365 * (x**2 + y**2) ** 4
                    - 4004 * (x**2 + y**2) ** 3
                    + 4290 * (x**2 + y**2) ** 2
                    + 330
                )
                * sin(7 * atan2(y, x))
                + y
                * (
                    -17820 * x**2
                    - 17820 * y**2
                    + 20475 * (x**2 + y**2) ** 4
                    - 52052 * (x**2 + y**2) ** 3
                    + 47190 * (x**2 + y**2) ** 2
                    + 2310
                )
                * cos(7 * atan2(y, x))
            )
        )

    # Derivatives for j = 132
    if m == 9 and n == 15 and wrt == "x":
        return (
            12
            * sqrt(2)
            * (x**2 + y**2) ** (7 / 2)
            * (
                x
                * (
                    3146 * x**2
                    + 3146 * y**2
                    + 2275 * (x**2 + y**2) ** 3
                    - 4732 * (x**2 + y**2) ** 2
                    - 660
                )
                * cos(9 * atan2(y, x))
                + 3
                * y
                * (
                    858 * x**2
                    + 858 * y**2
                    + 455 * (x**2 + y**2) ** 3
                    - 1092 * (x**2 + y**2) ** 2
                    - 220
                )
                * sin(9 * atan2(y, x))
            )
        )
    if m == 9 and n == 15 and wrt == "y":
        return (
            12
            * sqrt(2)
            * (x**2 + y**2) ** (7 / 2)
            * (
                -3
                * x
                * (
                    858 * x**2
                    + 858 * y**2
                    + 455 * (x**2 + y**2) ** 3
                    - 1092 * (x**2 + y**2) ** 2
                    - 220
                )
                * sin(9 * atan2(y, x))
                + y
                * (
                    3146 * x**2
                    + 3146 * y**2
                    + 2275 * (x**2 + y**2) ** 3
                    - 4732 * (x**2 + y**2) ** 2
                    - 660
                )
                * cos(9 * atan2(y, x))
            )
        )

    # Derivatives for j = 133
    if m == 11 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (9 / 2)
            * (
                x
                * (
                    -2366 * x**2
                    - 2366 * y**2
                    + 1575 * (x**2 + y**2) ** 2
                    + 858
                )
                * cos(11 * atan2(y, x))
                + 11
                * y
                * (
                    -182 * x**2
                    - 182 * y**2
                    + 105 * (x**2 + y**2) ** 2
                    + 78
                )
                * sin(11 * atan2(y, x))
            )
        )
    if m == 11 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (9 / 2)
            * (
                -11
                * x
                * (
                    -182 * x**2
                    - 182 * y**2
                    + 105 * (x**2 + y**2) ** 2
                    + 78
                )
                * sin(11 * atan2(y, x))
                + y
                * (
                    -2366 * x**2
                    - 2366 * y**2
                    + 1575 * (x**2 + y**2) ** 2
                    + 858
                )
                * cos(11 * atan2(y, x))
            )
        )

    # Derivatives for j = 134
    if m == 13 and n == 15 and wrt == "x":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (11 / 2)
            * (
                x * (225 * x**2 + 225 * y**2 - 182) * cos(13 * atan2(y, x))
                + 13
                * y
                * (15 * x**2 + 15 * y**2 - 14)
                * sin(13 * atan2(y, x))
            )
        )
    if m == 13 and n == 15 and wrt == "y":
        return (
            4
            * sqrt(2)
            * (x**2 + y**2) ** (11 / 2)
            * (
                -13
                * x
                * (15 * x**2 + 15 * y**2 - 14)
                * sin(13 * atan2(y, x))
                + y
                * (225 * x**2 + 225 * y**2 - 182)
                * cos(13 * atan2(y, x))
            )
        )

    # Derivatives for j = 135
    if m == 15 and n == 15 and wrt == "x":
        return (
            60
            * sqrt(2)
            * (x**2 + y**2) ** (13 / 2)
            * (x * cos(15 * atan2(y, x)) + y * sin(15 * atan2(y, x)))
        )
    if m == 15 and n == 15 and wrt == "y":
        return (
            60
            * sqrt(2)
            * (x**2 + y**2) ** (13 / 2)
            * (-x * sin(15 * atan2(y, x)) + y * cos(15 * atan2(y, x)))
        )

    # Raise value error if we have not returned yet
    raise ValueError(
        "No pre-computed derivative available for given arguments!"
    )
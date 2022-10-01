# Originally contributed by Xmijybke_BroCat.

"""Class for matrix, matrix calculate."""

from fractions import Fraction
# from Polynomial import Polynomial
from typing import Literal

__all__ = ['Matrix', 'unicode_matrix', 'latex_matrix', 'unit', 'diagonal_matrix']


class Matrix(object):
    """This class restore matrix.

    The class use nested list to contain every value.
    Each value should be an integer or fraction, maybe you need to import
    fractions. In fact, float type won't raise any error in it, but I'm
    sure you don't want to see an outrageous fraction like
    5404319552844595/4503599627370496.

    Here is an example:
    [[1, Fraction(2, 5), 3], [Fraction(4, 3), 7, -2], [3, -2, 6]]
    It means this matrix:

    |  1  2/5  3 |
    | 4/3  7  -2 |
    |  7  -2   6 | 3*3
    """
    __slots__ = ["_rows"]

    def __init__(self, rows: list[list[int | Fraction]]):
        """Constructs matrix object from nested list.

        Args:
            rows: a matrix written in a nested list.
                The matrix needn't be a square matrix

        Return:
            A matrix object

        Raises:
            SyntaxError: if rows isn't a nested matrix, or in the matrix
                different row has different length.
        """
        if isinstance(rows, list):
            if isinstance(rows[0], list):
                for i in range(1, len(rows)):
                    if len(rows[i]) != len(rows[0]):
                        raise MatrixError("unfixed row length in the matrix")
                self._rows = rows
            else:
                raise SyntaxError("values should be in a nested list")
        else:
            raise SyntaxError("values should be in a nested list")

    def __getitem__(self, item):
        """A[i, j]"""
        if isinstance(item, int):
            return self._rows[item]
        elif isinstance(item, tuple):
            if len(item) == 2:
                index_i, index_j = item
                return self._rows[index_i][index_j]
            else:
                len_tuple = len(item)
                raise ValueError("matrix only accept 2 index, %d were given"
                                 % len_tuple)
        elif item == "rn":  # rn = row number
            return len(self._rows)
        elif item == "cn":  # cn = column number
            return len(self._rows[0])
        elif item == "self":
            return self._rows
        else:
            return NotImplemented

    def show_matrix(self, length: int = 4) -> str:
        """prints the matrix directly.

        This function shows the matrix in a usual format: each row will be
        written in a line and there is a tab between values in the row.
        You can set how long each value is in the argument 'length'

        Args:
            length: The length of each value, defaulted to 4.

        Return:
            a long string shows the matrix.
        """
        output_rows = []
        for i in range(self["rn"]):
            output_row = "\t".join(str(value).center(length) for value in self[i])
            output_rows.append(" |  " + output_row + "  |")
        output = "\n".join(r for r in output_rows)
        return output + " %d*%d\n" % (self["rn"], self["cn"])

    def __str__(self):
        """print()"""
        max_length = 0
        for i in range(self["rn"]):
            for j in range(self["cn"]):
                value_length = len(str(self[i, j]))
                if value_length > max_length:
                    max_length = value_length
        if max_length % 4 != 0:
            max_length = 4 * (max_length // 4) + 4
        return self.show_matrix(max_length)

    def __repr__(self):
        """repr()"""
        return "Matrix(%s)" % repr(self._rows)

    def get_value(self, index_i: int, index_j: int):
        """gets a value in the matrix.

        The index of the first row or the first column is 1, not 0, that means
        self.get_value(1, 1) will return the value in the first row, the first
        column, and self.get_value(2, 3) will return the value in the second
        row, the third column.

        Args:
             index_i: the row number.
             index_j: the column number.

        Return:
            the value in the index_i row, the index_j column.

        Raises:
            IndexError: if the index is too large.
        """
        if index_i <= self["rn"] and index_j <= self["cn"]:
            return self[index_i-1, index_j-1]
        else:
            raise IndexError("index out of range")

    def locate(self, value: int | Fraction | float):
        """locates a value in the matrix

        If there are multiple satisfied position, return the one has a smaller
        row number.

        Args:
            value: the value you want to find.

        Return:
            the value's position.

        Raises:
            ValueError: if the matrix doesn't have this value."""
        position = None
        for i in range(self["rn"]):
            for j in range(self["cn"]):
                if self[i, j] == value:
                    position = (i+1, j+1)
                    return position
        if position is None:
            raise ValueError("found no '%s' in the matrix" % value)

    def unicode_math(self):
        """prints the matrix in unicode-math format."""
        output_rows = []
        for i in range(self["rn"]):
            output_row = "&".join(str(value) for value in self[i])
            output_rows.append(output_row)
        output = "@".join(r for r in output_rows)
        return "■(" + output + ")"

    def latex(self, bracket: Literal["n", "b", "p", "v", "V"] = "n") -> str:
        """prints the matrix in LaTeX format.

        Here arg 'bracket' decides the matrix's bracket format, 'n' means no
        bracket, 'b' means [matrix], 'p' means (matrix), 'v' means |matrix|
        and 'V' means ||matrix||

        :arg bracket: the format of the matrix's bracket.
        """
        brackets = {"n": (r"\begin{matrix} ", r" \end{matrix} "),
                    "b": (r"\begin{bmatrix} ", r" \end{bmatrix} "),
                    "p": (r"\begin{pmatrix} ", r" \end{pmatrix} "),
                    "v": (r"\begin{vmatrix} ", r" \end{vmatrix} "),
                    "V": (r"\begin{Vmatrix} ", r" \end{Vmatrix} ")}
        output_rows = []
        for i in range(self["rn"]):
            output_row = " & ".join(str(v) for v in self[i])
            output_rows.append(output_row)
        output = r" \\ ".join(r for r in output_rows)
        return brackets[bracket][0] + output + brackets[bracket][1]

    def rtrans(self, transformation: str):
        """transforms the rows in the matrix by selected transformation.

        Here, (i,j) means exchanging the ith row and the jth row in the
        matrix, k(i) means multiply the ith row in the matrix by k, while
        k(i)+(j) means add k times of the ith row to the jth row.

        Args:
            transformation: the transformation, (i, j), k(i) or k(i)+(j)
        """
        output = []
        if "," in transformation:
            trans = transformation[1:len(transformation)-1]
            r1, r2 = map(int, trans.split(","))
            for i in range(self["rn"]):
                if i == r1-1:
                    output.append(self[r2-1])
                elif i == r2-1:
                    output.append(self[r1-1])
                else:
                    output.append(self[i])
        elif "+" in transformation:
            pt1, r2 = transformation.split("+")
            r2 = int(r2[1:len(r2)-1])
            multiple, r1 = pt1.split("(")
            if multiple == "":
                mtp = 1
            elif multiple == "-":
                mtp = -1
            else:
                mtp = Fraction(multiple)
            r1 = int(r1[:len(r1)-1])
            for i in range(self["rn"]):
                if i == r2-1:
                    new_row = []
                    for j in range(self["cn"]):
                        value = self[i, j] + mtp * self[r1-1, j]
                        new_row.append(value)
                    output.append(new_row)
                else:
                    output.append(self[i])
        else:
            multiple, r = transformation.split("(")
            if multiple == "-":
                mtp = -1
            else:
                mtp = Fraction(multiple)
            r = int(r[:len(r)-1])
            for i in range(self["rn"]):
                if i == r-1:
                    new_row = []
                    for j in range(self["cn"]):
                        new_row.append(self[i, j] * mtp)
                    output.append(new_row)
                else:
                    output.append(self[i])
        return Matrix(output)

    def ltrans(self, transformation: str):
        """transforms the columns in the matrix by selected transformation.

        Args:
            transformation: the transformation, (i, j), k(i) or k(i)+(j)
        """
        return self.transpose().rtrans(transformation).transpose()

    def transpose(self):
        """A ^ 'T'"""
        output_matrix = []
        for i in range(self["cn"]):
            new_row = []
            for j in range(self["rn"]):
                new_row.append(self[j, i])
            output_matrix.append(new_row)
        return Matrix(output_matrix)

    def trace(self):
        """tr(A)

        For square matrix, it's trace means the sum of its values in the
        main diagonal.

        Raises:
            MatrixError: if the matrix isn't a square matrix."""
        if self["rn"] == self["cn"]:
            output = 0
            for i in range(self["rn"]):
                output += self[i, i]
            return output
        else:
            raise MatrixError("only square matrix has trace")

    def diagonal(self):
        output = []
        for i in range(self["rn"]):
            output.append(self[i, i])
        return output

    def upper_triangular(self, show_exchange_time: bool = False):
        """converts the matrix to an upper triangular matrix.

        Use Gauss-Jordan method to convert the matrix to an upper triangular
        matrix, but in order to keep its determinant, the rows won't have a
        leading 1.

        Args:
             show_exchange_time: whether you want to see how many times the
                rows have been exchanged.

        Return:
             an upper triangular matrix. If show_exchange_time is true, return
                the exchanging time at the same time.
        """
        max_time = min(self["rn"], self["cn"])
        change_time, rdc, rdr = 0, 0, 0  # rdc = reading column, rdr resp.
        output = self._rows
        while rdr < max_time and rdc < max_time:
            if output[rdr][rdc] == 0:
                for find in range(rdr+1, max_time):
                    if output[find][rdc] != 0:
                        change_time += 1
                        new_matrix = []
                        for i in range(self["rn"]):
                            if i == find:
                                new_matrix.append(output[rdr])
                            elif i == rdr:
                                new_matrix.append(output[find])
                            else:
                                new_matrix.append(output[i])
                        output = new_matrix
                        break
            if output[rdr][rdc] == 0:
                rdc += 1
                pass
            else:
                new_matrix = []
                for rw in range(self["rn"]):
                    if rw <= rdr:
                        new_matrix.append(output[rw])
                    else:
                        new_row = []
                        mtp = Fraction(output[rw][rdc], output[rdr][rdc])
                        for v in range(self["cn"]):
                            value = output[rw][v] - (mtp * output[rdr][v])
                            new_row.append(value)
                        new_matrix.append(new_row)
                output = new_matrix
                rdc += 1
                rdr += 1
                if rdc == self["cn"]:
                    break
        if show_exchange_time:
            return Matrix(output), change_time
        else:
            return Matrix(output)

    def rank(self):
        """calculates the rank of the matrix."""
        simplified_matrix = self.upper_triangular()["self"]
        rank_value = 0
        for i in simplified_matrix:
            if i.count(0) == self["cn"]:
                break
            else:
                rank_value += 1
        return rank_value

    def row_reduce(self):
        """converts the matrix to an upper triangular matrix.

        Use Gauss-Jordan method to convert the matrix to an upper triangular
        matrix, but in order to keep its determinant, the rows won't have a
        leading 1.

        Return:
             an upper triangular matrix. If show_exchange_time is true, return
                the exchanging time at the same time.
        """
        max_time = min(self["rn"], self["cn"])
        rdc, rdr = 0, 0  # rdc = reading column, rdr resp.
        output = self._rows
        while rdr < max_time and rdc < max_time:
            if output[rdr][rdc] == 0:
                for find in range(rdr+1, max_time):
                    if output[find][rdc] != 0:
                        new_matrix = []
                        for i in range(self["rn"]):
                            if i == find:
                                new_matrix.append(output[rdr])
                            elif i == rdr:
                                new_matrix.append(output[find])
                            else:
                                new_matrix.append(output[i])
                        output = new_matrix
                        break
            if output[rdr][rdc] == 0:
                rdc += 1
            else:
                new_matrix = []
                for rw in range(self["rn"]):
                    new_row = []
                    if rw == rdr:
                        mtp = Fraction(1, output[rdr][rdc])
                        for v in output[rw]:
                            new_row.append(v * mtp)
                        new_matrix.append(new_row)
                    else:
                        mtp = Fraction(output[rw][rdc], output[rdr][rdc])
                        for v in range(self["cn"]):
                            value = output[rw][v] - (mtp * output[rdr][v])
                            new_row.append(value)
                        new_matrix.append(new_row)
                output = new_matrix
                rdc += 1
                rdr += 1
                if rdc == self["cn"]:
                    break
        return Matrix(output)

    def det(self):
        """|A|

        Use upper_triangular to convert the matrix to an upper matrix,
        and calculate it's determinant by multiplying the values in the
        main diagonal.

        Raises:
            MatrixError: if the matrix isn't a square matrix.
        """
        if self["rn"] == self["cn"]:
            simplified_matrix, change_time = self.upper_triangular(True)
            det_value = 1
            for k in range(self["rn"]):
                det_value *= simplified_matrix[k, k]
            det_value = det_value * ((-1) ** change_time)
            return Fraction(det_value)
        else:
            raise MatrixError("only square matrix has determination")

    def cofactor(self, index_i: int, index_j: int):
        """gets the cofactor of a value in the matrix.

        the cofactor of a value is a matrix, but any value in the same
        row or the same column with it is missing.

        Args:
            index_i: the value's row number.
            index_j: the value's column number.

        Return:
            A matrix, the cofactor of the value.

        Raises:
            IndexValue: if the index is too large.
        """
        if index_i > self["rn"] or index_j > self["cn"]:
            raise IndexError("index out of range")
        else:
            output_matrix = []
            for i in range(self["rn"]):
                if i + 1 != index_i:
                    new_row = []
                    for j in range(self["cn"]):
                        if j + 1 != index_j:
                            new_row.append(self[i, j])
                    output_matrix.append(new_row)
            return Matrix(output_matrix)

    def adjunct(self, index_i: int, index_j: int):
        """gets the adjunct of a value in the matrix."""
        if self["rn"] == self["cn"]:
            coe = (-1) ** (index_i+index_j)
            return coe * self.cofactor(index_i, index_j).det()
        else:
            raise MatrixError("only square matrix has adjunct")

    def adjoint(self):
        """A ^ '*'"""
        if self["rn"] == self["cn"]:
            output_matrix = []
            for i in range(self["cn"]):
                new_row = []
                for j in range(self["rn"]):
                    new_row.append(self.adjunct(j+1, i+1))
                output_matrix.append(new_row)
            return Matrix(output_matrix)
        else:
            raise MatrixError("only square matrix has adjoint matrix")

    def inverse(self):
        """A ^ -1

        Raises:
            ZeroDivisionError: if the matrix is a singular matrix."""
        if self["rn"] != self["cn"]:
            raise MatrixError("only square matrix has inverse matrix")
        elif self.det() == 0:
            raise ZeroDivisionError("singular matrix has no inverse matrix")
        else:
            return Fraction(1, self.det()) * self.adjoint()

    def __add__(self, other):
        """A + B

        Here A + k was defined as A + kI"""
        if isinstance(other, Matrix):
            output_matrix = []
            if self["rn"] != other["rn"]:
                raise MatrixError("unmatched row number")
            elif self["cn"] != other["cn"]:
                raise MatrixError("unmatched line number")
            else:
                for i in range(self["rn"]):
                    row = []
                    for j in range(self["cn"]):
                        row.append(self[i, j] + other[i, j])
                    output_matrix.append(row)
            return Matrix(output_matrix)
        elif isinstance(other, (int, float, Fraction)):
            if self["rn"] == self["cn"]:
                return self + other * unit(self["rn"])
            else:
                raise MatrixError()
        else:
            return NotImplemented

    def __radd__(self, other):
        """kI + A"""
        if isinstance(other, (int, float, Fraction)):
            matrix_b = Matrix([[other] * self["cn"]] * self["rn"])
            return self + matrix_b
        else:
            return NotImplemented

    def __sub__(self, other):
        """A - B"""
        return self + (-1 * other)

    def __mul__(self, other):
        """A * B"""
        output_matrix = []
        if isinstance(other, Matrix):
            if self["cn"] != other["rn"]:
                raise MatrixError("unmatched row and line number")
            else:
                for i in range(self["rn"]):
                    row = []
                    for j in range(other["cn"]):
                        value = 0
                        for k in range(self["cn"]):
                            value += self[i, k] * other[k, j]
                        row.append(value)
                    output_matrix.append(row)
            return Matrix(output_matrix)
        elif isinstance(other, (int, float, Fraction)):
            times = Fraction(other)
            for i in range(self["rn"]):
                new_row = []
                for j in range(self["cn"]):
                    new_row.append(times * self[i, j])
                output_matrix.append(new_row)
            return Matrix(output_matrix)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """k * A"""
        output_matrix = []
        if isinstance(other, (int, float, Fraction)):
            times = Fraction(other)
            for i in range(self["rn"]):
                new_row = []
                for j in range(self["cn"]):
                    new_row.append(times * self[i, j])
                output_matrix.append(new_row)
            return Matrix(output_matrix)
        else:
            return NotImplemented

    def __pow__(self, power, modulo=None):
        """A ** n"""
        if isinstance(power, int):
            if self["rn"] == self["cn"]:
                if power == 0:
                    return unit(self["rn"])
                elif power == 1:
                    return self
                elif power == -1:
                    return self.inverse()
                elif power > 0:
                    output = self * self
                    for i in range(1, power-1):
                        output = output * self
                    return output
                else:
                    return pow(self.inverse(), -power)
            else:
                raise MatrixError("the matrix should be a square matrix")
        elif power == "T":
            return self.transpose()
        elif power == "*":
            return self.adjoint()
        else:
            return NotImplemented

    def __eq__(self, other):
        """A == B"""
        if isinstance(other, Matrix):
            return self["self"] == other["self"]
        else:
            return NotImplemented

    def similar(self, transition_matrix: 'Matrix'):
        """calculates similar matrix via selected transition matrix.

        B = P^(-1) * A * P

        Args:
            transition_matrix: an invertible square matrix has the same scale
                with the original matrix.
        Return:
            a matrix which is similar to the original matrix.

        Raises:
            MatrixError: if the transition matrix isn't a square matrix, or
                it has different scale with the original matrix.
            ZeroDivisionError: if the transition matrix is singular."""
        try:
            matrix_b = transition_matrix ** (-1) * self * transition_matrix
        except ZeroDivisionError:
            raise ZeroDivisionError("the transition matrix should be an \
invertible matrix")
        except MatrixError:
            raise
        else:
            return matrix_b

    def congruent(self, transition):
        if isinstance(transition, Matrix):
            return transition.transpose() * self * transition
        elif isinstance(transition, str):
            return self.rtrans(transition).ltrans(transition)
        else:
            raise TypeError("'%s' can't be interrupted as a transition method."
                            % type(transition).__name__)

    def is_positive(self):
        diag = self.upper_triangular().diagonal()
        for i in diag:
            if i < 0:
                return False
        return True

    # todo eigen_polynomial and eigen_value(need to solve function)
    '''
    def eigen_polynomial(self):
        if self["cn"] == self["rn"]:
            output = diagonal_matrix([Polynomial([0, 1])] * self["rn"]) - self
            return LambdaMatrix(output._rows).det()
        else:
            raise MatrixError("Square")
            '''


r'''
class LambdaMatrix(Matrix):
    def __init__(self, rows: list[list[int | float | Fraction | Polynomial]]):
        super().__init__(rows)

    def __repr__(self):
        return "LambdaMatrix(" + repr(self._rows) + ")"

    def cofactor(self, index_i: int, index_j: int):
        """gets the cofactor of a value in the matrix.

        the cofactor of a value is a matrix, but any value in the same
        row or the same column with it is missing.

        Args:
            index_i: the value's row number.
            index_j: the value's column number.

        Return:
            A matrix, the cofactor of the value.

        Raises:
            IndexValue: if the index is too large.
        """
        if index_i > self["rn"] or index_j > self["cn"]:
            raise IndexError("index out of range")
        else:
            output_matrix = []
            for i in range(self["rn"]):
                if i != index_i:
                    new_row = []
                    for j in range(self["cn"]):
                        if j != index_j:
                            new_row.append(self[i, j])
                    output_matrix.append(new_row)
            return LambdaMatrix(output_matrix)

    def det(self):
        r"""|A|

        Calculate the matrix's det value via cofactor.

        |A| = \sum_(i=1)^n▒(a_1i \bullet A_1i)

        Raises:
            MatrixError: if the matrix isn't a square matrix."""
        if self["rn"] != self["cn"]:
            raise MatrixError("Square")
        if self["rn"] == 1:
            return self[0, 0]
        else:
            output = 0
            for i in range(self["rn"]):
                output += self[0, i] * ((-1) ** i) * self.cofactor(0, i).det()
            return output
'''


class MatrixError(ValueError):
    """a type of Error, raised in matrix calculation."""
    pass


class LaTeXDecodeError(ValueError):
    """a type of Error, raised in LaTeX code reading."""
    pass


def unicode_matrix(input_matrix: str):
    """formats a matrix to Matrix object.

    the matrix should be written in unicode-math format: use & to connect
    different value in a row and use @ to connect different row, example:

    | 1  2  3 |
    | 4  5  6 |
    | 7  8  9 | 3*3

    this matrix should be written as '1&2&3@4&5&6@7&8&9' or
    '■(1&2&3@4&5&6@7&8&9)'

    Args:
        input_matrix: the matrix in unicode-math format.

    Returns:
        a Matrix object.
    """
    if input_matrix[0] == "■":
        input_matrix = input_matrix[2:len(input_matrix)-1]
    writing_rows, row_lengths = [], []
    reading_rows = input_matrix.split("@")
    for row in reading_rows:
        writing_row = []
        reading_row = row.split("&")
        for value in reading_row:
            writing_row.append(Fraction(value))
        writing_rows.append(writing_row)
        row_lengths.append(len(writing_row))
    return Matrix(writing_rows)


def latex_matrix(input_matrix: str) -> Matrix:
    r"""formats a matrix to Matrix object.

    This matrix should be written in LaTeX format: announce matrix (or bmatrix,
    pmatrix, vmatrix, Vmatrix) environment at first, use & to connect different
    value in a row and use \\ to connect different row. Any blank in the
    code will be omitted, but be sure to avoid using Enter or Tab. Example:

    | 1  2  3 |
    | 4  5  6 |
    | 7  8  9 | 3*3

    this matrix should be written as '\begin{vmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{vmatrix}'

    :arg input_matrix: the matrix in LaTeX format

    :return: a Matrix object

    :raise LaTeXDecodeError: if the matrix is written in a wrong environment
    :raise MatrixError: if the matrix can't be decoded
    """
    input_matrix = input_matrix.replace(" ", "")
    if input_matrix[7] == "m":
        input_matrix = input_matrix[14:-12]
    elif input_matrix[7] in ["b", "p", "V", "v"]:
        input_matrix = input_matrix[15:-13]
    else:
        raise LaTeXDecodeError("Can't decode this LaTeX code as any matrix.")
    rows = input_matrix.split(r"\\")
    output = []
    for row in rows:
        values = row.split("&")
        new_row = []
        for value in values:
            try:
                new_row.append(int(value))
            except ValueError:
                if value.startswith(r"\frac"):
                    value = value[6:-1]
                    n, d = value.split("}{")
                    new_row.append(Fraction(int(n), int(d)))
                else:
                    new_row.append(Fraction(value))
        output.append(new_row)
    return Matrix(output)


def unit(scale: int):
    """constructs a unit matrix.

    Unit matrix is a square matrix, in which each value in the main diagonal
    is 1, while any other value is 0.

    Args:
        scale: the scale of the unit matrix.

    Return:
        a unit matrix.
    """
    new_matrix = []
    for i in range(scale):
        new_row = []
        for j in range(scale):
            if i == j:
                new_row.append(1)
            else:
                new_row.append(0)
        new_matrix.append(new_row)
    return Matrix(new_matrix)


def diagonal_matrix(diagonal: list[int | Fraction | float]):
    """constructs a matrix by its main diagonal elements."""
    output = []
    for i in range(len(diagonal)):
        new_row = []
        for j in range(len(diagonal)):
            if i == j:
                new_row.append(diagonal[i])
            else:
                new_row.append(0)
        output.append(new_row)
    return Matrix(output)


if __name__ == '__main__':
    latex_code = r'''\begin{matrix} 20 &    -5 & -5  & 0 & 0 \\ \frac{1}{2} & 5.2 & 0 & 1  & 10 \\ -5  & 0  & 1 0 &  -1 & -5 \\ 0 &  -1 & 1 & 0   & 2 \end{matrix}'''
    b = latex_matrix(latex_code)
    print(b)

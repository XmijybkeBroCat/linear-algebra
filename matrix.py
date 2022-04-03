# Originally contributed by Xmijybke_BroCat.

"""Class for matrix, matrix calculate."""

from fractions import Fraction

__all__ = ['Matrix', 'formatting', 'unit']


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
    |  7  -2   6 |
    """
    def __init__(self, rows):
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
                        raise SyntaxError("unfixed row length in the matrix")
                self._rows = rows
            else:
                raise SyntaxError("values should be in a nested list")
        else:
            raise SyntaxError("values should be in a nested list")

    def get_matrix(self):
        """gets the nested list."""
        return self._rows

    def get_row_number(self):
        """gets the number of rows in the matrix."""
        return len(self._rows)

    def get_line_number(self):
        """gets the number of column in the matrix."""
        return len(self._rows[0])

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
        if index_i <= self.get_row_number() and index_j <= self.get_line_number():
            return Fraction(self._rows[index_i-1][index_j-1])
        else:
            raise IndexError("index out of range")

    def locate(self, value: any):
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
        for i in range(self.get_row_number()):
            for j in range(self.get_line_number()):
                if self.get_value(i+1, j+1) == value:
                    position = (i+1, j+1)
                    return position
        if position is None:
            raise ValueError("found no '%s' in the matrix" % value)

    def unicode_math(self):
        """prints the matrix in unicode-math format."""
        output_rows = []
        for i in range(self.get_row_number()):
            row = []
            for j in range(self.get_line_number()):
                row.append(str(self.get_value(i+1, j+1)))
            output_row = "&".join(value for value in row)
            output_rows.append(output_row)
        output = "@".join(r for r in output_rows)
        output = "■(" + output + ")"
        return output

    def show_matrix(self, length: int = 4):
        """prints the matrix directly.

        This function shows the matrix in a usual format: each row will be
        written in a line and there is a tab between values in the row.
        You can set how long every value is in the argument 'length'

        Args:
            length: The length of every value, defaulted to 4.

        Return:
            a long string shows the matrix.
        """
        output_rows = []
        for i in range(self.get_row_number()):
            row = []
            for j in range(self.get_line_number()):
                row.append(str(self.get_value(i+1, j+1)))
            output_row = "\t".join(value.center(length) for value in row)
            output_rows.append(output_row)
        output = "\n".join(r for r in output_rows)
        return output

    def transpose(self):
        """gets the transpose of the matrix."""
        output_matrix = []
        for i in range(self.get_line_number()):
            new_row = []
            for j in range(self.get_row_number()):
                new_row.append(self.get_value(j+1, i+1))
            output_matrix.append(new_row)
        return Matrix(output_matrix)

    def trace(self):
        """gets the trace of the matrix.

        For square matrix, it's trace means the sum of its values in the
        main diagonal.

        Raises:
            ValueError: if the matrix isn't a square matrix."""
        if self.get_row_number() == self.get_line_number():
            output = 0
            for i in range(self.get_row_number()):
                output += self.get_value(i+1, i+1)
            return output
        else:
            raise ValueError("only square matrix has trace")

    def upper_triangular(self, show_change_time: bool = False):
        """converts the matrix to an upper triangular matrix.

        Use Gauss-Jordan method to convert the matrix to an upper triangular
        matrix, but in order to keep its determination, the rows won't have a
        leading 1.

        Args:
             show_change_time: whether you want to see how many times the
                rows have been exchanged.

        Return:
             an upper triangular matrix. If show_change_time is true, return
                the changing time at the same time.
        """
        max_time = min(self.get_row_number(), self.get_line_number())
        change_time, rdl = 0, 0  # rdl = reading line
        output_matrix = self.get_matrix()
        for check in range(max_time):
            if output_matrix[check][rdl] == 0:
                for find in range(check+1, max_time):
                    if output_matrix[find][rdl] != 0:
                        change_time += 1
                        new_matrix = []
                        for i in range(self.get_row_number()):
                            if i == find:
                                new_matrix.append(output_matrix[check])
                            elif i == check:
                                new_matrix.append(output_matrix[find])
                            else:
                                new_matrix.append(output_matrix[i])
                        output_matrix = new_matrix
                        break
            if output_matrix[check][rdl] == 0:
                continue
            else:
                new_matrix = []
                for rewrite in range(self.get_row_number()):
                    if rewrite <= check:
                        new_matrix.append(output_matrix[rewrite])
                    else:
                        new_row = []
                        multiple = Fraction(output_matrix[rewrite][rdl], output_matrix[check][rdl])
                        for value in range(self.get_line_number()):
                            new_value = output_matrix[rewrite][value] - (multiple * output_matrix[check][value])
                            new_row.append(new_value)
                        new_matrix.append(new_row)
                output_matrix = new_matrix
                rdl += 1
                if rdl == self.get_line_number():
                    break
        if show_change_time:
            return Matrix(output_matrix), change_time
        else:
            return Matrix(output_matrix)

    def rank(self):
        """calculates the rank of the matrix."""
        simplified_matrix = self.upper_triangular().get_matrix()
        rank_value = 0
        for i in simplified_matrix:
            if i.count(0) == self.get_line_number():
                break
            else:
                rank_value += 1
        return rank_value

    def det(self):
        """calculates the determination of the matrix.

        Use upper_triangular to convert the matrix to an upper matrix,
        and calculate its determination by multiplying the values in the
        main diagonal.

        Raises:
            ValueError: if the matrix isn't a square matrix.
        """
        if self.get_row_number() == self.get_line_number():
            simplified_matrix, change_time = self.upper_triangular(True)
            simplified_matrix = simplified_matrix.get_matrix()
            det_value = 1
            for k in range(self.get_row_number()):
                det_value *= simplified_matrix[k][k]
            det_value = det_value * ((-1) ** change_time)
            return Fraction(det_value)
        else:
            raise ValueError("only square matrix has determination")

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
        if index_i > self.get_row_number() or index_j > self.get_line_number():
            raise IndexError("index out of range")
        else:
            output_matrix = []
            for i in range(self.get_row_number()):
                if i + 1 != index_i:
                    new_row = []
                    for j in range(self.get_line_number()):
                        if j + 1 != index_j:
                            new_row.append(self.get_value(i + 1, j + 1))
                    output_matrix.append(new_row)
            return Matrix(output_matrix)

    def adjunct(self, index_i: int, index_j: int):
        """gets the adjunct of a value in the matrix."""
        if self.get_row_number() == self.get_line_number():
            return ((-1) ** (index_i+index_j)) * self.cofactor(index_i, index_j).det()
        else:
            raise ValueError("only square matrix has adjunct")

    def adjoint(self):
        """gets the adjoint matrix of the matrix."""
        if self.get_row_number() == self.get_line_number():
            output_matrix = []
            for i in range(self.get_line_number()):
                new_row = []
                for j in range(self.get_row_number()):
                    new_row.append(self.adjunct(j+1, i+1))
                output_matrix.append(new_row)
            return Matrix(output_matrix)
        else:
            raise ValueError("only square matrix has adjoint matrix")

    def inverse(self):
        """gets the inverse matrix of the matrix.

        Raises:
            ZeroDivisionError: if the matrix is a singular matrix."""
        if self.get_row_number() != self.get_line_number():
            raise ValueError("only square matrix has inverse matrix")
        elif self.det() == 0:
            raise ZeroDivisionError("singular matrix has no inverse matrix")
        else:
            return self.adjoint() * Fraction(1, self.det())

    def __add__(self, other):
        """a + b"""
        if isinstance(other, Matrix):
            output_matrix = []
            if self.get_row_number() != other.get_row_number():
                raise ValueError("unmatched row number")
            elif self.get_line_number() != other.get_line_number():
                raise ValueError("unmatched line number")
            else:
                for i in range(self.get_row_number()):
                    row = []
                    for j in range(self.get_line_number()):
                        row.append(self.get_value(i+1, j+1) + other.get_value(i+1, j+1))
                    output_matrix.append(row)
            return Matrix(output_matrix)
        else:
            type_b = type(other).__name__
            raise TypeError("unsupported operand type(s) for +: 'Matrix' and '%s'" % type_b)

    def __sub__(self, other):
        """a - b"""
        return self + (other * -1)

    def __mul__(self, other):
        """a * b

        If you want to multiply a matrix with a number, the number should be on
        the right of '*'
        """
        output_matrix = []
        if isinstance(other, (int, float, Fraction)):
            times = Fraction(other)
            for i in range(self.get_row_number()):
                new_row = []
                for j in range(self.get_line_number()):
                    new_row.append(times * self.get_value(i + 1, j + 1))
                output_matrix.append(new_row)
            return Matrix(output_matrix)
        elif isinstance(other, Matrix):
            if self.get_line_number() != other.get_row_number():
                raise ValueError("unmatched row and line number")
            else:
                for i in range(self.get_row_number()):
                    row = []
                    for j in range(other.get_line_number()):
                        value = 0
                        for k in range(self.get_line_number()):
                            value += self.get_value(i + 1, k + 1) * other.get_value(k + 1, j + 1)
                        row.append(value)
                    output_matrix.append(row)
            return Matrix(output_matrix)
        else:
            type_b = type(other).__name__
            raise TypeError("unsupported operand type(s) for *: 'Matrix' and '%s'" % type_b)

    def __pow__(self, power, modulo=None):
        """a ** b"""
        if self.get_row_number() != self.get_line_number():
            raise ValueError("the matrix should be a square matrix")
        else:
            if isinstance(power, int):
                if power == 0:
                    return unit(self.get_row_number())
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
                type_b = type(power).__name__
                raise TypeError("unsupported operand type(s) for ** or pow(): 'Matrix' and '%s'" % type_b)

    def __eq__(self, other):
        """a == b"""
        if self.get_matrix() == other.get_matrix():
            return True
        else:
            return False


def formatting(input_matrix: str):
    """formats a matrix to Matrix object.

    the matrix should be written in unicode-math format: use & to connect
    different value in a row and use @ to connect different row, like this:

    | 1  2  3 |
    | 4  5  6 |
    | 7  8  9 |

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
            if "/" in value:
                writing_row.append(Fraction(value))
            elif "." in value:
                writing_row.append(Fraction(float(value)))
            else:
                writing_row.append(int(value))
        writing_rows.append(writing_row)
        row_lengths.append(len(writing_row))
    return Matrix(writing_rows)


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

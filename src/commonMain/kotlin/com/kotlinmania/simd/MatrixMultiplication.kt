// port-lint: source gen.rs
package com.kotlinmania.simd

/**
 * SIMD AlphaTensor Matrix Multiplication Algorithms
 *
 * Matrix multiplication algorithms discovered by AlphaTensor performing multiplication
 * with fewer operations than traditional methods.
 */

internal class Int32x16(
    private val data: IntArray,
) {
    init {
        require(data.size == 16) { "Int32x16 must have exactly 16 elements" }
    }

    operator fun get(index: Int): Int = data[index]

    operator fun times(other: Int32x16): Int32x16 {
        val result = IntArray(16)
        for (i in 0..15) {
            result[i] = data[i] * other.data[i]
        }
        return Int32x16(result)
    }

    companion object {
        fun from(elements: IntArray): Int32x16 {
            require(elements.size == 16) { "Must provide exactly 16 elements" }
            return Int32x16(elements.copyOf())
        }
    }
}

fun multiply2By2MatrixAWith2By2MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 4) { "Matrix B must have 4 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b21 = b[2]
    val b22 = b[3]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (a21 - a22),
                    (a11 + a21 - a22),
                    (a11 - a12 + a21 - a22),
                    a12,
                    (a11 + a21),
                    a11,
                    a22,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    b12,
                    (b12 + b21 + b22),
                    (b21 + b22),
                    b21,
                    (b11 + b12 + b21 + b22),
                    b11,
                    (b12 + b22),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0])

    val c11 = (hs[0][3] + hs[0][5])
    val c12 = (-hs[0][1] + hs[0][4] - hs[0][5] - hs[0][6])
    val c21 = (-hs[0][0] + hs[0][1] - hs[0][2] - hs[0][3])
    val c22 = (hs[0][0] + hs[0][6])

    return intArrayOf(c11, c12, c21, c22)
}

fun multiply2By2MatrixAWith2By3MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 6) { "Matrix B must have 6 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b21 = b[3]
    val b22 = b[4]
    val b23 = b[5]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a21,
                    (a21 + a22),
                    (a12 + a21),
                    a12,
                    (a11 + a12),
                    (a11 - a21),
                    a11,
                    (a12 - a22),
                    (a21 - a22),
                    a12,
                    a22,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 - b22),
                    b22,
                    (b13 + b22),
                    (b13 - b23),
                    b13,
                    (b12 + b13),
                    (b11 - b13),
                    (b22 + b23),
                    b11,
                    (b13 - b21),
                    (b11 + b21),
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0])

    val c11 = (hs[0][4] + hs[0][6] - hs[0][9])
    val c12 = (hs[0][8] + hs[0][10])
    val c13 = (hs[0][0] + hs[0][2] - hs[0][4] + hs[0][5])
    val c21 = (hs[0][0] + hs[0][1])
    val c22 = (-hs[0][3] + hs[0][4])
    val c23 = (-hs[0][1] + hs[0][2] - hs[0][3] - hs[0][7])

    return intArrayOf(c11, c12, c13, c21, c22, c23)
}

fun multiply2By2MatrixAWith2By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 8) { "Matrix B must have 8 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a11,
                    (a11 - a12),
                    (a11 + a22),
                    a22,
                    a11,
                    (a11 - a12),
                    (a11 - a12 - a22),
                    (a12 + a22),
                    (a11 - a12 + a21 - a22),
                    a22,
                    (a11 + a21),
                    (a21 - a22),
                    a21,
                    (a12 + a22),
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b14 + b24),
                    b24,
                    (b11 - b24),
                    (b11 + b21 - b22),
                    (b13 + b23),
                    b23,
                    (b12 + b13 + b23),
                    (b12 + b13 + b22 + b23),
                    (b12 + b13),
                    b22,
                    (b11 + b14),
                    b11,
                    b12,
                    (b12 + b13 + b21 + b22 + b23 + b24),
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0])

    val c11 = (hs[0][1] + hs[0][2] - hs[0][3] - hs[0][7] - hs[0][9] + hs[0][13])
    val c12 = (hs[0][3] + hs[0][9] + hs[0][11])
    val c13 = (-hs[0][4] + hs[0][6] + hs[0][7] - hs[0][9])
    val c14 = (hs[0][9] + hs[0][12])
    val c21 = (hs[0][4] - hs[0][5])
    val c22 = (hs[0][5] - hs[0][6] + hs[0][8] - hs[0][12])
    val c23 = (hs[0][0] - hs[0][1])
    val c24 = (-hs[0][0] - hs[0][2] + hs[0][10] - hs[0][11])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24)
}

fun multiply2By2MatrixAWith2By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 10) { "Matrix B must have 10 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a21,
                    (a21 + a22),
                    (a12 + a21),
                    (a12 - a22),
                    (a11 - a21),
                    a12,
                    (a11 + a12),
                    a21,
                    (a21 - a22),
                    (a12 + a21 - a22),
                    a12,
                    a12,
                    (a12 - a22),
                    (a11 - a12 - a21 + a22),
                    a11,
                    (a11 - a21),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a11,
                    (a11 - a12 - a21 + a22),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 - b22),
                    b22,
                    (b15 - b22),
                    (b22 - b25),
                    (b12 - b15),
                    (b15 - b25),
                    b15,
                    (b11 + b21),
                    b21,
                    (b13 - b21),
                    (b21 - b24),
                    (b23 + b24),
                    (b13 + b23),
                    b13,
                    (b13 + b14),
                    (b13 - b14 + b23 - b24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b11 - b14),
                    (b13 + b23 - b24),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (-hs[0][8] - hs[0][9] - hs[0][13] + hs[0][14] + hs[1][0])
    val c12 = (hs[0][7] - hs[0][8])
    val c13 = (hs[0][0] - hs[0][2] + hs[0][4] + hs[0][6])
    val c14 = (hs[0][0] + hs[0][1])
    val c15 = (hs[0][8] + hs[0][9] + hs[0][10] + hs[0][11] + hs[0][13])
    val c21 = (hs[0][8] + hs[0][9] + hs[0][10] + hs[0][11] - hs[0][12])
    val c22 = (-hs[0][8] - hs[0][9] - hs[0][10] - hs[0][13] + hs[0][14])
    val c23 = (-hs[0][8] - hs[0][9] - hs[0][10] - hs[0][12] - hs[0][13] + hs[0][14] + hs[0][15] - hs[1][1])
    val c24 = (-hs[0][5] + hs[0][6])
    val c25 = (hs[0][1] + hs[0][2] + hs[0][3] - hs[0][5])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25)
}

fun multiply2By2MatrixAWith2By6MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 12) { "Matrix B must have 12 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b21 = b[6]
    val b22 = b[7]
    val b23 = b[8]
    val b24 = b[9]
    val b25 = b[10]
    val b26 = b[11]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a21,
                    (a21 - a22),
                    (a12 + a21),
                    (a12 + a22),
                    (a11 + a21),
                    a12,
                    a22,
                    (a21 - a22),
                    (a11 + a22),
                    (a12 + a22),
                    a11,
                    a22,
                    (a11 + a21),
                    (a11 - a12),
                    (a21 + a22),
                    (a11 - a12),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a22),
                    (a11 - a21),
                    a11,
                    (a11 + a12),
                    (a12 - a22),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 + b22),
                    b22,
                    (b14 - b22),
                    (b22 + b24),
                    (b12 + b14),
                    (b14 + b24),
                    (b11 + b21),
                    b11,
                    (b11 + b26),
                    (b21 - b26),
                    (b16 + b26),
                    (b13 - b23),
                    (b11 - b16),
                    b14,
                    b13,
                    b26,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b13 + b25),
                    (b13 + b15),
                    (b15 - b25),
                    b25,
                    (b23 + b25),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (-hs[0][6] + hs[0][8] + hs[0][9] - hs[0][15])
    val c12 = (hs[0][6] + hs[0][7])
    val c13 = (-hs[0][0] - hs[0][2] + hs[0][4] - hs[0][13])
    val c14 = (hs[0][0] - hs[0][1])
    val c15 = (-hs[0][11] + hs[1][0] - hs[1][3] + hs[1][4])
    val c16 = (-hs[0][11] + hs[0][14])
    val c21 = (hs[0][5] + hs[0][13])
    val c22 = (hs[0][1] + hs[0][2] + hs[0][3] - hs[0][5])
    val c23 = (hs[1][2] + hs[1][3])
    val c24 = (-hs[0][14] + hs[1][0] - hs[1][1] + hs[1][2])
    val c25 = (hs[0][10] - hs[0][15])
    val c26 = (hs[0][7] + hs[0][8] - hs[0][10] - hs[0][12])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c21, c22, c23, c24, c25, c26)
}

fun multiply2By2MatrixAWith2By7MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 14) { "Matrix B must have 14 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b21 = b[7]
    val b22 = b[8]
    val b23 = b[9]
    val b24 = b[10]
    val b25 = b[11]
    val b26 = b[12]
    val b27 = b[13]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a11,
                    (a11 - a12),
                    (a11 - a22),
                    (a21 - a22),
                    (a12 - a22),
                    a22,
                    (a21 + a22),
                    (a11 + a22),
                    (a11 - a21),
                    a11,
                    a22,
                    (a11 + a22),
                    (a21 + a22),
                    a11,
                    (a11 + a12),
                    (a11 - a12),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a12 - a22),
                    (a21 - a22),
                    (a11 - a21),
                    a22,
                    (a12 - a22),
                    (a11 + a12),
                    a11,
                    a21,
                    (a11 - a21),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b11 + b21),
                    b21,
                    (b14 - b21),
                    b14,
                    (b21 + b24),
                    (b13 - b23),
                    b13,
                    (b13 - b27),
                    (b13 - b17),
                    (b17 - b27),
                    (b16 - b26),
                    (b16 - b25),
                    b16,
                    (b15 - b25),
                    b25,
                    b22,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b23 - b27),
                    (b16 + b22),
                    (b15 - b16),
                    (b14 + b24),
                    ((b23 shl 1) - b25 + b26 - (b27 shl 1)),
                    b27,
                    (b11 - b12 - (b17 shl 1) + b21 - b22 + (b27 shl 1)),
                    (b12 + (b16 shl 1) + b22),
                    (b11 + b14 - (b15 shl 1) + (b16 shl 1)),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (hs[0][0] - hs[0][1])
    val c12 = (hs[0][0] + hs[0][2] - hs[0][3] - (hs[1][2] shl 1) - hs[1][8])
    val c13 = (hs[0][0] - (hs[0][9] shl 1) - hs[0][15] - hs[1][6])
    val c14 = (-hs[0][12] - hs[1][1] + hs[1][7])
    val c15 = (-hs[0][5] + hs[0][7] + hs[1][0] + hs[1][5])
    val c16 = (-hs[0][5] + hs[0][6])
    val c17 = (hs[0][1] + hs[0][2] + hs[0][4] + hs[1][3])
    val c21 = (hs[0][3] + hs[1][3])
    val c22 = (hs[0][13] + hs[0][14])
    val c23 = (-hs[0][11] + hs[0][12] + hs[0][13] - hs[1][2])
    val c24 = (-hs[0][10] + hs[0][11] + hs[0][14] - (hs[1][0] shl 1) + hs[1][4])
    val c25 = (-hs[0][10] + hs[0][12])
    val c26 = (hs[0][9] + hs[1][5])
    val c27 = (hs[0][6] - hs[0][7] + hs[0][8] + hs[0][9])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c21, c22, c23, c24, c25, c26, c27)
}

fun multiply2By2MatrixAWith2By8MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 16) { "Matrix B must have 16 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a21 = a[2]
    val a22 = a[3]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b18 = b[7]
    val b21 = b[8]
    val b22 = b[9]
    val b23 = b[10]
    val b24 = b[11]
    val b25 = b[12]
    val b26 = b[13]
    val b27 = b[14]
    val b28 = b[15]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a11,
                    (a11 + a12),
                    (a11 + a12 + a22),
                    a21,
                    a22,
                    a21,
                    (a21 + a22),
                    (a12 - a21 - a22),
                    (a11 + a12 - a21 - a22),
                    a11,
                    a12,
                    a12,
                    (a12 + a22),
                    (a11 + a12),
                    (a11 + a12 + a21),
                    (a11 + a12 + a21 + a22),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a22,
                    (a11 + a12 + a21 + a22),
                    a11,
                    (a11 + a12 - a22),
                    a21,
                    (a12 - a22),
                    (a11 + a12),
                    a22,
                    (a11 + a21),
                    (a12 - a22),
                    (a11 + a12 - a21 - a22),
                    a21,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 - b22),
                    b22,
                    (b12 + b14 - b22),
                    b14,
                    b24,
                    (b11 - b21),
                    b21,
                    (b11 - b16 - b21),
                    (b11 - b16),
                    b16,
                    b26,
                    (b17 - b27),
                    (b12 + b14 - b22 - b24),
                    b17,
                    (b17 - b27 - b28),
                    (b12 + b14),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b23 - b28),
                    (b12 + b14 - b27 - b28),
                    (b13 + b15 - (b16 shl 1) - b23 - b25),
                    (b15 - b23 - b25),
                    b18,
                    (b15 - b25),
                    (b23 + b25),
                    b23,
                    (b17 + b18 - b27 - b28),
                    (b11 + (b15 shl 1) - b16 - b21 - (b25 shl 1) + b26),
                    b15,
                    (b13 + b14),
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (hs[0][6] - hs[0][7] + hs[0][8] + hs[0][9])
    val c12 = (hs[0][5] + hs[0][6])
    val c13 = (hs[0][0] + hs[0][1])
    val c14 = (-hs[0][1] - hs[0][2] - hs[0][3] + hs[0][15])
    val c15 = ((hs[0][9] shl 1) + hs[1][2] - hs[1][3] + hs[1][5] + hs[1][7])
    val c16 = (-hs[0][3] + hs[1][7] + hs[1][11])
    val c17 = (-hs[0][0] + hs[0][2] - hs[0][4] - hs[0][12])
    val c18 = (hs[0][3] + hs[0][4])
    val c21 = (hs[1][3] - hs[1][5] + hs[1][6] - hs[1][7])
    val c22 = (hs[1][3] + hs[1][6] - hs[1][7] - hs[1][10])
    val c23 = (hs[0][9] + hs[0][10])
    val c24 = (hs[0][5] + hs[0][7] + hs[0][10] + (hs[1][5] shl 1) - hs[1][9])
    val c25 = (-hs[0][11] + hs[0][13])
    val c26 = (-hs[0][13] + hs[0][14] + hs[0][15] + hs[1][0] - hs[1][1] - hs[1][7])
    val c27 = (hs[0][11] - hs[0][14] - hs[1][4] + hs[1][8])
    val c28 = (-hs[1][0] + hs[1][4] + hs[1][7])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c18, c21, c22, c23, c24, c25, c26, c27, c28)
}

fun multiply2By3MatrixAWith3By3MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 6) { "Matrix A must have 6 elements" }
    require(b.size == 9) { "Matrix B must have 9 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b21 = b[3]
    val b22 = b[4]
    val b23 = b[5]
    val b31 = b[6]
    val b32 = b[7]
    val b33 = b[8]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a23,
                    (a11 - a23),
                    a11,
                    (a11 - a13),
                    (a13 - a23),
                    a12,
                    (a12 - a13 + a23),
                    (a11 - a21 + a22),
                    (a21 - a22),
                    (a12 - a13 - a22 + a23),
                    (a12 - a13),
                    a22,
                    (a11 - a12 - a21 + a22),
                    (a21 - a23),
                    (a11 - a21),
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b11 + b31 + b32),
                    (b11 + b33),
                    (b12 - b13 - b33),
                    b33,
                    (b21 + b31 - b33),
                    (b22 - b23 + b32),
                    (b21 - b32),
                    (b12 + b23),
                    b12,
                    b21,
                    b32,
                    (b12 + b21 + b22),
                    b23,
                    b11,
                    (b11 - b13 - b23),
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0])

    val c11 = (hs[0][0] + hs[0][1] - hs[0][3] + hs[0][4] + hs[0][6] + hs[0][10])
    val c12 = (hs[0][0] + hs[0][6] - hs[0][9] + hs[0][10] + hs[0][13])
    val c13 = (hs[0][5] + hs[0][7] + hs[0][8] - hs[0][10] - hs[0][12])
    val c21 = (-hs[0][6] + hs[0][8] + hs[0][9] - hs[0][10] + hs[0][11])
    val c22 = (-hs[0][2] - hs[0][3] + hs[0][7] + hs[0][8] - hs[0][12])
    val c23 = (-hs[0][1] - hs[0][2] + hs[0][7] + hs[0][8] + hs[0][13] + hs[0][14])

    return intArrayOf(c11, c12, c13, c21, c22, c23)
}

fun multiply2By3MatrixAWith3By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 6) { "Matrix A must have 6 elements" }
    require(b.size == 12) { "Matrix B must have 12 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]
    val b31 = b[8]
    val b32 = b[9]
    val b33 = b[10]
    val b34 = b[11]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a12,
                    (a12 - a13),
                    (a12 - a13 + a23),
                    (a12 - a13 - a22 + a23),
                    a22,
                    (a12 - a13 - a21 - a22),
                    (a13 - a23),
                    (a11 + a23),
                    (a11 + a13),
                    (a21 + a22),
                    (a21 + a23),
                    a23,
                    (a11 - a21 - a22),
                    (a11 + a12 - a21 - a22),
                    a12,
                    a22,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a11,
                    (a11 + a13),
                    (a21 + a23),
                    (a11 - a21),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 - b22 - b32),
                    (b12 - b32),
                    (b12 - b22 - b24 - b32),
                    (b12 - b22 - b24 + b31),
                    b24,
                    (b12 + b31),
                    (b12 - b22 - b24 - b32 - b33 - b34),
                    (b14 + b33),
                    (b12 - b33),
                    (b11 - b31),
                    (b14 + b31),
                    (b14 - b34),
                    (b11 - b21 - b23 - b31),
                    (b12 + b21 + b23 + b31),
                    b23,
                    (b11 - b21 - b31),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b13 - b33),
                    b12,
                    b31,
                    (b11 + b13 + b14 - b21 - b23 - b31),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (-hs[0][5] + hs[0][9] + hs[0][12] + hs[0][13] - hs[0][14] - hs[1][1])
    val c12 = (hs[0][9] - hs[0][15] + hs[1][2])
    val c13 = (-hs[0][0] + hs[0][1] + hs[1][1])
    val c14 = (hs[0][1] - hs[0][2] + hs[0][3] - hs[0][4] - hs[0][5] - hs[1][2])
    val c21 = (-hs[0][8] + hs[0][14] + hs[1][0] + hs[1][1])
    val c22 = (hs[0][7] - hs[0][10] + hs[0][12] + hs[0][15] + hs[1][0] + hs[1][2] - hs[1][3])
    val c23 = (hs[0][0] - hs[0][2] - hs[0][6] + hs[0][7] + hs[0][8] - hs[0][11] - hs[1][1])
    val c24 = (hs[0][4] + hs[0][10] - hs[0][11] - hs[1][2])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24)
}

fun multiply2By3MatrixAWith3By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 6) { "Matrix A must have 6 elements" }
    require(b.size == 15) { "Matrix B must have 15 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a23,
                    (a22 - a23),
                    (a12 + a22 - a23),
                    a12,
                    (a12 + a22),
                    (a12 + a21 + a22),
                    (a21 + a22),
                    (a11 + a12 + a21 + a22),
                    a21,
                    a13,
                    (a12 - a13 + a22 - a23),
                    (a11 - a12 + a13 - a22 + a23),
                    (a11 - a12 + a13 + a21 - a22 + a23),
                    (a11 - a12 + a13 + a21 + a23),
                    (a11 + a13 + a21 + a23),
                    (a13 + a21 + a23),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a13 + a23),
                    a22,
                    a11,
                    a21,
                    a12,
                    (a21 + a23),
                    (a11 - a12 + a13),
                    a11,
                    a13,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b22 + b32),
                    b22,
                    (b22 + b31 + b32),
                    (b21 - b24),
                    (b11 - b21 - b22 - b31 - b32),
                    (b11 - b24),
                    b24,
                    b11,
                    b12,
                    b31,
                    (b12 + b13 - b31 - b32 - b33),
                    (b12 + b13 - b33),
                    (b12 + b13 - b23 - b25 - b33),
                    (b23 + b25 + b33),
                    (b15 + b23 + b25 + b33),
                    (b15 + b34 - b35),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b15 - b35),
                    b25,
                    (b13 - b33),
                    (b14 - b15 - b24 - b34 + b35),
                    (b23 + b33),
                    (b34 - b35),
                    b33,
                    (b11 - b13 - b14 + b33),
                    b34,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (hs[0][3] - hs[0][5] - hs[0][6] + hs[0][7] + hs[0][9])
    val c12 = (-hs[0][0] - hs[0][2] - hs[0][3] - hs[0][4] + hs[0][5] + hs[0][6])
    val c13 = (-hs[0][1] + hs[0][2] - hs[0][9] + hs[0][10] + hs[0][11] - hs[1][2])
    val c14 = (hs[0][0] + hs[0][1] + hs[0][8])
    val c15 = (hs[1][2] + hs[1][4] + hs[1][6])
    val c21 = (-hs[0][8] - hs[0][11] + hs[0][12] + hs[0][13] - hs[1][1] - hs[1][6])
    val c22 = (-hs[0][5] - hs[0][6] + hs[0][7] - hs[1][2] - hs[1][7] + hs[1][8])
    val c23 = (hs[0][6] + hs[0][15] - hs[1][0] + hs[1][3] - hs[1][8])
    val c24 = (-hs[0][13] + hs[0][14] - hs[0][15] - hs[1][4] + hs[1][5] + hs[1][8])
    val c25 = (hs[0][15] - hs[1][0] + hs[1][1] - hs[1][5] - hs[1][8])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25)
}

fun multiply2By4MatrixAWith4By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 8) { "Matrix A must have 8 elements" }
    require(b.size == 16) { "Matrix B must have 16 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]
    val b31 = b[8]
    val b32 = b[9]
    val b33 = b[10]
    val b34 = b[11]
    val b41 = b[12]
    val b42 = b[13]
    val b43 = b[14]
    val b44 = b[15]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a13,
                    (a12 - a13),
                    (a13 - a22),
                    (a13 - a23),
                    (a13 - a23 - a24),
                    (a23 + a24),
                    (a22 - a23),
                    (a21 + a23 + a24),
                    (a11 + a23 + a24),
                    a22,
                    (a11 - a22),
                    (a12 - a22),
                    (a12 - a14 - a22),
                    (a12 - a14 - a22 + a24),
                    (a12 - a14),
                    (a11 - a12),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a21 - a22),
                    (a13 + a14 - a23 - a24),
                    a11,
                    (a11 + a13 + a14),
                    (a11 - a21),
                    (a11 - a21 + a24),
                    (a21 - a24),
                    (a11 - a14 - a21 + a24),
                    a24,
                    a14,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b24 - b31 + b34),
                    b24,
                    (b24 + b32),
                    (b32 - b34 + b44),
                    (b31 - b44),
                    (b14 - b31),
                    b32,
                    b14,
                    (b14 + b33),
                    (b12 + b21 + b22 + b32),
                    (b12 - b23),
                    (b22 + b23 - b24 + b42),
                    (b21 - b42),
                    b42,
                    b21,
                    b23,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b12,
                    (b33 + b44),
                    (b11 - b13 - b14 - b23 + b33),
                    b33,
                    (b12 + b13 + b14 - b33 + b43),
                    (b11 - b14 - b33 + b43),
                    (b11 - b14),
                    (b33 - b43),
                    (b11 - b14 - b31 + b41 + b42),
                    (b21 + b33 + b41 - b43 - b44),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (hs[0][4] - hs[0][5] + hs[0][8] + hs[0][14] + hs[1][1] - hs[1][3] + hs[1][5] + hs[1][6] + hs[1][7] + hs[1][9])
    val c12 = (-hs[0][5] + hs[0][7] - hs[0][12] - hs[0][13] + hs[0][14] + hs[1][6] + hs[1][8])
    val c13 = (hs[0][1] + hs[0][2] + hs[0][9] + hs[0][10] + hs[0][11] + hs[0][12] - hs[0][14] + hs[0][15])
    val c14 = (-hs[0][6] + hs[0][9] + hs[0][12] + hs[0][13] - hs[0][14] + hs[1][0])
    val c21 = (-hs[0][15] - hs[1][2] + hs[1][3] + hs[1][5] + hs[1][6] + hs[1][7])
    val c22 = (-hs[0][7] + hs[0][8] + hs[0][10] - hs[1][0] - hs[1][2] - hs[1][4] + hs[1][5] + hs[1][6])
    val c23 = (hs[0][0] + hs[0][1] + hs[0][4] - hs[0][5] + hs[0][8] + hs[1][1] - hs[1][3])
    val c24 = (hs[0][0] - hs[0][2] + hs[0][3] + hs[0][4] - hs[0][5] - hs[0][6] + hs[0][7])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24)
}

fun multiply2By4MatrixAWith4By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 8) { "Matrix A must have 8 elements" }
    require(b.size == 20) { "Matrix B must have 20 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    -a21,
                    (a13 - a21 + a23),
                    (-a11 + a13 - a21 + a23),
                    -a11,
                    (-a21 + a23),
                    (-a11 + a13 - a14 - a21 + a23),
                    (-a11 + a13 - a14 - a21 + a23 - a24),
                    (-a11 + a13 - a14),
                    (-a11 + a13 - a14 - a21 + a22 + a23 - a24),
                    (-a21 + a22 + a23 - a24),
                    (-a11 + a12 + a13 - a14 - a21 + a22 + a23 - a24),
                    a22,
                    (-a13 + a22 - a23),
                    (-a12 + a13 - a22 + a23),
                    (a22 - a23),
                    (-a12 + a13 + a14 - a22 + a23),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a11 - a12 + a13 - a22 + a23),
                    (-a11 - a12 + a13),
                    (-a12 + a13 + a14),
                    (-a12 + a13 + a14 - a22 + a23 + a24),
                    (-a13 - a23),
                    -a13,
                    (-a13 - a23 + a24),
                    a24,
                    -a11,
                    (a23 - a24),
                    (-a11 - a12 + a13 - a21 - a22 + a23),
                    -a12,
                    (-a21 - a22),
                    -a14,
                    (-a13 + a14 - a23 + a24),
                    a22,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-b11 - b31),
                    (b11 - b15 + b31),
                    (b11 - b15 - b41),
                    -b15,
                    (-b13 + b31),
                    (-b13 - b41),
                    (b21 + b41),
                    (-b13 + b44),
                    (b21 - b44),
                    b44,
                    b21,
                    (b22 + b32),
                    (b22 - b25 + b32),
                    (-b12 + b22 - b25 + b42),
                    (b21 + b23 - b32 - b34 - b44),
                    (b21 + b23 - b42),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b12 + b34 + b44),
                    (b34 + b44),
                    (b21 + b23),
                    b42,
                    (b11 - b15 + b22 - b25 + b31 + b32 - b35 - b45),
                    (-b13 - b21 - b23 - b33 + b35),
                    (-b13 - b21 - b23 - b33 - b45),
                    (-b13 - b21 - b23 - b33 - b41 - b42 - b43),
                    (-b14 - b34),
                    (-b13 - b21 - b23 - b33),
                    -b12,
                    (b21 + b24 + b34 + b44),
                    (-b12 - b14 + b44),
                    (-b13 + b21 + b23 + b43 - b45),
                    -b45,
                    (-b12 - b14 + b24 + b34 + ((b44 shl 1) + b44)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b25,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2])

    val c11 = (hs[0][1] - hs[0][2] + hs[0][3] - hs[0][4] + hs[0][5] - hs[0][7] - hs[0][8] - hs[0][9] + hs[0][10])
    val c12 = (hs[0][0] + hs[0][4] - hs[0][5] - hs[0][6] + hs[0][7] + hs[0][8] + hs[0][9])
    val c13 = (-hs[0][12] - hs[0][13] - hs[0][14] - hs[0][15] + hs[1][0] - hs[1][1] + hs[1][2] - hs[2][0])
    val c14 = (hs[0][11] + hs[0][14] + hs[0][15] - hs[1][0] + hs[1][1] - hs[1][2] + hs[1][3] + hs[1][10])
    val c15 = (hs[0][7] + hs[0][8] + hs[0][9] - hs[0][10] - hs[1][2] + hs[1][6] + hs[1][9] - hs[1][13] - hs[1][14])
    val c21 = (hs[0][5] + hs[0][6] - hs[0][7] - hs[0][8] - hs[0][9] - hs[0][15] + hs[1][2] - hs[1][3] - hs[1][7] - hs[1][9])
    val c22 = (hs[0][8] + hs[0][9] - hs[0][10] + hs[1][1] + hs[1][8] - hs[1][11])
    val c23 = (-hs[0][9] + hs[1][0] - hs[1][1] - hs[1][10] + hs[1][12] + hs[1][15])
    val c24 = (hs[0][3] - hs[1][5] + hs[1][6] + hs[1][9] - hs[1][14] - hs[2][0])
    val c25 = (hs[0][0] + hs[0][1] + hs[0][11] - hs[0][12] + hs[1][4] + hs[1][5] - hs[1][6] - hs[1][9])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25)
}

fun multiply2By5MatrixAWith5By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 10) { "Matrix A must have 10 elements" }
    require(b.size == 25) { "Matrix B must have 25 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a21 = a[5]
    val a22 = a[6]
    val a23 = a[7]
    val a24 = a[8]
    val a25 = a[9]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]
    val b51 = b[20]
    val b52 = b[21]
    val b53 = b[22]
    val b54 = b[23]
    val b55 = b[24]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a21,
                    (a12 + a21 - a22),
                    (a11 - a12 - a21 + a22),
                    (a11 - a12 - a14 - a21 + a22),
                    (a11 - a12 + a15 - a21 + a22),
                    (a12 - a22 - a23),
                    a11,
                    (a21 - a22),
                    (a11 - a12 + a15),
                    (a11 - a12 + a15 - a21 + a22 - a25),
                    (a11 - a12 - a13 - a21 + a22 + a23),
                    (a22 + a23),
                    (a12 + a13 + a14 - a22 - a23),
                    (a12 + a13 - a15 - a22 - a23 + a25),
                    (a12 - a22 + a24),
                    (a22 - a24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a24,
                    (a12 - a14 - a22 + a24),
                    (a12 - a14 + a15 - a22 + a24),
                    (a12 + a13 + a14),
                    (a12 - a14 + a15),
                    (a12 + a13 - a15 - a22 - a23),
                    (a12 - a22 - a25),
                    (a12 + a13 - a15),
                    (a12 - a14 + a15 - a22 + a24 - a25),
                    a14,
                    a15,
                    (a12 + a15 - a22 - a25),
                    (a11 - a12 - a13 - a21 + a22),
                    a12,
                    (a12 + a13 - a22 - a23),
                    (a11 - a12 - a13),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a22 + a25),
                    (a12 - a22),
                    a23,
                    (a11 - a12 - a14 - a21 + a22 + a24),
                    a25,
                    (a12 + a13 + a14 - a22 - a23 - a24),
                    (a11 - a12 - a14),
                    a13,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 + b22),
                    (b11 - b12 - b22),
                    (b11 - b12 - b32 - b42 + b52),
                    (b13 + b42),
                    (b15 - b52),
                    (b14 + b24 + b31),
                    b11,
                    (b13 + b14 + b15 + b22),
                    b15,
                    b52,
                    b32,
                    (b14 + b24 + b33 - b35),
                    (b33 + b44),
                    b54,
                    (b13 + b23 - b33 - b41 - b44),
                    (b13 + b23 - b33 + b45),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b13 + b23 - b33 - b42 + b43 + b44),
                    (b41 + b44 - b53),
                    (b45 + b53),
                    b33,
                    b45,
                    (b35 + b54),
                    (b15 + b25 - b35 + b45 - b51 - b53 - b54),
                    b35,
                    b53,
                    (b13 - b33 + b41 + b43 + b44),
                    (b15 - b35 - b45 + b51 + b53 + b54 - b55),
                    (b51 + b53 + b54),
                    (b14 + b32),
                    (b13 + b14 - b15 + b21 + b23 + b24 - b25 - b33 + b35 - b45),
                    (b31 + b44 - b54),
                    b14,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b15 + b25 - b35 + b45),
                    (b11 - b12 + b21 - b22 - b31 + b41 + b44 - b51 - b53 - b54),
                    (b14 + b24 + b32 - b34),
                    b42,
                    (b15 + b25 - b35 + b45 + b52 + b53 + b54 - b55),
                    b44,
                    b13,
                    (b14 + b31 + b34),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2])

    val c11 = (-hs[0][5] + hs[0][6] - hs[0][11] - hs[0][12] - hs[0][14] - hs[0][15] - hs[1][1] - hs[1][2] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][6] - hs[1][7] + hs[1][11] + hs[1][13] + hs[1][14] + hs[2][0])
    val c12 = (hs[0][0] + hs[0][1] - hs[0][5] - hs[0][11] - hs[0][12] - hs[0][13] - hs[0][14] - hs[0][15] - hs[1][2] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][6] - hs[1][7] + hs[1][8] + hs[1][13] + hs[2][0] - hs[2][1] + hs[2][5])
    val c13 = (-hs[0][1] - hs[0][2] - hs[0][3] - hs[0][4] + hs[0][6] - hs[0][7] + hs[0][8] - hs[1][12] + hs[1][15] + hs[2][6])
    val c14 = (hs[0][0] - hs[0][3] - hs[0][4] - hs[0][7] + hs[0][8] - hs[0][9] + hs[0][10] - hs[1][12] + hs[1][15] + hs[2][3] + hs[2][6])
    val c15 = (hs[0][14] + hs[0][15] + hs[1][1] + hs[1][2] + hs[1][3] - hs[1][4] + hs[1][9] + hs[2][6])
    val c21 = (-hs[0][3] - hs[0][12] + hs[0][15] + hs[1][0] + hs[1][2] + hs[1][3] - hs[1][4] - hs[1][8] + hs[2][3] + hs[2][5] + hs[2][6])
    val c22 = (hs[0][5] + hs[0][11] + hs[0][12] - hs[1][3] - hs[1][5] + hs[1][7] - hs[1][14] + hs[1][15] + hs[2][7])
    val c23 = (hs[0][10] + hs[0][11] + hs[0][12] + hs[0][13] - hs[1][3] - hs[1][5] + hs[1][7] - hs[1][12] + hs[1][15] - hs[2][2] - hs[2][5])
    val c24 = (hs[0][8] - hs[1][4] + hs[1][6] + hs[1][7] - hs[1][10] + hs[1][11] + hs[2][0])
    val c25 = (-hs[0][4] + hs[0][8] - hs[0][9] + hs[0][13] + hs[1][2] - hs[1][4] - hs[1][5] + hs[1][7] - hs[1][8] + hs[2][0] - hs[2][4])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25)
}

fun multiply3By3MatrixAWith3By3MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 9) { "Matrix A must have 9 elements" }
    require(b.size == 9) { "Matrix B must have 9 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val a31 = a[6]
    val a32 = a[7]
    val a33 = a[8]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b21 = b[3]
    val b22 = b[4]
    val b23 = b[5]
    val b31 = b[6]
    val b32 = b[7]
    val b33 = b[8]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a32,
                    (a11 - a31 + a32),
                    (a22 - a23 - a32),
                    a12,
                    (a12 + a22 - a23),
                    (a22 - a23 - a32 + a33),
                    a33,
                    (a11 - a31 - a33),
                    (a12 - a13 + a22 - a23),
                    (a31 - a32),
                    (a31 + a33),
                    (a11 - a12 - a31 + a32),
                    (a11 + a13 - a31 - a33),
                    a11,
                    (a12 + a22),
                    (a22 - a23),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a21 + a22 - a31 - a32),
                    (a21 + (a31 shl 1) + (a33 shl 1)),
                    (a22 - a32),
                    (a11 - a31),
                    (a12 - a21 + a22),
                    a13,
                    (a21 + a23 + (a31 shl 1) + (a33 shl 1)),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b12 - b21 + b22),
                    (b12 - (b21 shl 1) + b22 + b23),
                    (b21 + b32),
                    (b21 - b23),
                    (b21 + b33),
                    b32,
                    (b11 - b31 + b32),
                    (b11 - b31 + b33),
                    b33,
                    b12,
                    b11,
                    ((b21 shl 1) - b22 - b23),
                    (b31 - b33),
                    b13,
                    (b13 + b23 + b33),
                    b21,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b12,
                    (b11 - b21 - b31),
                    (b12 - b22 - b32),
                    (b11 - b12 - b13 + (b21 shl 1) - b22 - b23 - b31 + b33),
                    b13,
                    (b32 - b33),
                    (b21 + b31),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (hs[0][4] + hs[0][7] - hs[0][8] + hs[0][10] + hs[0][12] - hs[0][15])
    val c12 = (-(hs[0][10] shl 1) + hs[0][15] + hs[1][1] + hs[1][6])
    val c13 = (-hs[0][2] + hs[0][5] - hs[0][6] + hs[0][10] + hs[0][15])
    val c21 = (hs[0][1] + hs[0][3] + hs[0][4] - hs[0][8] + hs[0][9] + hs[0][11] - hs[0][15] + hs[1][5])
    val c22 = (hs[0][0] - hs[0][2] + hs[0][9] + hs[0][15] + hs[1][0] - hs[1][2])
    val c23 = (hs[0][0] - hs[0][2] + hs[0][5] + hs[0][9] + hs[0][15])
    val c31 = (-hs[0][3] + hs[0][4] - hs[0][8] + hs[0][13] - hs[0][15])
    val c32 = (hs[0][3] - hs[0][4] + hs[0][14] + hs[0][15] - hs[1][4])
    val c33 = (-hs[0][0] + hs[0][1] - hs[0][2] + hs[0][5] - hs[0][6] - hs[0][7] + hs[0][13] + hs[0][15] + hs[1][3])

    return intArrayOf(c11, c12, c13, c21, c22, c23, c31, c32, c33)
}

fun multiply3By3MatrixAWith3By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 9) { "Matrix A must have 9 elements" }
    require(b.size == 12) { "Matrix B must have 12 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val a31 = a[6]
    val a32 = a[7]
    val a33 = a[8]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]
    val b31 = b[8]
    val b32 = b[9]
    val b33 = b[10]
    val b34 = b[11]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a23,
                    (a21 + a23),
                    (a23 + a31),
                    (a11 + a21 + a23),
                    (a11 + a13 + a21 + a23),
                    (a23 - a33),
                    (a31 + a33),
                    (a23 - a32 - a33),
                    a31,
                    (a22 + a23 - a32 - a33),
                    (a13 - a31 - a33),
                    (a11 + a13 - a31 - a33),
                    (a21 - a31),
                    (a21 - a22 - a31),
                    (a21 - a22),
                    (a13 - a33),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a21 - a22),
                    (a13 - a32 - a33),
                    a32,
                    a12,
                    (a12 + a13 - a32 - a33),
                    a22,
                    (a32 + a33),
                    a11,
                    (a11 + a21),
                    (a11 - a12 + a21 - a22),
                    (a21 - a22 - a31 + a32),
                    a13,
                    a12,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b11 - b31 - b34),
                    b11,
                    (b11 + b32),
                    (b11 - b31 - b33 - (b34 shl 1)),
                    (b31 + b33 + (b34 shl 1)),
                    (b21 - b31 - b32),
                    b32,
                    (b21 + b34),
                    (b12 + b14 - b32),
                    b21,
                    (b12 + b13 + b14 - b32),
                    (b12 + b13 + b14),
                    (b11 + b12 + b22),
                    (b14 - b22),
                    b14,
                    (b12 + b13 + b14 + b23 - b32 - b33),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b14 + b23 + b24),
                    (b23 + b34),
                    (b22 + b24 - b34),
                    (b22 + b23 + b24),
                    b23,
                    (b14 + b21 + b24),
                    b34,
                    b13,
                    (b11 + b13 + b14 + b23 + b24 - b31 - b33 - (b34 shl 1)),
                    (b23 + b24),
                    b22,
                    (b33 + b34),
                    (b21 + b22 + b24),
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1])

    val c11 = (-hs[0][1] + hs[0][3] + hs[0][4] - hs[1][1] - hs[1][3] + hs[1][4] - hs[1][6] - hs[1][11] + hs[1][12])
    val c12 = (-hs[0][0] + hs[0][1] - hs[0][7] + hs[0][9] - hs[1][6])
    val c13 = (-hs[0][0] + hs[0][2] + hs[0][5] - hs[0][6] - hs[0][7] - hs[1][6])
    val c14 = (hs[0][6] - hs[0][10] + hs[0][11] + hs[0][14] - hs[1][0] + hs[1][3] - hs[1][7] + hs[1][9])
    val c21 = (-hs[0][1] + hs[0][2] + hs[0][8] + hs[0][12] + hs[0][13] - hs[0][14])
    val c22 = (hs[0][6] + hs[0][8] + hs[0][13] - hs[0][14] + hs[1][10])
    val c23 = (-hs[1][1] + hs[1][4] - hs[1][6] + hs[1][7] + hs[1][11])
    val c24 = (hs[0][0] - hs[0][3] - hs[0][7] + hs[0][9] - hs[1][0] - hs[1][5] - hs[1][6] - hs[1][7] + hs[1][8])
    val c31 = (-hs[0][8] - hs[0][10] + hs[0][15] - hs[1][1] - hs[1][6] + hs[1][11])
    val c32 = (-hs[0][14] + hs[1][0] + hs[1][1] - hs[1][4] + hs[1][6] - hs[1][9])
    val c33 = (hs[0][7] - hs[0][9] + hs[0][14] + hs[1][5] + hs[1][6])
    val c34 = (-hs[0][13] + hs[0][14] + hs[1][2] + hs[1][6] - hs[1][10])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34)
}

fun multiply3By3MatrixAWith3By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 9) { "Matrix A must have 9 elements" }
    require(b.size == 15) { "Matrix B must have 15 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a21 = a[3]
    val a22 = a[4]
    val a23 = a[5]
    val a31 = a[6]
    val a32 = a[7]
    val a33 = a[8]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a33,
                    (a13 + a23),
                    (a13 - a22 + a23),
                    (a11 + a13 - a22 + a23),
                    (a21 + a22 - a31),
                    (a21 - a31),
                    (a22 - a23),
                    (a21 + a22 - a31 - a33),
                    a13,
                    (a21 + a22 - a31 - a32),
                    (a12 - a13 + a22 - a23),
                    (a11 + a12),
                    (a21 + a22),
                    (a11 + a13 + a21 + a23),
                    (a11 + a12 + a32),
                    a11,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a12 + a32 - a33),
                    a32,
                    a31,
                    (a32 - a33),
                    (a21 - a31 - a33),
                    (a11 + a13 + a23),
                    (a12 + a32 - a33),
                    (a12 + a32),
                    (a21 - a31),
                    (a11 + a12 + a31 + a32),
                    (a31 + a33),
                    (a21 + a23 - a31 - a33),
                    a21,
                    (a12 + a32),
                    (a13 + a23),
                    a23,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a12 - a13 + a32 - a33),
                    (a11 + a13),
                    a12,
                    a22,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    b31,
                    (b15 - b25 - b35),
                    (b15 - b25 - b31 - b35),
                    (b15 + b24),
                    (b15 - b21 - b25 - b31 - b35),
                    (b11 + b15 - b21 - b25 - b31 - b35),
                    (b31 + b35),
                    (b22 + b23 - b31 - b35),
                    b31,
                    (b21 + b22 + b23 + b25),
                    (b24 + b25),
                    b24,
                    b15,
                    (b13 + b14 + b15),
                    (b11 + b14 - b24 - b34),
                    (b14 - b24 - b34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b12 + b34),
                    b21,
                    b11,
                    (b22 + b23),
                    (b12 - b22 - b23 - b32 - b33),
                    (b13 + b14 - b24 - b34),
                    (b12 - b22 - b23 - b32),
                    ((b11 shl 1) - b12 + (b14 shl 1) - (b21 shl 1) + b22 - (b24 shl 1) + b32 - (b34 shl 1)),
                    (b12 + b13 - b22 - b23 - b32 - b33),
                    (b11 + b12 + b14),
                    b12,
                    (b31 + b32 + b33 + b35),
                    b13,
                    (b11 + b14 - b21 - b24 - b34),
                    (b13 + b14 - b24 - b33 - b34),
                    b33,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b32 + b34),
                    b34,
                    b23,
                    b23,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2])

    val c11 = (hs[0][8] + hs[0][14] - hs[0][15] - hs[1][1] - hs[1][13])
    val c12 = (hs[0][1] - hs[0][2] - hs[0][4] + hs[0][5] - hs[0][8] + hs[1][2])
    val c13 = (hs[0][0] + hs[1][1] + hs[1][2])
    val c14 = (hs[1][0] - hs[1][3] - hs[1][6] - hs[2][0] - hs[2][1] - hs[2][2])
    val c15 = (hs[0][6] + hs[0][7] + hs[1][4] + hs[1][10] + hs[1][11] - hs[1][15] - hs[2][3])
    val c21 = (hs[1][3] + hs[1][6] + hs[1][7] + hs[1][10] - (hs[1][13] shl 1) + hs[2][2])
    val c22 = (-hs[0][15] + hs[1][5] - hs[1][14] - hs[1][15] + hs[2][2])
    val c23 = (hs[1][12] + hs[1][15] + hs[2][3])
    val c24 = (hs[1][4] - hs[1][6] - hs[1][7] - hs[1][8] + hs[1][12] + (hs[1][13] shl 1) - hs[2][2])
    val c25 = (hs[0][11] + hs[0][15] + hs[2][1])
    val c31 = (-hs[0][3] - hs[0][12] + hs[0][13] - hs[1][5] - hs[1][12] - hs[2][1])
    val c32 = (-hs[0][11] - hs[0][14] - hs[1][0] - hs[1][2] + hs[1][9] - hs[1][10])
    val c33 = (-hs[0][2] + hs[0][3] + hs[0][6] - hs[0][8] + hs[0][10] - hs[0][11])
    val c34 = (-hs[0][1] + hs[0][2] - hs[0][6] + hs[0][8] + hs[0][12])
    val c35 = (-hs[0][0] - hs[0][4] + hs[0][7] - hs[0][9] + hs[0][12] - hs[1][1] - hs[1][3])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25, c31, c32, c33, c34, c35)
}

fun multiply3By4MatrixAWith4By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 12) { "Matrix A must have 12 elements" }
    require(b.size == 16) { "Matrix B must have 16 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val a31 = a[8]
    val a32 = a[9]
    val a33 = a[10]
    val a34 = a[11]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]
    val b31 = b[8]
    val b32 = b[9]
    val b33 = b[10]
    val b34 = b[11]
    val b41 = b[12]
    val b42 = b[13]
    val b43 = b[14]
    val b44 = b[15]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a12,
                    (a12 + a23),
                    (a22 + a23 - a32),
                    (a12 - a22 + a24),
                    -a11,
                    (-a11 - a33),
                    (-a11 + a31 + a34),
                    -a33,
                    -a23,
                    -a32,
                    a34,
                    (-a31 - a34),
                    (a22 - a24 - a32),
                    -a14,
                    (-a11 + a31 + a32),
                    (a21 - a31 - a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a21 - a31),
                    (-a13 + a14 + a33),
                    (-a11 - a12 - a13),
                    (-a13 + a14 + a23),
                    (-a13 + a33),
                    (-(a11 shl 1) - a13 + a23),
                    (a12 + a13 - a33),
                    (-a11 + a12 - a14 - a22 + a24 + a31 + a34),
                    (a21 + a32),
                    (a12 - a21 - a22),
                    a21,
                    (-a13 + a14 - a22 + a24 + a32 + a33 - a34),
                    (a21 - a31 - a33),
                    (-a13 + a14),
                    -a11,
                    -a24,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a22 - a24),
                    (a21 + a22 + a23),
                    (a22 + a23 - a32 - a33),
                    a21,
                    (-a13 + a14 - a21 + a23 - a24 + a31 + a34),
                    (a22 - a32),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b21 + b24 - b31),
                    (-b23 - b31),
                    (-b23 + b32 + b33 + b34),
                    (-b24 + b41),
                    (-b11 - b14 + b31),
                    (b12 + b31),
                    (-b14 - b41),
                    (b12 - b32 - b34),
                    (-b23 + b33 + b34),
                    (-b11 - b12 + b21 + b22 + b24 + b41),
                    (-b14 + b42 + b44),
                    -b14,
                    (-b24 + b42),
                    (b34 + b41 + b44),
                    (-b11 - b12 + b41),
                    (-b14 - b43),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b12 + b13 - b43),
                    (b34 - b42),
                    -b31,
                    (b34 - b43),
                    (b21 + b22 + b24 - b31 - b32 - b42),
                    (-b31 - b33 - b43),
                    (b21 + b22 + b24),
                    b41,
                    (-b11 - b12 + b21 + b41),
                    (b21 + b23 + b41),
                    (-b11 + b21 + b41),
                    -b42,
                    b12,
                    b34,
                    (-b11 + b13 - b14 - (b33 shl 1) - (b43 shl 1)),
                    (-b24 - b43 - b44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -b24,
                    -b23,
                    (-b32 - b33 - b34),
                    (-b13 - b14 + b23),
                    b43,
                    (b22 + b23 - b32 - b33 - b34 + b42),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2])

    val c11 = (hs[0][0] + hs[0][3] + hs[0][4] - hs[0][6] - hs[0][11] + hs[1][2] - hs[1][7] + hs[2][0])
    val c12 = (hs[0][0] - hs[0][1] + hs[0][3] - hs[1][9] - hs[1][10] + hs[2][0] + hs[2][1])
    val c13 = (hs[0][4] - hs[0][5] - hs[0][6] - hs[0][11] - hs[0][14] + hs[1][8] - hs[1][10] + hs[1][12])
    val c14 = (-hs[0][0] - hs[0][5] + hs[0][7] - hs[1][1] - hs[1][2] + hs[1][4] + hs[1][6] + hs[1][13])
    val c21 = (hs[0][2] + hs[0][8] - hs[0][9] - hs[0][12] - hs[1][8] + hs[1][10] + hs[2][0] + hs[2][5])
    val c22 = (hs[0][7] - hs[0][9] - hs[0][12] - hs[1][1] - hs[1][8] + hs[1][10] + hs[1][11] - hs[1][12] + hs[1][13] + hs[2][0])
    val c23 = (-hs[0][1] + hs[0][4] - hs[0][8] - hs[1][2] - hs[1][3] + hs[1][5] + hs[1][13] - hs[1][14])
    val c24 = (-hs[0][8] - hs[0][11] + hs[0][15] - hs[1][3] + hs[1][13] - hs[2][1] - hs[2][3] - hs[2][4])
    val c31 = (hs[0][2] - hs[0][7] - hs[0][11] + hs[0][15] - hs[1][0] + hs[1][12] - hs[2][1] + hs[2][2] - hs[2][3])
    val c32 = (-hs[0][3] + hs[0][6] + hs[0][11] - hs[0][13] + hs[1][7] - hs[1][13] - hs[2][0])
    val c33 = (hs[0][11] - hs[0][15] + hs[1][3] - hs[1][13] + hs[1][15] - hs[2][0] + hs[2][4])
    val c34 = (hs[0][10] + hs[0][11] + hs[0][12] + hs[1][1] - hs[1][11] - hs[1][13] - hs[2][0])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34)
}

fun multiply3By4MatrixAWith4By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 12) { "Matrix A must have 12 elements" }
    require(b.size == 20) { "Matrix B must have 20 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val a31 = a[8]
    val a32 = a[9]
    val a33 = a[10]
    val a34 = a[11]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    -a22,
                    (-a22 - a32 - a34),
                    -a21,
                    (a11 + a14 - a21),
                    (a11 - a13 - a21),
                    (a11 - a13 - a21 + a22 + a23 + a32 + a33),
                    (a11 + a14 - a21 - a22 - a24 - a32 - a34),
                    -a12,
                    (-a12 + a32 + a34),
                    (a11 - a12 - a21),
                    (-a12 + a31),
                    a31,
                    (-a22 + a31 - a32),
                    (a11 - a13 - a31),
                    (a32 + a34),
                    (a11 - a13),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a14),
                    (a11 + a14 - a31),
                    (-a22 - a32 - a33),
                    (-a12 + a32 + a33),
                    (-a12 - a14 + a32 + a34),
                    (-a12 - a14 + a23 - a24 + a32 + a34),
                    (a23 - a24),
                    (-a22 - a32),
                    (-a21 + a22 - a31 + a32),
                    (-a12 + a32),
                    (a11 - a21),
                    (-a12 - a14 - a24 + a32 + a34),
                    (a11 - a12),
                    a33,
                    (a11 - a31),
                    (a23 - a24 - a33 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a34,
                    a13,
                    a14,
                    (a32 + a33),
                    (a33 - a34),
                    ((a11 shl 1) - a13 + a14 - a23 + a24 - (a31 shl 1) + a33 - a34),
                    (a11 - a13 - a23 + a24 - a31 + a33),
                    (a13 - a14 + a23 - a24 - a33 + a34),
                    a23,
                    (a11 - (a12 shl 1) - a14 - a21 + a22 - a24 + a32 + a34),
                    (a11 - a13 - a23 - a31 + a33),
                    a24,
                    (a11 - a13 - a31 + a33),
                    (a13 + a14 - a23 + a24),
                    (a31 - a32),
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b23 - b24 - b25),
                    (b25 - b43),
                    (b13 - b14 - b15),
                    (-b15 + b43),
                    (b14 + b33),
                    -b33,
                    -b43,
                    (-b12 - b22 + b24 + b25),
                    (b25 - b42 + b44),
                    (-b12 - b22 - b23 + b24 + b25),
                    (-b12 - b21),
                    (b11 - b14 - b15 + b21),
                    (b11 + b13 - b14 - b15 + b21),
                    (b14 + b31 - b35),
                    b25,
                    b14,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -b15,
                    (-b15 - b35 + b41),
                    (b24 - b33),
                    (b24 - b32 - b44),
                    (b42 - b44),
                    (b32 + b44),
                    -b35,
                    (b11 + b13 - b14 - b15 + b21 + b23 - b33 - b43),
                    (b11 + b13 - b14 - b15),
                    (-b21 + b22 - b32 - b42),
                    (-b12 - b13 - b22 - b23 + b24 + b25 - b33 + b43),
                    (b22 + b23 - b24 - b25 + b32 + b42),
                    -b12,
                    (b24 + b31 - b34 - b35 - b44),
                    (b11 - b12 + b31 - b41),
                    -b44,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b25 + b41 - b45),
                    (b14 - b32 + b34),
                    (-b15 - b35 - b42 + b44 + b45),
                    b24,
                    -b44,
                    b41,
                    (-b35 - b41),
                    b32,
                    (-b33 + b34 + b44),
                    (b22 + b23 - b24 - b25),
                    (b31 + b41),
                    (-b35 + b43 - b45),
                    (b31 - b35),
                    -b35,
                    -b21,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2])

    val c11 = (hs[0][10] + hs[0][11] - hs[0][13] + hs[0][15] - hs[1][0] + hs[1][1] - hs[1][6] - hs[1][12] + hs[1][14] - hs[2][13])
    val c12 = (hs[0][2] - hs[0][12] - hs[1][6] - hs[1][8] - hs[2][6] - hs[2][10] + hs[2][12] - hs[2][14])
    val c13 = (hs[0][11] - hs[0][13] + hs[0][15] - hs[1][0] + hs[1][1] - (hs[1][6] shl 1) - hs[2][5] - hs[2][6] + hs[2][12] - hs[2][13] + hs[2][14])
    val c14 = (hs[0][7] - hs[0][8] + hs[0][14] - hs[1][3] - hs[1][4] - hs[1][5] - hs[1][12] - hs[1][15] + hs[2][3] + hs[2][7])
    val c15 = (hs[0][0] + hs[0][9] + hs[1][4] + hs[1][5] - hs[1][11] - hs[1][12] + hs[1][15] + hs[2][4] + hs[2][9])
    val c21 = (hs[0][7] - hs[0][8] - hs[0][10] + hs[0][14] - hs[1][3] + hs[1][9] + hs[2][3] + hs[2][4] + hs[2][14])
    val c22 = (-hs[0][2] + hs[0][3] - hs[0][4] - hs[0][7] + hs[0][9] + hs[0][15] - hs[1][0] - hs[1][10])
    val c23 = (-hs[0][0] - hs[0][1] - hs[0][2] + hs[0][3] - hs[0][4] - hs[0][5] + hs[0][6] - hs[0][14] + hs[0][15] - hs[1][0] - hs[1][2] - hs[2][3])
    val c24 = (hs[0][0] + hs[0][1] - hs[0][11] + hs[0][12] + hs[0][14] + hs[1][2] - hs[1][7] + hs[2][3])
    val c25 = (hs[0][15] - hs[1][3] - hs[1][5] - hs[1][15] + hs[2][1] + hs[2][3] + hs[2][7])
    val c31 = (-hs[0][4] - hs[0][5] + hs[0][15] - hs[1][2] + hs[1][15] - hs[2][3] + hs[2][4] + hs[2][8])
    val c32 = (-hs[0][13] + hs[0][15] - hs[1][13] + hs[2][3] + hs[2][4] + hs[2][12])
    val c33 = (-hs[0][8] + hs[0][14] - hs[1][0] - hs[1][4] - hs[1][6] + hs[2][2] - hs[2][13])
    val c34 = (-hs[0][1] + hs[0][3] + hs[0][6] - hs[0][14] - hs[1][0] - hs[1][6] - hs[2][11])
    val c35 = (hs[0][14] - hs[1][0] + hs[1][1] - (hs[1][6] shl 1) - hs[2][0] - hs[2][5] - hs[2][6] - hs[2][13])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25, c31, c32, c33, c34, c35)
}

fun multiply3By4MatrixAWith4By11MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 12) { "Matrix A must have 12 elements" }
    require(b.size == 44) { "Matrix B must have 44 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val a31 = a[8]
    val a32 = a[9]
    val a33 = a[10]
    val a34 = a[11]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b18 = b[7]
    val b19 = b[8]
    val b110 = b[9]
    val b111 = b[10]
    val b21 = b[11]
    val b22 = b[12]
    val b23 = b[13]
    val b24 = b[14]
    val b25 = b[15]
    val b26 = b[16]
    val b27 = b[17]
    val b28 = b[18]
    val b29 = b[19]
    val b210 = b[20]
    val b211 = b[21]
    val b31 = b[22]
    val b32 = b[23]
    val b33 = b[24]
    val b34 = b[25]
    val b35 = b[26]
    val b36 = b[27]
    val b37 = b[28]
    val b38 = b[29]
    val b39 = b[30]
    val b310 = b[31]
    val b311 = b[32]
    val b41 = b[33]
    val b42 = b[34]
    val b43 = b[35]
    val b44 = b[36]
    val b45 = b[37]
    val b46 = b[38]
    val b47 = b[39]
    val b48 = b[40]
    val b49 = b[41]
    val b410 = b[42]
    val b411 = b[43]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    -a22,
                    (-a22 - a32 - a34),
                    -a21,
                    (a11 + a14 - a21),
                    (a11 - a13 - a21),
                    (a11 - a13 - a21 + a22 + a23 + a32 + a33),
                    (a11 + a14 - a21 - a22 - a24 - a32 - a34),
                    -a12,
                    (-a12 + a32 + a34),
                    (a11 - a12 - a21),
                    (-a12 + a31),
                    a31,
                    (-a22 + a31 - a32),
                    (a11 - a13 - a31),
                    (a32 + a34),
                    (a11 - a13),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a14),
                    (a11 + a14 - a31),
                    (-a22 - a32 - a33),
                    (-a12 + a32 + a33),
                    (-a12 - a14 + a32 + a34),
                    (-a12 - a14 + a23 - a24 + a32 + a34),
                    (a23 - a24),
                    (-a22 - a32),
                    (-a21 + a22 - a31 + a32),
                    (-a12 + a32),
                    (a11 - a21),
                    (-a12 - a14 - a24 + a32 + a34),
                    (a11 - a12),
                    a33,
                    (a11 - a31),
                    (a23 - a24 - a33 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a34,
                    a13,
                    a14,
                    (a32 + a33),
                    (a33 - a34),
                    ((a11 shl 1) - a13 + a14 - a23 + a24 - (a31 shl 1) + a33 - a34),
                    (a11 - a13 - a23 + a24 - a31 + a33),
                    (a13 - a14 + a23 - a24 - a33 + a34),
                    a23,
                    (a11 - (a12 shl 1) - a14 - a21 + a22 - a24 + a32 + a34),
                    (a11 - a13 - a23 - a31 + a33),
                    a24,
                    (a11 - a13 - a31 + a33),
                    (a13 + a14 - a23 + a24),
                    (a31 - a32),
                    a13,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a31 - a32 - a33 - a34),
                    (a21 + a22),
                    -a22,
                    (a11 + a12),
                    (a11 + a14),
                    (-a13 + a22),
                    (a33 + a34),
                    a32,
                    (-a21 - a22 + a31 + a32 + a33 + a34),
                    (a22 + a23),
                    (a12 - a14 - a33 - a34),
                    (a21 + a24),
                    (a22 + a23 - a24),
                    (a12 + a13),
                    (-a11 - a12 + a33 + a34),
                    (a11 - a22),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a21 + a22 + a23 - a24 + a31 + a32 + a33 + a34),
                    (a14 + a22),
                    (a21 - a22 - a23 + a24),
                    -a34,
                    (-a22 + a24),
                    a31,
                    (a32 + a33),
                    -a24,
                    -a14,
                    (a21 + a22 - a31),
                    (a13 - a32),
                    -a11,
                    (a14 - a22 - a23 + a24),
                    (-a11 + a24),
                    (a11 + a22 + a23 - a24),
                    (-a32 - a33),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a13 + a14 + a33 + a34),
                    (-a12 + a14),
                    (a21 + a22 + a23 + a24 - a31 - a32 - a33 - a34),
                    (-a21 - a22 + a31 + a32),
                    (-a21 + a22 + a23 - a24 + a31),
                    (-a11 + a12 + a13 - a14),
                    (a32 + a33 - a34),
                    (a12 + a22),
                    (-a12 + a14 - a32),
                    (-a13 - a33),
                    (a11 + a12 + a32),
                    (-a13 - a23),
                    (a21 + a24 - a31),
                    (a11 + a12 + a31 + a32),
                    (a11 + a21),
                    (-a21 - a24 + a31 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a14 - a24),
                    (a12 - a14 + a32 - a34),
                    (a14 + a24),
                    (-a21 + a31),
                    (a12 + a32),
                    (a21 - a22 - a23 + a24 - a31 + a32 + a33 - a34),
                    (a12 + a13 - a14 + a22 + a23 - a24),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-b23 + b24 + b25),
                    (-b25 + b43),
                    (-b13 + b14 + b15),
                    (b15 - b43),
                    (-b14 - b33),
                    b33,
                    b43,
                    (b12 + b22 - b24 - b25),
                    (-b25 + b42 - b44),
                    (b12 + b22 + b23 - b24 - b25),
                    (b12 + b21),
                    (-b11 + b14 + b15 - b21),
                    (-b11 - b13 + b14 + b15 - b21),
                    (-b14 - b31 + b35),
                    -b25,
                    -b14,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b15,
                    (b15 + b35 - b41),
                    (-b24 + b33),
                    (-b24 + b32 + b44),
                    (-b42 + b44),
                    (-b32 - b44),
                    b35,
                    (-b11 - b13 + b14 + b15 - b21 - b23 + b33 + b43),
                    (-b11 - b13 + b14 + b15),
                    (b21 - b22 + b32 + b42),
                    (b12 + b13 + b22 + b23 - b24 - b25 + b33 - b43),
                    (-b22 - b23 + b24 + b25 - b32 - b42),
                    b12,
                    (-b24 - b31 + b34 + b35 + b44),
                    (-b11 + b12 - b31 + b41),
                    b44,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b25 - b41 + b45),
                    (-b14 + b32 - b34),
                    (b15 + b35 + b42 - b44 - b45),
                    -b24,
                    b44,
                    -b41,
                    (b35 + b41),
                    -b32,
                    (b33 - b34 - b44),
                    (-b22 - b23 + b24 + b25),
                    (-b31 - b41),
                    (b35 - b43 + b45),
                    (-b31 + b35),
                    b35,
                    b21,
                    (-b29 + b39),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b18 + b111),
                    -b111,
                    (-b111 + b211 - b311 + b411),
                    (b210 - b310),
                    (-b310 - b410),
                    (-b29 + b311),
                    (-b26 + b36),
                    (b27 + b29 - b37 - b39),
                    (b26 - b36 - b411),
                    -b311,
                    (-b18 + b111 + b26 - b36 + b38 + b310 - b311 - b410),
                    -b19,
                    (b16 - b26 + b36 - b46),
                    -b29,
                    (-b18 + b111 - b410),
                    (-b111 - b210 + b310),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b18 - b111 - b26 + (b36 shr 1) - (b311 shr 1) - (b46 shr 1) + (b411 shr 1)),
                    (-b26 + b28 - b211 + b36 - b38 + b311 - b411),
                    (-b16 + b18 - b111),
                    -b47,
                    -b411,
                    -b17,
                    (-b311 + b411),
                    (-b19 + b49),
                    (b18 - b111 - b28 + b211 + b38 - b311 - b48 + b411),
                    (b111 - b27 - b29 - b211 + b37 + b39 + b311 - b411),
                    (b29 - b37 - b39),
                    (-b110 + b210 - b310 + b410),
                    (b18 - b111 - b26 + b38 - b311 - b46),
                    (b19 - b310 - b410),
                    (-b16 + b18 - b111 - b310),
                    (b310 - b410),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b38 - b310 + b311),
                    (b26 - b28 + b211 - b36 + b38 - b311),
                    (-(b36 shr 1) - (b311 shr 1) - (b46 shr 1) - (b411 shr 1)),
                    (-b26 - b27 - b29 - b211 + b36 + b37 + b39 + b311),
                    (-b16 + b26 - b36 - b37 - b39 + b46),
                    -b310,
                    (-b37 - b39),
                    (b26 - b28 + b29 - b210 - b36 + b38 + b310 - b311),
                    (-b18 + b111 + b28 - b211 - b38 + b311 - b47 + b48 - b411),
                    -b37,
                    (-b17 + b110 - b210 + b310 - b410),
                    (b39 - b311),
                    (b19 - b37 - b39 - b47 - b49),
                    (b17 - b18 - b110 + b111),
                    (b16 - b18 + b19 - b110),
                    (-b37 - b39 - b47 - b49),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b310 + b49 - b410),
                    (-b38 - b310 + b311 - b47 + b48 + b410 - b411),
                    (b36 - b38 + b311 + b46 - b48),
                    (b16 + b17 + b19 + b111 - b26 - b27 - b29 - b211 + b36 + b37 + b39 + b311 - b46 - b47 - b49 - b411),
                    (b17 - b18 - b110 + b111 - b27 + b28 + b210 - b211 - b38 - b310 + b311 - b47 + b48 + b410 - b411),
                    (-(b36 shr 1) - b37 - b39 - (b311 shr 1) + (b46 shr 1) + (b411 shr 1)),
                    (-b36 + b38 + b310 - b311),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3], lefts[4] * rights[4], lefts[5] * rights[5], lefts[6] * rights[6])

    val c11 = (-hs[0][10] - hs[0][11] + hs[0][13] - hs[0][15] + hs[1][0] - hs[1][1] + hs[1][6] + hs[1][12] - hs[1][14] + hs[2][13])
    val c12 = (-hs[0][2] + hs[0][12] + hs[1][6] + hs[1][8] + hs[2][6] + hs[2][10] - hs[2][12] + hs[2][14])
    val c13 = (-hs[0][11] + hs[0][13] - hs[0][15] + hs[1][0] - hs[1][1] + (hs[1][6] shl 1) + hs[2][5] + hs[2][6] - hs[2][12] + hs[2][13] - hs[2][14])
    val c14 = (-hs[0][7] + hs[0][8] - hs[0][14] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][12] + hs[1][15] - hs[2][3] - hs[2][7])
    val c15 = (-hs[0][0] - hs[0][9] - hs[1][4] - hs[1][5] + hs[1][11] + hs[1][12] - hs[1][15] - hs[2][4] - hs[2][9])
    val c16 = (-hs[0][7] + hs[0][8] + hs[0][10] - hs[0][14] + hs[1][3] - hs[1][9] - hs[2][3] - hs[2][4] - hs[2][14])
    val c17 = (hs[0][2] - hs[0][3] + hs[0][4] + hs[0][7] - hs[0][9] - hs[0][15] + hs[1][0] + hs[1][10])
    val c18 = (hs[0][0] + hs[0][1] + hs[0][2] - hs[0][3] + hs[0][4] + hs[0][5] - hs[0][6] + hs[0][14] - hs[0][15] + hs[1][0] + hs[1][2] + hs[2][3])
    val c19 = (-hs[0][0] - hs[0][1] + hs[0][11] - hs[0][12] - hs[0][14] - hs[1][2] + hs[1][7] - hs[2][3])
    val c110 = (-hs[0][15] + hs[1][3] + hs[1][5] + hs[1][15] - hs[2][1] - hs[2][3] - hs[2][7])
    val c111 = (hs[0][4] + hs[0][5] - hs[0][15] + hs[1][2] - hs[1][15] + hs[2][3] - hs[2][4] - hs[2][8])
    val c21 = (hs[0][13] - hs[0][15] + hs[1][13] - hs[2][3] - hs[2][4] - hs[2][12])
    val c22 = (hs[0][8] - hs[0][14] + hs[1][0] + hs[1][4] + hs[1][6] - hs[2][2] + hs[2][13])
    val c23 = (hs[0][1] - hs[0][3] - hs[0][6] + hs[0][14] + hs[1][0] + hs[1][6] + hs[2][11])
    val c24 = (-hs[0][14] + hs[1][0] - hs[1][1] + (hs[1][6] shl 1) + hs[2][0] + hs[2][5] + hs[2][6] + hs[2][13])
    val c25 = (hs[3][4] - hs[3][6] + hs[3][10] - hs[3][12] + hs[3][14] - hs[4][12] - hs[4][14] - hs[5][0] - hs[6][6])
    val c26 = (hs[3][0] - hs[3][8] + hs[3][9] + hs[3][12] - hs[4][0] - hs[4][2] + hs[4][4] - hs[5][2])
    val c27 = (hs[3][0] + hs[3][6] - hs[4][0] - hs[4][2] + hs[4][6] - hs[5][4] + hs[5][6] - hs[6][5])
    val c28 = (-hs[2][15] - hs[3][7] + hs[4][8] - hs[4][10] + hs[4][11] - hs[5][8] - hs[5][10] - hs[6][4])
    val c29 = (hs[2][15] + hs[3][2] + hs[3][5] + hs[3][9] - hs[3][12] - hs[4][5] + hs[4][7] - hs[4][9] - hs[5][4] + hs[5][11] - hs[5][12] - hs[6][3])
    val c210 = (hs[2][15] + hs[3][7] + hs[4][3] - hs[4][5] + hs[4][10] + hs[5][9])
    val c211 = (hs[3][2] - hs[3][3] + hs[3][4] - hs[3][5] - hs[3][6] + hs[3][10] - hs[3][13] + hs[3][14] - hs[3][15] - hs[4][1] + hs[4][8] - hs[5][0] + hs[5][5] - hs[5][7])
    val c31 = (hs[3][0] - hs[3][1] - hs[3][2] - hs[3][8] - hs[4][0] + hs[4][1] + hs[4][4] - hs[4][8] - hs[4][12] - hs[5][2] - hs[6][2])
    val c32 = (hs[3][0] - hs[3][1] - hs[3][7] - hs[3][8] - hs[3][10] + hs[4][3] - hs[4][6] - hs[4][9] + hs[4][15] - hs[5][1] - hs[5][3] - hs[5][8] - hs[6][1])
    val c33 = (hs[2][15] - hs[3][4] - hs[3][13] + hs[4][7] - hs[4][13] - hs[6][0])
    val c34 = (-hs[2][15] - hs[3][5] - hs[3][9] - hs[3][11] - hs[4][7] - hs[5][11])
    val c35 = (-hs[2][15] - hs[3][11] - hs[4][3] - hs[4][10] - hs[5][6] - hs[5][9] - hs[5][12] - hs[5][15])
    val c36 = (hs[3][3] - hs[3][4] + hs[4][11] - hs[5][5])
    val c37 = (hs[3][1] - hs[3][11] + hs[3][15] - hs[4][2] - hs[4][11] - hs[4][13] - hs[4][14] - hs[5][14])
    val c38 = (-hs[3][0] - hs[3][3] - hs[3][14] - hs[4][5] - hs[4][15] - hs[5][10] - hs[5][13])
    val c39 = (hs[3][2] - hs[3][3] - hs[3][5] - hs[3][13] - hs[3][15] - hs[4][1] - hs[5][1] - hs[5][7])
    val c310 = (-hs[3][1] - hs[3][2] - hs[3][9] - hs[4][4])
    val c311 = (-hs[3][1] - hs[3][6] - hs[3][7] - hs[3][8] - hs[4][6] - hs[4][9] - hs[5][3])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c18, c19, c110, c111, c21, c22, c23, c24, c25, c26, c27, c28, c29, c210, c211, c31, c32, c33, c34, c35, c36, c37, c38, c39, c310, c311)
}

fun multiply3By5MatrixAWith5By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 15) { "Matrix A must have 15 elements" }
    require(b.size == 25) { "Matrix B must have 25 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a21 = a[5]
    val a22 = a[6]
    val a23 = a[7]
    val a24 = a[8]
    val a25 = a[9]
    val a31 = a[10]
    val a32 = a[11]
    val a33 = a[12]
    val a34 = a[13]
    val a35 = a[14]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]
    val b51 = b[20]
    val b52 = b[21]
    val b53 = b[22]
    val b54 = b[23]
    val b55 = b[24]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a33,
                    a34,
                    (-a21 + a33),
                    (-a11 - a34),
                    -a11,
                    (-a11 - a14 + a24),
                    (a12 + a14 - a34),
                    (a23 - a25 - a33),
                    (-a31 + a33 - a34),
                    (a14 - a15 - a34),
                    -a21,
                    (-a22 - a23 + a33),
                    (a12 + a14 + a22 + a23 - a32 - a33 - a34),
                    (-a14 + a15 - a23 + a25 + a33 + a34 - a35),
                    (a23 - a24 - a33),
                    (-a21 + a31 + a32),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a31 - a32),
                    (a21 - a31 - a35),
                    (-a31 - a35),
                    -a24,
                    (a14 - a15 - a24),
                    (a12 + a14 - a24),
                    (a14 - a15 + a21 - a24 + a25 - a31 - a35),
                    (a13 - a24),
                    (a14 - a24),
                    (a12 + a14 - a21 - a22 - a24 + a31 + a32),
                    (a14 - a15),
                    (a13 + a21 - a23),
                    (-a11 + a13 - a14),
                    (a12 + a14),
                    (-a22 - a23),
                    a33,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a34,
                    a13,
                    (a13 - a23 + a25),
                    (a13 - a22 - a23),
                    (a13 - a23),
                    a12,
                    (a23 - a25),
                    (-a11 + a31 + a35),
                    (-a11 + a13 - a15 - a23 + a25 + a31 + a35),
                    (-a11 + a31 + a32),
                    (a12 + a13 - a15 - a22 - a23 + a25),
                    (a12 + a13 - a15 - a23 + a25),
                    (a13 - a14 + a34),
                    a13,
                    (a12 + a13 - a15 - a22 - a23),
                    -a35,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a32 - a35),
                    a25,
                    (-a11 - a12 - a13 + a22 + a23 + a31 + a32),
                    a15,
                    (-a11 + a13 - a14 + a21 - a23 + a24),
                    -a32,
                    -a22,
                    (-a21 + a31 - a34),
                    (-a11 + a31),
                    (a32 - a35),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-b15 - b31 + b33 - b35),
                    (b15 - b41 + b43 - b45),
                    (b15 - b34),
                    (b15 + b42),
                    (-b11 + b12 - b13 - b42),
                    (-b11 + b12 - b13 + b14 - b42),
                    (b25 + b41),
                    (-b33 - b55),
                    b15,
                    (-b43 - b55),
                    (-b11 - b13 + b14 + b34),
                    (b25 + b31),
                    b25,
                    -b55,
                    (b25 - b34 - b35 - b55),
                    (-b11 + b24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -b11,
                    (b13 - b54),
                    b13,
                    (b25 - b34 - b35 + b41 + b43 - b44 - b45 - b55),
                    (-b43 - b54),
                    (-b24 + b41),
                    -b54,
                    (b25 - b34 - b35 - b45 - b55),
                    (-b11 + b12 - b13 + b14 + b24 - b42 - b44 - b54),
                    -b24,
                    -b43,
                    b34,
                    -b42,
                    b41,
                    -b31,
                    (-b31 - b32 + b33 - b35),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b14 + b15 - b24 + b41 - b43 + b44 + b45 - b54),
                    (b31 - b32 + b33 - b42),
                    (b33 - b51 + b52),
                    (-b22 + b23 + b31),
                    (-b11 + b12 - b13 + b14 - b22 + b23 + b32 + b34 - b51 + b52),
                    (-b21 + b22 - b23 + b41),
                    b33,
                    (b13 - b23 + b51 - b52),
                    (b23 - b51 + b52),
                    (-b11 + b22 - b23 + b51),
                    (b23 - b51),
                    b23,
                    (b25 - b42 - b45 - b55),
                    (b25 - b35 - b45 - b55),
                    -b51,
                    (-b13 + b23 + b53 - b55),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b23,
                    (b33 - b51 + b53 - b54),
                    (-b22 + b23 - b51),
                    (-b43 - b51 + b52 - b53),
                    (-b11 + b12 - b13 + b14),
                    (-b11 + b21 + b25 + b51),
                    (-b21 + b23 + b24 + b31),
                    (b14 + b15 - b24 - b54),
                    (b12 + b15 - b22 - b52),
                    (b23 - b51),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3])

    val c11 = (hs[1][0] + hs[1][13] + hs[1][14] + hs[2][3] - hs[2][5] + hs[2][9] + hs[2][14] + hs[3][2])
    val c12 = (hs[0][15] + hs[1][0] - hs[1][5] + hs[1][9] + hs[1][13] + hs[1][14] - hs[2][10] + hs[2][11] + hs[2][14] + hs[3][6])
    val c13 = (-hs[0][6] + hs[0][11] + hs[0][12] + hs[1][0] + hs[1][13] + hs[1][14] - hs[3][0] - hs[3][5] + hs[3][9])
    val c14 = (-hs[0][4] + hs[1][0] - hs[1][2] + hs[1][12] + hs[1][14] - hs[2][1] + hs[2][2] + hs[2][3] + hs[2][6] - hs[2][7] - hs[2][8] + hs[2][9] + hs[2][11] + hs[2][14] + hs[3][2])
    val c15 = (-hs[0][5] + hs[0][10] + hs[1][11] + hs[1][12] + hs[1][14] - hs[2][1] + hs[2][2] + hs[2][3] - hs[2][4] + hs[2][6] - hs[2][10] + hs[2][11] + hs[2][14] + hs[3][4])
    val c21 = (hs[0][0] - hs[0][3] - hs[0][4] + hs[0][8] + hs[1][0] - hs[1][2] - hs[1][15] - hs[2][7] + hs[2][9] + hs[3][8] + hs[3][9])
    val c22 = (-hs[1][2] - hs[1][10] + hs[2][2] + hs[2][6] - hs[2][7] - hs[2][8] + hs[2][11] - hs[3][3])
    val c23 = (hs[1][1] - hs[1][2] + hs[1][4] - hs[1][6] - hs[1][10] + hs[2][6] - hs[2][10] + hs[2][11] + hs[2][14] + hs[3][1])
    val c24 = (hs[0][7] + hs[0][9] + hs[0][13] - hs[1][2] - hs[1][10] + hs[2][6] - hs[2][15] + hs[3][0])
    val c25 = (hs[0][4] - hs[0][5] + hs[1][3] + hs[1][4] - hs[1][5] - hs[1][7] - hs[1][8] - hs[1][10] + hs[1][13] + hs[2][13])
    val c31 = (-hs[0][10] + hs[0][15] + hs[1][0] + hs[1][1] - hs[1][2] + hs[1][3] + hs[1][4] - hs[1][5] - hs[1][6] - hs[1][7] + hs[1][9] - hs[1][10] - hs[1][11] + hs[1][13] + hs[2][13])
    val c32 = (hs[0][1] - hs[0][2] + hs[0][8] - hs[0][10] + hs[0][15] + hs[1][0] + hs[1][1] - hs[1][2] + hs[2][0] + hs[3][7])
    val c33 = (-hs[0][1] - hs[0][3] + hs[0][6] + hs[0][9] - hs[1][10] - hs[1][12] - hs[1][13] + hs[2][12] - hs[2][13])
    val c34 = (-hs[0][0] - hs[0][2] + hs[0][7] - hs[0][11] - hs[0][14] + hs[1][7] + hs[1][11] - hs[1][14] + hs[2][6] - hs[2][13])
    val c35 = (-hs[0][0] - hs[0][1] + hs[0][6] + hs[0][7] - hs[0][8] + hs[0][9] - hs[0][11] - hs[0][12] + hs[0][13] - hs[1][10] - hs[1][13] - hs[1][14] + hs[2][6])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25, c31, c32, c33, c34, c35)
}

fun multiply3By5MatrixAWith5By9MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 15) { "Matrix A must have 15 elements" }
    require(b.size == 45) { "Matrix B must have 45 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a21 = a[5]
    val a22 = a[6]
    val a23 = a[7]
    val a24 = a[8]
    val a25 = a[9]
    val a31 = a[10]
    val a32 = a[11]
    val a33 = a[12]
    val a34 = a[13]
    val a35 = a[14]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b18 = b[7]
    val b19 = b[8]
    val b21 = b[9]
    val b22 = b[10]
    val b23 = b[11]
    val b24 = b[12]
    val b25 = b[13]
    val b26 = b[14]
    val b27 = b[15]
    val b28 = b[16]
    val b29 = b[17]
    val b31 = b[18]
    val b32 = b[19]
    val b33 = b[20]
    val b34 = b[21]
    val b35 = b[22]
    val b36 = b[23]
    val b37 = b[24]
    val b38 = b[25]
    val b39 = b[26]
    val b41 = b[27]
    val b42 = b[28]
    val b43 = b[29]
    val b44 = b[30]
    val b45 = b[31]
    val b46 = b[32]
    val b47 = b[33]
    val b48 = b[34]
    val b49 = b[35]
    val b51 = b[36]
    val b52 = b[37]
    val b53 = b[38]
    val b54 = b[39]
    val b55 = b[40]
    val b56 = b[41]
    val b57 = b[42]
    val b58 = b[43]
    val b59 = b[44]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (a22 - a23 + a33),
                    (-a23 - a25 + a33),
                    (-a13 + a21 - a23),
                    (a13 + a23 + a25),
                    (-a13 - a23 - a24),
                    (-a23 - a24),
                    (a23 + a25),
                    (a12 - a13 + a32),
                    (-a12 - a15 - a32),
                    (a13 + a22),
                    (a11 - a32),
                    (a11 + a31 - a33),
                    (-a21 + a33),
                    (-a11 - a31 - a34),
                    (a12 + a15 - a23 - a25 + a32 + a33 + a35),
                    (a11 + a13 + a14 + a23 + a24 + a31 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a11 - a13 - a15 - a23 - a25 - a31 - a35),
                    (a11 + a31 + a35),
                    (-a23 - a24 + a33),
                    (-a12 - a14 - a32),
                    (-a12 - a15 + a22),
                    (-a12 - a14 + a22),
                    (-a11 - a15 - a21 - a25 - (a31 shl 1) - (a35 shl 1)),
                    -a33,
                    -a21,
                    a32,
                    -a13,
                    -a22,
                    (-a11 - a12 - a22),
                    -a34,
                    a11,
                    (-a12 - a14 + a22 + a24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a35,
                    a14,
                    a15,
                    (a12 + a14 - a23 - a24 + a32 + a33 + a34),
                    (a22 + a24 + a32 + a34),
                    (-a31 - a35),
                    (-a21 - a31 - a35),
                    (a12 + a14),
                    a24,
                    a22,
                    -a21,
                    -a25,
                    (a21 + a31 + a34),
                    (-a11 - a15 - a31 - a35),
                    (-a21 + a31 + a32),
                    a33,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a34,
                    (-a21 + a33),
                    (-a11 - a34),
                    -a11,
                    (-a11 - a14 + a24),
                    (a12 + a14 - a34),
                    (a23 - a25 - a33),
                    (-a31 + a33 - a34),
                    (a14 - a15 - a34),
                    -a21,
                    (-a22 - a23 + a33),
                    (a12 + a14 + a22 + a23 - a32 - a33 - a34),
                    (-a14 + a15 - a23 + a25 + a33 + a34 - a35),
                    (a23 - a24 - a33),
                    (-a21 + a31 + a32),
                    (-a31 - a32),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a21 - a31 - a35),
                    (-a31 - a35),
                    -a24,
                    (a14 - a15 - a24),
                    (a12 + a14 - a24),
                    (a14 - a15 + a21 - a24 + a25 - a31 - a35),
                    (a13 - a24),
                    (a14 - a24),
                    (a12 + a14 - a21 - a22 - a24 + a31 + a32),
                    (a14 - a15),
                    (a13 + a21 - a23),
                    (-a11 + a13 - a14),
                    (a12 + a14),
                    (-a22 - a23),
                    a33,
                    a34,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a13,
                    (a13 - a23 + a25),
                    (a13 - a22 - a23),
                    (a13 - a23),
                    a12,
                    (a23 - a25),
                    (-a11 + a31 + a35),
                    (-a11 + a13 - a15 - a23 + a25 + a31 + a35),
                    (-a11 + a31 + a32),
                    (a12 + a13 - a15 - a22 - a23 + a25),
                    (a12 + a13 - a15 - a23 + a25),
                    (a13 - a14 + a34),
                    a13,
                    (a12 + a13 - a15 - a22 - a23),
                    -a35,
                    (a32 - a35),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a25,
                    (-a11 - a12 - a13 + a22 + a23 + a31 + a32),
                    a15,
                    (-a11 + a13 - a14 + a21 - a23 + a24),
                    -a32,
                    -a22,
                    (-a21 + a31 - a34),
                    (-a11 + a31),
                    (a32 - a35),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-b32 + b42 + b52),
                    (b34 - b52),
                    (-b31 + b41 + b51),
                    (-b34 + b51),
                    (-b33 - b41),
                    b33,
                    b34,
                    (b21 + b22 - b42 - b52),
                    (b24 - b44 - b52),
                    (b21 + b22 + b32 - b42 - b52),
                    (b12 + b21),
                    (-b11 - b12 + b41 + b51),
                    (-b11 - b12 - b31 + b41 + b51),
                    (-b13 - b41 + b53),
                    -b52,
                    -b41,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b51,
                    (-b14 + b51 + b53),
                    (b33 - b42),
                    (b23 - b42 + b44),
                    (-b24 + b44),
                    (-b23 - b44),
                    b53,
                    (-b11 - b12 - b31 - b32 + b33 + b34 + b41 + b51),
                    (-b11 - b31 + b41 + b51),
                    (b12 - b22 + b23 + b24),
                    (b21 + b22 + b31 + b32 + b33 - b34 - b42 - b52),
                    (-b22 - b23 - b24 - b32 + b42 + b52),
                    b21,
                    (-b13 - b42 + b43 + b44 + b53),
                    (-b11 - b13 + b14 + b21),
                    b44,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b14 - b52 + b54),
                    (b23 - b41 - b43),
                    (b24 - b44 + b51 + b53 - b54),
                    -b42,
                    b44,
                    -b14,
                    (b14 + b53),
                    -b23,
                    (b33 - b43 - b44),
                    (-b22 - b32 + b42 + b52),
                    (-b13 - b14),
                    (-b34 + b53 + b54),
                    (-b13 + b53),
                    b53,
                    b12,
                    (b19 + b35 - b37 + b39),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b19 + b45 - b47 + b49),
                    (-b19 + b38),
                    (-b19 - b46),
                    (b15 - b16 + b17 + b46),
                    (b15 - b16 + b17 - b18 + b46),
                    (-b29 - b45),
                    (b37 + b59),
                    -b19,
                    (b47 + b59),
                    (b15 + b17 - b18 - b38),
                    (-b29 - b35),
                    -b29,
                    b59,
                    (-b29 + b38 + b39 + b59),
                    (b15 - b28),
                    b15,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b17 + b58),
                    -b17,
                    (-b29 + b38 + b39 - b45 - b47 + b48 + b49 + b59),
                    (b47 + b58),
                    (b28 - b45),
                    b58,
                    (-b29 + b38 + b39 + b49 + b59),
                    (b15 - b16 + b17 - b18 - b28 + b46 + b48 + b58),
                    b28,
                    b47,
                    -b38,
                    b46,
                    -b45,
                    b35,
                    (b35 + b36 - b37 + b39),
                    (-b18 - b19 + b28 - b45 + b47 - b48 - b49 + b58),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b35 + b36 - b37 + b46),
                    (-b37 + b55 - b56),
                    (b26 - b27 - b35),
                    (b15 - b16 + b17 - b18 + b26 - b27 - b36 - b38 + b55 - b56),
                    (b25 - b26 + b27 - b45),
                    -b37,
                    (-b17 + b27 - b55 + b56),
                    (-b27 + b55 - b56),
                    (b15 - b26 + b27 - b55),
                    (-b27 + b55),
                    -b27,
                    (-b29 + b46 + b49 + b59),
                    (-b29 + b39 + b49 + b59),
                    b55,
                    (b17 - b27 - b57 + b59),
                    -b27,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b37 + b55 - b57 + b58),
                    (b26 - b27 + b55),
                    (b47 + b55 - b56 + b57),
                    (b15 - b16 + b17 - b18),
                    (b15 - b25 - b29 - b55),
                    (b25 - b27 - b28 - b35),
                    (-b18 - b19 + b28 + b58),
                    (-b16 - b19 + b26 + b56),
                    (-b27 + b55),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3], lefts[4] * rights[4], lefts[5] * rights[5], lefts[6] * rights[6])

    val c11 = (-hs[0][3] - hs[0][4] - hs[0][5] - hs[0][6] - hs[0][9] - hs[0][13] - hs[0][15] - hs[1][0] - hs[1][1] - hs[1][10] - hs[1][12] - hs[1][14] - (hs[2][5] shl 1) - hs[2][6] - hs[2][9] - hs[2][10] - hs[2][12])
    val c12 = (hs[0][2] + hs[0][3] + hs[0][4] + hs[0][5] + hs[0][6] + hs[0][9] + hs[1][8] + hs[1][10] + hs[2][9])
    val c13 = (-hs[0][10] - hs[0][11] - hs[0][12] + hs[0][13] + hs[1][1] + hs[1][8] + hs[1][14] + (hs[2][5] shl 1) + hs[2][6] + hs[2][10] + hs[2][12] - hs[2][14])
    val c14 = (hs[0][7] + hs[0][8] + hs[0][9] + hs[0][10] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][9] + hs[1][11] + hs[1][12] + (hs[2][9] shl 1))
    val c15 = (hs[0][0] + hs[0][1] - hs[0][5] + hs[0][6] + hs[0][12] + hs[1][2] + hs[1][7] - hs[1][8] - hs[2][9])
    val c16 = (hs[0][1] - hs[0][5] + hs[0][6] - hs[0][8] + hs[0][12] - hs[0][14] + hs[1][2] - hs[1][3] - hs[1][4] - hs[1][5] + hs[1][7] - hs[1][8] - hs[1][9] - hs[1][11] - hs[2][3] - hs[2][9] + hs[2][14])
    val c17 = (hs[0][4] + hs[0][5] + hs[0][13] + hs[0][15] - hs[2][1] + hs[2][5] + hs[2][6] - hs[2][7] + hs[2][10] + hs[2][12] - hs[2][13])
    val c18 = (-hs[0][5] - hs[1][5] - hs[1][6] - hs[1][15] + hs[2][5] + hs[2][6] - hs[2][7] - hs[2][8] + hs[2][10] + hs[2][13])
    val c19 = (-hs[0][5] + hs[1][2] - hs[1][3] - hs[1][13] + hs[1][15] - hs[2][3] - hs[2][4] - hs[2][5] - hs[2][6] + hs[2][7] - hs[2][10] - hs[2][12])
    val c21 = (-hs[0][3] - hs[0][6] - hs[1][0] - hs[1][1] + hs[1][4] + hs[1][5] + hs[1][11] - hs[2][2] - hs[2][5] + hs[2][7] + hs[2][9] - hs[2][13])
    val c22 = (hs[0][6] + hs[1][5] + hs[1][6] + hs[1][11] + hs[1][15] - hs[2][5] - hs[2][6] + hs[2][7] + hs[2][9] - hs[2][11] - hs[2][13])
    val c23 = (hs[0][1] + hs[0][6] - hs[0][8] - hs[0][14] - hs[1][4] - hs[1][5] - hs[1][11] - hs[1][15] - hs[2][0] + hs[2][4] + hs[2][5] - hs[2][7] - hs[2][9])
    val c24 = (-hs[3][15] - hs[4][12] - hs[4][13] - hs[5][2] + hs[5][4] - hs[5][8] - hs[5][13] - hs[6][1])
    val c25 = (-hs[3][14] - hs[3][15] + hs[4][4] - hs[4][8] - hs[4][12] - hs[4][13] + hs[5][9] - hs[5][10] - hs[5][13] - hs[6][5])
    val c26 = (hs[3][5] - hs[3][10] - hs[3][11] - hs[3][15] - hs[4][12] - hs[4][13] + hs[5][15] + hs[6][4] - hs[6][8])
    val c27 = (hs[3][3] - hs[3][15] + hs[4][1] - hs[4][11] - hs[4][13] + hs[5][0] - hs[5][1] - hs[5][2] - hs[5][5] + hs[5][6] + hs[5][7] - hs[5][8] - hs[5][10] - hs[5][13] - hs[6][1])
    val c28 = (hs[3][4] - hs[3][9] - hs[4][10] - hs[4][11] - hs[4][13] + hs[5][0] - hs[5][1] - hs[5][2] + hs[5][3] - hs[5][5] + hs[5][9] - hs[5][10] - hs[5][13] - hs[6][3])
    val c29 = (-hs[2][15] + hs[3][2] + hs[3][3] - hs[3][7] - hs[3][15] + hs[4][1] + hs[4][14] + hs[5][6] - hs[5][8] - hs[6][7] - hs[6][8])
    val c31 = (hs[4][1] + hs[4][9] - hs[5][1] - hs[5][5] + hs[5][6] + hs[5][7] - hs[5][10] + hs[6][2])
    val c32 = (-hs[4][0] + hs[4][1] - hs[4][3] + hs[4][5] + hs[4][9] - hs[5][5] + hs[5][9] - hs[5][10] - hs[5][13] - hs[6][0])
    val c33 = (-hs[3][6] - hs[3][8] - hs[3][12] + hs[4][1] + hs[4][9] - hs[5][5] + hs[5][14] - hs[5][15])
    val c34 = (-hs[3][3] + hs[3][4] - hs[4][2] - hs[4][3] + hs[4][4] + hs[4][6] + hs[4][7] + hs[4][9] - hs[4][12] - hs[5][12])
    val c35 = (hs[3][9] - hs[3][14] - hs[3][15] - hs[4][0] + hs[4][1] - hs[4][2] - hs[4][3] + hs[4][4] + hs[4][5] + hs[4][6] - hs[4][8] + hs[4][9] + hs[4][10] - hs[4][12] - hs[5][12])
    val c36 = (-hs[3][0] + hs[3][1] - hs[3][7] + hs[3][9] - hs[3][14] - hs[3][15] - hs[4][0] + hs[4][1] - hs[4][15] - hs[6][6])
    val c37 = (hs[3][0] + hs[3][2] - hs[3][5] - hs[3][8] + hs[4][9] + hs[4][11] + hs[4][12] - hs[5][11] + hs[5][12])
    val c38 = (hs[2][15] + hs[3][1] - hs[3][6] + hs[3][10] + hs[3][13] - hs[4][6] - hs[4][10] + hs[4][13] - hs[5][5] + hs[5][12])
    val c39 = (hs[2][15] + hs[3][0] - hs[3][5] - hs[3][6] + hs[3][7] - hs[3][8] + hs[3][10] + hs[3][11] - hs[3][12] + hs[4][9] + hs[4][12] + hs[4][13] - hs[5][5])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c18, c19, c21, c22, c23, c24, c25, c26, c27, c28, c29, c31, c32, c33, c34, c35, c36, c37, c38, c39)
}

fun multiply3By9MatrixAWith9By11MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 27) { "Matrix A must have 27 elements" }
    require(b.size == 99) { "Matrix B must have 99 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a16 = a[5]
    val a17 = a[6]
    val a18 = a[7]
    val a19 = a[8]
    val a21 = a[9]
    val a22 = a[10]
    val a23 = a[11]
    val a24 = a[12]
    val a25 = a[13]
    val a26 = a[14]
    val a27 = a[15]
    val a28 = a[16]
    val a29 = a[17]
    val a31 = a[18]
    val a32 = a[19]
    val a33 = a[20]
    val a34 = a[21]
    val a35 = a[22]
    val a36 = a[23]
    val a37 = a[24]
    val a38 = a[25]
    val a39 = a[26]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b18 = b[7]
    val b19 = b[8]
    val b110 = b[9]
    val b111 = b[10]
    val b21 = b[11]
    val b22 = b[12]
    val b23 = b[13]
    val b24 = b[14]
    val b25 = b[15]
    val b26 = b[16]
    val b27 = b[17]
    val b28 = b[18]
    val b29 = b[19]
    val b210 = b[20]
    val b211 = b[21]
    val b31 = b[22]
    val b32 = b[23]
    val b33 = b[24]
    val b34 = b[25]
    val b35 = b[26]
    val b36 = b[27]
    val b37 = b[28]
    val b38 = b[29]
    val b39 = b[30]
    val b310 = b[31]
    val b311 = b[32]
    val b41 = b[33]
    val b42 = b[34]
    val b43 = b[35]
    val b44 = b[36]
    val b45 = b[37]
    val b46 = b[38]
    val b47 = b[39]
    val b48 = b[40]
    val b49 = b[41]
    val b410 = b[42]
    val b411 = b[43]
    val b51 = b[44]
    val b52 = b[45]
    val b53 = b[46]
    val b54 = b[47]
    val b55 = b[48]
    val b56 = b[49]
    val b57 = b[50]
    val b58 = b[51]
    val b59 = b[52]
    val b510 = b[53]
    val b511 = b[54]
    val b61 = b[55]
    val b62 = b[56]
    val b63 = b[57]
    val b64 = b[58]
    val b65 = b[59]
    val b66 = b[60]
    val b67 = b[61]
    val b68 = b[62]
    val b69 = b[63]
    val b610 = b[64]
    val b611 = b[65]
    val b71 = b[66]
    val b72 = b[67]
    val b73 = b[68]
    val b74 = b[69]
    val b75 = b[70]
    val b76 = b[71]
    val b77 = b[72]
    val b78 = b[73]
    val b79 = b[74]
    val b710 = b[75]
    val b711 = b[76]
    val b81 = b[77]
    val b82 = b[78]
    val b83 = b[79]
    val b84 = b[80]
    val b85 = b[81]
    val b86 = b[82]
    val b87 = b[83]
    val b88 = b[84]
    val b89 = b[85]
    val b810 = b[86]
    val b811 = b[87]
    val b91 = b[88]
    val b92 = b[89]
    val b93 = b[90]
    val b94 = b[91]
    val b95 = b[92]
    val b96 = b[93]
    val b97 = b[94]
    val b98 = b[95]
    val b99 = b[96]
    val b910 = b[97]
    val b911 = b[98]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-a11 + a17 - a24 + a27 + a31 - a34),
                    (-a12 + a18 - a25 + a28 + a32 - a35),
                    (-a13 + a19 - a26 + a29 + a33 - a36),
                    (-a11 - a17 - a24 + a27 + a31 + a34),
                    (-a12 - a18 - a25 + a28 + a32 + a35),
                    (-a13 - a19 - a26 + a29 + a33 + a36),
                    (-a11 - a17 - a24 - a27 - a31 + a34),
                    (-a12 - a18 - a25 - a28 - a32 + a35),
                    (-a13 - a19 - a26 - a29 - a33 + a36),
                    (-a11 + a17 - a24 - a27 - a31 - a34),
                    (-a12 + a18 - a25 - a28 - a32 - a35),
                    (-a13 + a19 - a26 - a29 - a33 - a36),
                    (a11 + a17 - a24 + a27 - a31 - a34),
                    (a12 + a18 - a25 + a28 - a32 - a35),
                    (a13 + a19 - a26 + a29 - a33 - a36),
                    (a11 + a17 - a24 - a27 + a31 - a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a12 + a18 - a25 - a28 + a32 - a35),
                    (a13 + a19 - a26 - a29 + a33 - a36),
                    (a11 - a17 - a24 + a27 - a31 + a34),
                    (a12 - a18 - a25 + a28 - a32 + a35),
                    (a13 - a19 - a26 + a29 - a33 + a36),
                    (a11 - a17 - a24 - a27 + a31 + a34),
                    (a12 - a18 - a25 - a28 + a32 + a35),
                    (a13 - a19 - a26 - a29 + a33 + a36),
                    (a11 - a17 - a24 + a27 + a31 - a34),
                    (a12 - a18 - a25 + a28 + a32 - a35),
                    (a13 - a19 - a26 + a29 + a33 - a36),
                    (a11 + a17 - a24 - a27 - a31 + a34),
                    (a12 + a18 - a25 - a28 - a32 + a35),
                    (a13 + a19 - a26 - a29 - a33 + a36),
                    (-a11 - a17 + a24 - a27 - a31 - a34),
                    (-a12 - a18 + a25 - a28 - a32 - a35),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a13 - a19 + a26 - a29 - a33 - a36),
                    (a11 - a17 - a24 - a27 - a31 - a34),
                    (a12 - a18 - a25 - a28 - a32 - a35),
                    (a13 - a19 - a26 - a29 - a33 - a36),
                    (-a11 - a17 - a24 - a27 + a31 - a34),
                    (-a12 - a18 - a25 - a28 + a32 - a35),
                    (-a13 - a19 - a26 - a29 + a33 - a36),
                    (-a11 - a17 - a24 + a27 - a31 - a34),
                    (-a12 - a18 - a25 + a28 - a32 - a35),
                    (-a13 - a19 - a26 + a29 - a33 - a36),
                    (a11 - a17 + a24 + a27 - a31 - a34),
                    (a12 - a18 + a25 + a28 - a32 - a35),
                    (a13 - a19 + a26 + a29 - a33 - a36),
                    (a11 - a17 + a24 - a27 + a31 - a34),
                    (a12 - a18 + a25 - a28 + a32 - a35),
                    (a13 - a19 + a26 - a29 + a33 - a36),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a11 - a17 + a31 + a37),
                    (-a12 - a18 + a32 + a38),
                    (-a13 - a19 + a33 + a39),
                    (a11 + a17 + a31 + a37),
                    (a12 + a18 + a32 + a38),
                    (a13 + a19 + a33 + a39),
                    (-a11 + a17 - a31 + a37),
                    (-a12 + a18 - a32 + a38),
                    (-a13 + a19 - a33 + a39),
                    (-a11 + a17 + a31 - a37),
                    (-a12 + a18 + a32 - a38),
                    (-a13 + a19 + a33 - a39),
                    (-a11 - a14 + a31 + a34),
                    (-a12 - a15 + a32 + a35),
                    (-a13 - a16 + a33 + a36),
                    (-a11 + a14 - a31 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a12 + a15 - a32 + a35),
                    (-a13 + a16 - a33 + a36),
                    (-a11 - a14 - a31 - a34),
                    (-a12 - a15 - a32 - a35),
                    (-a13 - a16 - a33 - a36),
                    (a11 - a14 - a31 + a34),
                    (a12 - a15 - a32 + a35),
                    (a13 - a16 - a33 + a36),
                    (a21 + a24 + a31 + a34),
                    (a22 + a25 + a32 + a35),
                    (a23 + a26 + a33 + a36),
                    (a21 - a24 + a31 - a34),
                    (a22 - a25 + a32 - a35),
                    (a23 - a26 + a33 - a36),
                    (a21 - a24 - a31 + a34),
                    (a22 - a25 - a32 + a35),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a23 - a26 - a33 + a36),
                    (a21 + a24 - a31 - a34),
                    (a22 + a25 - a32 - a35),
                    (a23 + a26 - a33 - a36),
                    (a24 - a27 + a34 - a37),
                    (a25 - a28 + a35 - a38),
                    (a26 - a29 + a36 - a39),
                    (a24 + a27 + a34 + a37),
                    (a25 + a28 + a35 + a38),
                    (a26 + a29 + a36 + a39),
                    (a24 + a27 - a34 - a37),
                    (a25 + a28 - a35 - a38),
                    (a26 + a29 - a36 - a39),
                    (-a24 + a27 + a34 - a37),
                    (-a25 + a28 + a35 - a38),
                    (-a26 + a29 + a36 - a39),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a14 - a17 - a24 - a27),
                    (-a15 - a18 - a25 - a28),
                    (-a16 - a19 - a26 - a29),
                    (a14 - a17 - a24 + a27),
                    (a15 - a18 - a25 + a28),
                    (a16 - a19 - a26 + a29),
                    (-a14 + a17 - a24 + a27),
                    (-a15 + a18 - a25 + a28),
                    (-a16 + a19 - a26 + a29),
                    (a14 + a17 - a24 - a27),
                    (a15 + a18 - a25 - a28),
                    (a16 + a19 - a26 - a29),
                    (a11 - a17 - a21 + a27),
                    (a12 - a18 - a22 + a28),
                    (a13 - a19 - a23 + a29),
                    (a11 + a17 + a21 + a27),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a12 + a18 + a22 + a28),
                    (a13 + a19 + a23 + a29),
                    (-a11 + a17 - a21 + a27),
                    (-a12 + a18 - a22 + a28),
                    (-a13 + a19 - a23 + a29),
                    (a11 + a17 - a21 - a27),
                    (a12 + a18 - a22 - a28),
                    (a13 + a19 - a23 - a29),
                    -a22,
                    (-a22 - a32 - a34),
                    -a21,
                    (a11 + a14 - a21),
                    (a11 - a13 - a21),
                    (a11 - a13 - a21 + a22 + a23 + a32 + a33),
                    (a11 + a14 - a21 - a22 - a24 - a32 - a34),
                    -a12,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a12 + a32 + a34),
                    (a11 - a12 - a21),
                    (-a12 + a31),
                    a31,
                    (-a22 + a31 - a32),
                    (a11 - a13 - a31),
                    (a32 + a34),
                    (a11 - a13),
                    (a11 + a14),
                    (a11 + a14 - a31),
                    (-a22 - a32 - a33),
                    (-a12 + a32 + a33),
                    (-a12 - a14 + a32 + a34),
                    (-a12 - a14 + a23 - a24 + a32 + a34),
                    (a23 - a24),
                    (-a22 - a32),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a21 + a22 - a31 + a32),
                    (-a12 + a32),
                    (a11 - a21),
                    (-a12 - a14 - a24 + a32 + a34),
                    (a11 - a12),
                    a33,
                    (a11 - a31),
                    (a23 - a24 - a33 + a34),
                    a34,
                    a13,
                    a14,
                    (a32 + a33),
                    (a33 - a34),
                    ((a11 shl 1) - a13 + a14 - a23 + a24 - (a31 shl 1) + a33 - a34),
                    (a11 - a13 - a23 + a24 - a31 + a33),
                    (a13 - a14 + a23 - a24 - a33 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a23,
                    (a11 - (a12 shl 1) - a14 - a21 + a22 - a24 + a32 + a34),
                    (a11 - a13 - a23 - a31 + a33),
                    a24,
                    (a11 - a13 - a31 + a33),
                    (a13 + a14 - a23 + a24),
                    (a31 - a32),
                    (-a29 + a36 - a39),
                    (-a19 + a38 - a39),
                    (-a29 - a38),
                    (-a19 - a36),
                    (-a16 + a18 - a36),
                    (-a18 - a26),
                    (a19 - a35 + a39),
                    (a29 + a37 + a39),
                    (a36 + a38 - a39),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a19 + a37 + a39),
                    (a26 - a28 - a38),
                    (-a29 + a35 - a39),
                    (a35 - a39),
                    (a37 + a39),
                    -a29,
                    (a25 + a28 + a38),
                    (a15 + a16 + a25 + a28 + a35 + a36 + a38),
                    (a27 + a28 + a38),
                    (-a16 - a17 - a27 - a28 - a36 - a37 - a38),
                    (a18 + a28),
                    (a18 + a27 + a28),
                    (-a18 - a25 - a28),
                    (-a27 - a28),
                    (-a18 - a28 + a29),
                    -a18,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a25 + a28),
                    (-a17 - a18 - a19 - a27 - a28 - a37 - a39),
                    (a26 - a28 + a29),
                    (a16 - a19 + a26),
                    (a15 + a18 - a19 + a25 + a28 + a35 - a39),
                    (a15 + a16 + a25 + a26 - a29 + a35 - a39),
                    -a36,
                    a38,
                    (-a16 - a26),
                    (a16 + a17 + a26),
                    (a15 + a16 + a26),
                    -a26,
                    -a15,
                    (a16 + a17 + a26 + a27 + a29 + a37 + a39),
                    (-a16 - a17 - a36),
                    (-a16 - a17),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a15 + a16 + a36),
                    (-a25 - a26 - a27),
                    (a16 + a17 + a25 + a26 + a27),
                    a19,
                    (a18 - a19 + a28 - a29),
                    (a15 + a16 + a25 + a26 + a27),
                    -a37,
                    (-a35 + a37),
                    a27,
                    (a15 + a16),
                    -a17,
                    a26,
                    -a35,
                    a25,
                    a38,
                    a36,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a35 + a36),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-(((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) - b110 + (((b47 shr 1) shr 1) shr 1) + b410 - (((b411 shr 1) shr 1) shr 1) - b76 - (((b77 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1)),
                    (-(((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) - b210 + (((b57 shr 1) shr 1) shr 1) + b510 - (((b511 shr 1) shr 1) shr 1) - b86 - (((b87 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1)),
                    (-(((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) - b310 + (((b67 shr 1) shr 1) shr 1) + b610 - (((b611 shr 1) shr 1) shr 1) - b96 - (((b97 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1)),
                    (-(((b18 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) + b110 - (((b47 shr 1) shr 1) shr 1) + b410 - (((b411 shr 1) shr 1) shr 1) - b76 + (((b77 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1)),
                    (-(((b28 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) + b210 - (((b57 shr 1) shr 1) shr 1) + b510 - (((b511 shr 1) shr 1) shr 1) - b86 + (((b87 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1)),
                    (-(((b38 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) + b310 - (((b67 shr 1) shr 1) shr 1) + b610 - (((b611 shr 1) shr 1) shr 1) - b96 + (((b97 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1)),
                    (-(((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 + (((b47 shr 1) shr 1) shr 1) - b410 - (((b411 shr 1) shr 1) shr 1) - b76 + (((b77 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1)),
                    (-(((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 + (((b57 shr 1) shr 1) shr 1) - b510 - (((b511 shr 1) shr 1) shr 1) - b86 + (((b87 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1)),
                    (-(((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 + (((b67 shr 1) shr 1) shr 1) - b610 - (((b611 shr 1) shr 1) shr 1) - b96 + (((b97 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1)),
                    ((((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 + (((b47 shr 1) shr 1) shr 1) + b410 + (((b411 shr 1) shr 1) shr 1) + b76 + (((b77 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1)),
                    ((((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 + (((b57 shr 1) shr 1) shr 1) + b510 + (((b511 shr 1) shr 1) shr 1) + b86 + (((b87 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1)),
                    ((((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 + (((b67 shr 1) shr 1) shr 1) + b610 + (((b611 shr 1) shr 1) shr 1) + b96 + (((b97 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1)),
                    (-b16 + (((b18 shr 1) shr 1) shr 1) - (((b111 shr 1) shr 1) shr 1) - b46 + (((b47 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) + b710 - (((b711 shr 1) shr 1) shr 1)),
                    (-b26 + (((b28 shr 1) shr 1) shr 1) - (((b211 shr 1) shr 1) shr 1) - b56 + (((b57 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) + b810 - (((b811 shr 1) shr 1) shr 1)),
                    (-b36 + (((b38 shr 1) shr 1) shr 1) - (((b311 shr 1) shr 1) shr 1) - b66 + (((b67 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) + b910 - (((b911 shr 1) shr 1) shr 1)),
                    (-b16 + (((b18 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) + b46 - (((b47 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) + b710 + (((b711 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b26 + (((b28 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) + b56 - (((b57 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) + b810 + (((b811 shr 1) shr 1) shr 1)),
                    (-b36 + (((b38 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) + b66 - (((b67 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) + b910 + (((b911 shr 1) shr 1) shr 1)),
                    (-b16 - (((b18 shr 1) shr 1) shr 1) - (((b111 shr 1) shr 1) shr 1) + b46 + (((b47 shr 1) shr 1) shr 1) - (((b49 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - b710 + (((b711 shr 1) shr 1) shr 1)),
                    (-b26 - (((b28 shr 1) shr 1) shr 1) - (((b211 shr 1) shr 1) shr 1) + b56 + (((b57 shr 1) shr 1) shr 1) - (((b59 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - b810 + (((b811 shr 1) shr 1) shr 1)),
                    (-b36 - (((b38 shr 1) shr 1) shr 1) - (((b311 shr 1) shr 1) shr 1) + b66 + (((b67 shr 1) shr 1) shr 1) - (((b69 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - b910 + (((b911 shr 1) shr 1) shr 1)),
                    (b16 + (((b18 shr 1) shr 1) shr 1) - (((b111 shr 1) shr 1) shr 1) + b46 + (((b47 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) + (((b77 shr 1) shr 1) shr 1) + b710 + (((b711 shr 1) shr 1) shr 1)),
                    (b26 + (((b28 shr 1) shr 1) shr 1) - (((b211 shr 1) shr 1) shr 1) + b56 + (((b57 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) + (((b87 shr 1) shr 1) shr 1) + b810 + (((b811 shr 1) shr 1) shr 1)),
                    (b36 + (((b38 shr 1) shr 1) shr 1) - (((b311 shr 1) shr 1) shr 1) + b66 + (((b67 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) + (((b97 shr 1) shr 1) shr 1) + b910 + (((b911 shr 1) shr 1) shr 1)),
                    (-b16 - (((b17 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) + b46 + (((b48 shr 1) shr 1) shr 1) + (((b411 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) - b710),
                    (-b26 - (((b27 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) + b56 + (((b58 shr 1) shr 1) shr 1) + (((b511 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) - b810),
                    (-b36 - (((b37 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) + b66 + (((b68 shr 1) shr 1) shr 1) + (((b611 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) - b910),
                    (-b16 + (((b17 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b46 - (((b48 shr 1) shr 1) shr 1) - (((b411 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) + b710),
                    (-b26 + (((b27 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b56 - (((b58 shr 1) shr 1) shr 1) - (((b511 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) + b810),
                    (-b36 + (((b37 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b66 - (((b68 shr 1) shr 1) shr 1) - (((b611 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) + b910),
                    (-b16 + (((b17 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) - b46 + (((b48 shr 1) shr 1) shr 1) - (((b411 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1) + b710),
                    (-b26 + (((b27 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) - b56 + (((b58 shr 1) shr 1) shr 1) - (((b511 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1) + b810),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b36 + (((b37 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) - b66 + (((b68 shr 1) shr 1) shr 1) - (((b611 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1) + b910),
                    (b16 + (((b17 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) + b46 + (((b48 shr 1) shr 1) shr 1) - (((b411 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) + b710),
                    (b26 + (((b27 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) + b56 + (((b58 shr 1) shr 1) shr 1) - (((b511 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) + b810),
                    (b36 + (((b37 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) + b66 + (((b68 shr 1) shr 1) shr 1) - (((b611 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) + b910),
                    (-(((b17 shr 1) shr 1) shr 1) + b110 + (((b111 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - b410 - b76 + (((b78 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (-(((b27 shr 1) shr 1) shr 1) + b210 + (((b211 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - b510 - b86 + (((b88 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (-(((b37 shr 1) shr 1) shr 1) + b310 + (((b311 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - b610 - b96 + (((b98 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    ((((b17 shr 1) shr 1) shr 1) - b110 + (((b111 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - (((b49 shr 1) shr 1) shr 1) - b410 + b76 - (((b78 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    ((((b27 shr 1) shr 1) shr 1) - b210 + (((b211 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - (((b59 shr 1) shr 1) shr 1) - b510 + b86 - (((b88 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    ((((b37 shr 1) shr 1) shr 1) - b310 + (((b311 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - (((b69 shr 1) shr 1) shr 1) - b610 + b96 - (((b98 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (-(((b17 shr 1) shr 1) shr 1) - b110 - (((b111 shr 1) shr 1) shr 1) - (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - b410 - b76 - (((b78 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (-(((b27 shr 1) shr 1) shr 1) - b210 - (((b211 shr 1) shr 1) shr 1) - (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - b510 - b86 - (((b88 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (-(((b37 shr 1) shr 1) shr 1) - b310 - (((b311 shr 1) shr 1) shr 1) - (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - b610 - b96 - (((b98 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (-(((b17 shr 1) shr 1) shr 1) - b110 + (((b111 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) + b410 - b76 - (((b78 shr 1) shr 1) shr 1) - (((b711 shr 1) shr 1) shr 1)),
                    (-(((b27 shr 1) shr 1) shr 1) - b210 + (((b211 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) + b510 - b86 - (((b88 shr 1) shr 1) shr 1) - (((b811 shr 1) shr 1) shr 1)),
                    (-(((b37 shr 1) shr 1) shr 1) - b310 + (((b311 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) + b610 - b96 - (((b98 shr 1) shr 1) shr 1) - (((b911 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - b76 + (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1) - b710 + (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - b86 + (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1) - b810 + (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - b96 + (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1) - b910 + (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + b76 - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1) + b710 + (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + b86 - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1) + b810 + (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + b96 - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1) + b910 + (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - b76 - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) - b710 - (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - b86 - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) - b810 - (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - b96 - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) - b910 - (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + b76 + (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) + b710 - (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + b86 + (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) + b810 - (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + b96 + (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) + b910 - (((b911 shr 1) shr 1) shr 1)),
                    (-b16 - b110 - b46 + (((b47 shr 1) shr 1) shr 1) - (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - b410 + (((b411 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1)),
                    (-b26 - b210 - b56 + (((b57 shr 1) shr 1) shr 1) - (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - b510 + (((b511 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1)),
                    (-b36 - b310 - b66 + (((b67 shr 1) shr 1) shr 1) - (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - b610 + (((b611 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1)),
                    (-b16 - b110 + b46 - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) + b410 + (((b411 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b26 - b210 + b56 - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) + b510 + (((b511 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1)),
                    (-b36 - b310 + b66 - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) + b610 + (((b611 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1)),
                    (-b16 - b110 - b46 - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - (((b49 shr 1) shr 1) shr 1) - b410 - (((b411 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1)),
                    (-b26 - b210 - b56 - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - (((b59 shr 1) shr 1) shr 1) - b510 - (((b511 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1)),
                    (-b36 - b310 - b66 - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - (((b69 shr 1) shr 1) shr 1) - b610 - (((b611 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1)),
                    (b16 + b110 - b46 - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - b410 + (((b411 shr 1) shr 1) shr 1) + (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1)),
                    (b26 + b210 - b56 - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - b510 + (((b511 shr 1) shr 1) shr 1) + (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1)),
                    (b36 + b310 - b66 - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - b610 + (((b611 shr 1) shr 1) shr 1) + (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1)),
                    (-b16 - (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 - (((b111 shr 1) shr 1) shr 1) - b46 + b410 + (((b79 shr 1) shr 1) shr 1) - (((b711 shr 1) shr 1) shr 1)),
                    (-b26 - (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 - (((b211 shr 1) shr 1) shr 1) - b56 + b510 + (((b89 shr 1) shr 1) shr 1) - (((b811 shr 1) shr 1) shr 1)),
                    (-b36 - (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 - (((b311 shr 1) shr 1) shr 1) - b66 + b610 + (((b99 shr 1) shr 1) shr 1) - (((b911 shr 1) shr 1) shr 1)),
                    (-b16 - (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) + b110 + (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (-b26 - (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) + b210 + (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (-b36 - (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) + b310 + (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (-b16 + (((b17 shr 1) shr 1) shr 1) - (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 - (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (-b26 + (((b27 shr 1) shr 1) shr 1) - (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 - (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b36 + (((b37 shr 1) shr 1) shr 1) - (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 - (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (b16 - (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) - b110 - (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (b26 - (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) - b210 - (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (b36 - (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) - b310 - (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + b76 + (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) - b710 + (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + b86 + (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) - b810 + (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + b96 + (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) - b910 + (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - b76 - (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1) - (((b79 shr 1) shr 1) shr 1) + b710 + (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - b86 - (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1) - (((b89 shr 1) shr 1) shr 1) + b810 + (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - b96 - (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1) - (((b99 shr 1) shr 1) shr 1) + b910 + (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + b76 - (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1) - b710 - (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + b86 - (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1) - b810 - (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + b96 - (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1) - b910 - (((b911 shr 1) shr 1) shr 1)),
                    ((((b19 shr 1) shr 1) shr 1) + (((b111 shr 1) shr 1) shr 1) - (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - b76 + (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1) + (((b79 shr 1) shr 1) shr 1) + b710 - (((b711 shr 1) shr 1) shr 1)),
                    ((((b29 shr 1) shr 1) shr 1) + (((b211 shr 1) shr 1) shr 1) - (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - b86 + (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1) + (((b89 shr 1) shr 1) shr 1) + b810 - (((b811 shr 1) shr 1) shr 1)),
                    ((((b39 shr 1) shr 1) shr 1) + (((b311 shr 1) shr 1) shr 1) - (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - b96 + (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1) + (((b99 shr 1) shr 1) shr 1) + b910 - (((b911 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b16 + b110 + b46 + (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) - b410 - (((b411 shr 1) shr 1) shr 1) + (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1)),
                    (b26 + b210 + b56 + (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) - b510 - (((b511 shr 1) shr 1) shr 1) + (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1)),
                    (b36 + b310 + b66 + (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) - b610 - (((b611 shr 1) shr 1) shr 1) + (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1)),
                    (-b16 - b110 + b46 + (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - (((b49 shr 1) shr 1) shr 1) - b410 + (((b411 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1)),
                    (-b26 - b210 + b56 + (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - (((b59 shr 1) shr 1) shr 1) - b510 + (((b511 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1)),
                    (-b36 - b310 + b66 + (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - (((b69 shr 1) shr 1) shr 1) - b610 + (((b611 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1)),
                    (-b16 - b110 - b46 + (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) + (((b49 shr 1) shr 1) shr 1) + b410 - (((b411 shr 1) shr 1) shr 1) - (((b77 shr 1) shr 1) shr 1) - (((b78 shr 1) shr 1) shr 1)),
                    (-b26 - b210 - b56 + (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) + (((b59 shr 1) shr 1) shr 1) + b510 - (((b511 shr 1) shr 1) shr 1) - (((b87 shr 1) shr 1) shr 1) - (((b88 shr 1) shr 1) shr 1)),
                    (-b36 - b310 - b66 + (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) + (((b69 shr 1) shr 1) shr 1) + b610 - (((b611 shr 1) shr 1) shr 1) - (((b97 shr 1) shr 1) shr 1) - (((b98 shr 1) shr 1) shr 1)),
                    (b16 + b110 - b46 + (((b47 shr 1) shr 1) shr 1) + (((b48 shr 1) shr 1) shr 1) - (((b49 shr 1) shr 1) shr 1) + b410 + (((b411 shr 1) shr 1) shr 1) + (((b77 shr 1) shr 1) shr 1) + (((b78 shr 1) shr 1) shr 1)),
                    (b26 + b210 - b56 + (((b57 shr 1) shr 1) shr 1) + (((b58 shr 1) shr 1) shr 1) - (((b59 shr 1) shr 1) shr 1) + b510 + (((b511 shr 1) shr 1) shr 1) + (((b87 shr 1) shr 1) shr 1) + (((b88 shr 1) shr 1) shr 1)),
                    (b36 + b310 - b66 + (((b67 shr 1) shr 1) shr 1) + (((b68 shr 1) shr 1) shr 1) - (((b69 shr 1) shr 1) shr 1) + b610 + (((b611 shr 1) shr 1) shr 1) + (((b97 shr 1) shr 1) shr 1) + (((b98 shr 1) shr 1) shr 1)),
                    (b16 + (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 + (((b111 shr 1) shr 1) shr 1) - b46 + b410 + (((b79 shr 1) shr 1) shr 1) - (((b711 shr 1) shr 1) shr 1)),
                    (b26 + (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 + (((b211 shr 1) shr 1) shr 1) - b56 + b510 + (((b89 shr 1) shr 1) shr 1) - (((b811 shr 1) shr 1) shr 1)),
                    (b36 + (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 + (((b311 shr 1) shr 1) shr 1) - b66 + b610 + (((b99 shr 1) shr 1) shr 1) - (((b911 shr 1) shr 1) shr 1)),
                    (b16 - (((b17 shr 1) shr 1) shr 1) - (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) + b110 + (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b26 - (((b27 shr 1) shr 1) shr 1) - (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) + b210 + (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (b36 - (((b37 shr 1) shr 1) shr 1) - (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) + b310 + (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (b16 + (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) + (((b19 shr 1) shr 1) shr 1) + b110 - (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (b26 + (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) + (((b29 shr 1) shr 1) shr 1) + b210 - (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (b36 + (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) + (((b39 shr 1) shr 1) shr 1) + b310 - (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (-b16 + (((b17 shr 1) shr 1) shr 1) + (((b18 shr 1) shr 1) shr 1) - (((b19 shr 1) shr 1) shr 1) - b110 + (((b111 shr 1) shr 1) shr 1) + b46 - b410 - (((b79 shr 1) shr 1) shr 1) + (((b711 shr 1) shr 1) shr 1)),
                    (-b26 + (((b27 shr 1) shr 1) shr 1) + (((b28 shr 1) shr 1) shr 1) - (((b29 shr 1) shr 1) shr 1) - b210 + (((b211 shr 1) shr 1) shr 1) + b56 - b510 - (((b89 shr 1) shr 1) shr 1) + (((b811 shr 1) shr 1) shr 1)),
                    (-b36 + (((b37 shr 1) shr 1) shr 1) + (((b38 shr 1) shr 1) shr 1) - (((b39 shr 1) shr 1) shr 1) - b310 + (((b311 shr 1) shr 1) shr 1) + b66 - b610 - (((b99 shr 1) shr 1) shr 1) + (((b911 shr 1) shr 1) shr 1)),
                    (-b23 + b24 + b25),
                    (-b25 + b43),
                    (-b13 + b14 + b15),
                    (b15 - b43),
                    (-b14 - b33),
                    b33,
                    b43,
                    (b12 + b22 - b24 - b25),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b25 + b42 - b44),
                    (b12 + b22 + b23 - b24 - b25),
                    (b12 + b21),
                    (-b11 + b14 + b15 - b21),
                    (-b11 - b13 + b14 + b15 - b21),
                    (-b14 - b31 + b35),
                    -b25,
                    -b14,
                    b15,
                    (b15 + b35 - b41),
                    (-b24 + b33),
                    (-b24 + b32 + b44),
                    (-b42 + b44),
                    (-b32 - b44),
                    b35,
                    (-b11 - b13 + b14 + b15 - b21 - b23 + b33 + b43),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b11 - b13 + b14 + b15),
                    (b21 - b22 + b32 + b42),
                    (b12 + b13 + b22 + b23 - b24 - b25 + b33 - b43),
                    (-b22 - b23 + b24 + b25 - b32 - b42),
                    b12,
                    (-b24 - b31 + b34 + b35 + b44),
                    (-b11 + b12 - b31 + b41),
                    b44,
                    (-b25 - b41 + b45),
                    (-b14 + b32 - b34),
                    (b15 + b35 + b42 - b44 - b45),
                    -b24,
                    b44,
                    -b41,
                    (b35 + b41),
                    -b32,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b33 - b34 - b44),
                    (-b22 - b23 + b24 + b25),
                    (-b31 - b41),
                    (b35 - b43 + b45),
                    (-b31 + b35),
                    b35,
                    b21,
                    (b53 - b73 + b91 + b93),
                    (b54 - b74 - b91 + b94),
                    (b83 - b91),
                    (-b64 - b91),
                    (b51 - b61 + b64 + b71),
                    (b51 - b61 + b64 + b71 - b81),
                    (-b54 - b92),
                    (b73 + b95),
                    -b91,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b74 + b95),
                    (b51 + b71 - b81 - b83),
                    (-b53 - b92),
                    -b92,
                    b95,
                    (b83 - b92 + b93 + b95),
                    (b51 - b82),
                    b51,
                    (-b71 + b85),
                    -b71,
                    (-b54 - b74 + b83 + b84 - b92 + b93 + b94 + b95),
                    (b74 + b85),
                    (-b54 + b82),
                    b85,
                    (b83 - b92 + b93 + b94 + b95),
                    (b51 - b61 + b64 + b71 - b81 - b82 + b84 + b85),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b82,
                    b74,
                    -b83,
                    b64,
                    -b54,
                    b53,
                    (b53 + b63 - b73 + b93),
                    (-b54 + b74 - b81 + b82 - b84 + b85 - b91 - b94),
                    (-b53 + b63 + b64 - b73),
                    (b55 - b65 - b73),
                    (-b53 + b62 - b72),
                    (b51 + b55 - b61 + b62 - b63 - b65 + b71 - b72 - b81 - b83),
                    (b52 - b54 - b62 + b72),
                    -b73,
                    (-b55 + b65 - b71 + b72),
                    (b55 - b65 - b72),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b51 - b55 - b62 + b72),
                    (b55 - b72),
                    -b72,
                    (b64 - b92 + b94 + b95),
                    (-b92 + b93 + b94 + b95),
                    b55,
                    (b71 - b72 - b75 + b95),
                    -b72,
                    (b55 - b73 - b75 + b85),
                    (b55 + b62 - b72),
                    (b55 - b65 + b74 + b75),
                    (b51 - b61 + b71 - b81),
                    (b51 - b52 - b55 - b92),
                    (b52 - b53 - b72 - b82),
                    (-b81 + b82 + b85 - b91),
                    (-b61 + b62 + b65 - b91),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b55 - b72),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3], lefts[4] * rights[4], lefts[5] * rights[5], lefts[6] * rights[6], lefts[7] * rights[7], lefts[8] * rights[8], lefts[9] * rights[9], lefts[10] * rights[10], lefts[11] * rights[11], lefts[12] * rights[12], lefts[13] * rights[13], lefts[14] * rights[14])

    val c11 = (-hs[8][2] - hs[8][3] + hs[8][5] - hs[8][7] + hs[8][8] - hs[8][9] + hs[8][14] + hs[9][4] - hs[9][6] + hs[10][5] + hs[10][10] + hs[10][11] + hs[10][12] + hs[12][3] + hs[12][14] + hs[12][15] + hs[13][0] + hs[13][9] + hs[13][11] + hs[13][15])
    val c12 = (-hs[7][10] + hs[8][4] + hs[8][14] + hs[9][0] + hs[9][14] + hs[10][2] - hs[10][4] + hs[10][6] + hs[10][9] + hs[11][1] + hs[11][6] - hs[11][8] - hs[11][13] + hs[12][0] - hs[12][2] - hs[13][11] + hs[13][14])
    val c13 = (-hs[8][3] + hs[8][5] - hs[8][7] + hs[8][8] - hs[8][9] + (hs[8][14] shl 1) + hs[9][13] + hs[9][14] - hs[10][4] + hs[10][5] - hs[10][6] + hs[10][15] - hs[11][6] + hs[11][7] + hs[11][8] + hs[11][9] + hs[11][13] - hs[12][0] - hs[12][14] - hs[12][15] - hs[13][0] - hs[13][9] - hs[13][14] - hs[13][15])
    val c14 = (-hs[7][15] + hs[8][0] - hs[8][6] + hs[8][11] + hs[8][12] + hs[8][13] + hs[9][4] + hs[9][7] - hs[9][11] - hs[9][15] - hs[10][13] - hs[11][3] - hs[11][12] - hs[12][0] - hs[12][4] - hs[12][12] - hs[13][1] - hs[13][2] - hs[13][5] + hs[13][9])
    val c15 = (-hs[7][8] - hs[8][1] - hs[8][12] - hs[8][13] + hs[9][3] + hs[9][4] - hs[9][7] - hs[9][12] - hs[10][1] + hs[11][2] - hs[11][3] + hs[12][0] + hs[12][5] + hs[12][10] + hs[13][1] + hs[13][5] - hs[13][9] + hs[13][13])
    val c16 = (-hs[7][15] + hs[8][0] + hs[8][2] - hs[8][6] + hs[8][11] - hs[9][1] - hs[9][11] - hs[9][12] - hs[10][6] + hs[11][3] - hs[11][6] + hs[11][7] - hs[12][0] - hs[13][0] - hs[13][7] - hs[13][9] + hs[13][12] - hs[14][0])
    val c17 = (hs[7][10] - hs[7][11] + hs[7][12] + hs[7][15] - hs[8][1] - hs[8][7] + hs[8][8] + hs[9][2] - hs[11][14] - hs[12][2] - hs[12][3] - hs[12][8] - hs[12][9] - hs[12][10] - hs[12][11] - hs[12][15] - hs[13][1] - hs[13][2] - hs[13][3] - hs[13][4] - hs[13][5] + hs[13][9] - hs[13][11])
    val c18 = (hs[7][8] + hs[7][9] + hs[7][10] - hs[7][11] + hs[7][12] + hs[7][13] - hs[7][14] + hs[8][6] - hs[8][7] + hs[8][8] + hs[8][10] + hs[9][11] - hs[10][14] + hs[11][2] - hs[11][3] + hs[11][4] - hs[11][5] + hs[12][2] + hs[12][5] + hs[12][9] + hs[12][10] + hs[12][11] - hs[12][13] + hs[12][15] + hs[13][1] + hs[13][2] + hs[13][5] - hs[13][9] + hs[13][11])
    val c19 = (-hs[7][8] - hs[7][9] + hs[8][3] - hs[8][4] - hs[8][6] - hs[8][10] + hs[8][15] - hs[9][11] - hs[10][7] - hs[10][9] + hs[10][14] - hs[10][15] - hs[11][2] + hs[11][3] - hs[11][4] + hs[11][5] - hs[12][6])
    val c110 = (-hs[8][7] + hs[8][11] + hs[8][13] + hs[9][7] - hs[9][9] - hs[9][11] - hs[9][15] + hs[10][12] - hs[10][13] - hs[11][0] - hs[11][3] + hs[11][4] - hs[11][11] - hs[11][12] - hs[11][13] - hs[11][15] - hs[12][0] - hs[12][1] + hs[12][3] - hs[12][4] + hs[13][3] + hs[13][11])
    val c111 = (hs[7][12] + hs[7][13] - hs[8][7] + hs[8][10] - hs[9][7] + hs[9][11] - hs[9][12] - hs[10][0] - hs[10][12] + hs[11][5] + hs[11][10] + hs[11][11] + hs[11][12] + hs[11][13] + hs[11][14] + hs[11][15] + hs[12][0] - hs[13][11])
    val c21 = (hs[8][5] - hs[8][7] + hs[9][5] - hs[9][11] - hs[9][12] - hs[10][4] - hs[10][8] + hs[10][10] + hs[10][13] + hs[10][15] + hs[11][0] + hs[11][3] - hs[11][4] - hs[12][7] - hs[13][3] + hs[13][14])
    val c22 = (hs[8][0] - hs[8][6] + hs[8][8] + hs[8][12] + hs[8][14] - hs[9][10] + hs[10][5] + hs[11][0] - hs[11][4] + hs[11][11] + hs[11][13] + hs[12][1] + hs[12][15] + hs[13][1] + hs[13][2] + hs[13][5] - hs[13][10])
    val c23 = (hs[7][9] - hs[7][11] - hs[7][14] + hs[8][6] + hs[8][8] + hs[8][14] + hs[10][3] + hs[10][14] - hs[11][4] - hs[11][13] - hs[12][9] + hs[12][13] - hs[12][15] - hs[13][1] - hs[13][2] - hs[13][8])
    val c24 = (-hs[8][6] + hs[8][8] - hs[8][9] + (hs[8][14] shl 1) + hs[9][8] + hs[9][13] + hs[9][14] + hs[10][5] + hs[11][4] + hs[11][8] + hs[11][9] + hs[11][13] - hs[12][14] - hs[12][15] + hs[13][6] + hs[13][7] + hs[14][0])
    val c25 = (-(((hs[0][0] shr 1) shr 1) shr 1) - (((hs[0][1] shr 1) shr 1) shr 1) - (((hs[0][2] shr 1) shr 1) shr 1) + (((hs[0][3] shr 1) shr 1) shr 1) + (((hs[0][4] shr 1) shr 1) shr 1) + (((hs[0][5] shr 1) shr 1) shr 1) + (((hs[0][6] shr 1) shr 1) shr 1) + (((hs[0][7] shr 1) shr 1) shr 1) + (((hs[0][8] shr 1) shr 1) shr 1) + (((hs[0][9] shr 1) shr 1) shr 1) + (((hs[0][10] shr 1) shr 1) shr 1) + (((hs[0][11] shr 1) shr 1) shr 1) + (((hs[2][4] shr 1) shr 1) shr 1) + (((hs[2][5] shr 1) shr 1) shr 1) + (((hs[2][6] shr 1) shr 1) shr 1) - (((hs[2][7] shr 1) shr 1) shr 1) - (((hs[2][8] shr 1) shr 1) shr 1) - (((hs[2][9] shr 1) shr 1) shr 1) + (((hs[2][10] shr 1) shr 1) shr 1) + (((hs[2][11] shr 1) shr 1) shr 1) + (((hs[2][12] shr 1) shr 1) shr 1) + (((hs[2][13] shr 1) shr 1) shr 1) + (((hs[2][14] shr 1) shr 1) shr 1) + (((hs[2][15] shr 1) shr 1) shr 1) + (((hs[3][12] shr 1) shr 1) shr 1) + (((hs[3][13] shr 1) shr 1) shr 1) + (((hs[3][14] shr 1) shr 1) shr 1) + (((hs[3][15] shr 1) shr 1) shr 1) + (((hs[4][0] shr 1) shr 1) shr 1) + (((hs[4][1] shr 1) shr 1) shr 1) + (((hs[4][2] shr 1) shr 1) shr 1) + (((hs[4][3] shr 1) shr 1) shr 1) + (((hs[4][4] shr 1) shr 1) shr 1) + (((hs[4][5] shr 1) shr 1) shr 1) + (((hs[4][6] shr 1) shr 1) shr 1) + (((hs[4][7] shr 1) shr 1) shr 1) + (((hs[4][8] shr 1) shr 1) shr 1) + (((hs[4][9] shr 1) shr 1) shr 1) + (((hs[4][10] shr 1) shr 1) shr 1) - (((hs[4][11] shr 1) shr 1) shr 1) - (((hs[4][12] shr 1) shr 1) shr 1) - (((hs[4][13] shr 1) shr 1) shr 1) - (((hs[4][14] shr 1) shr 1) shr 1) - (((hs[4][15] shr 1) shr 1) shr 1) - (((hs[5][0] shr 1) shr 1) shr 1) - (((hs[5][1] shr 1) shr 1) shr 1) - (((hs[5][2] shr 1) shr 1) shr 1) - (((hs[5][3] shr 1) shr 1) shr 1) - (((hs[6][0] shr 1) shr 1) shr 1) - (((hs[6][1] shr 1) shr 1) shr 1) - (((hs[6][2] shr 1) shr 1) shr 1) + (((hs[6][3] shr 1) shr 1) shr 1) + (((hs[6][4] shr 1) shr 1) shr 1) + (((hs[6][5] shr 1) shr 1) shr 1) + (((hs[6][6] shr 1) shr 1) shr 1) + (((hs[6][7] shr 1) shr 1) shr 1) + (((hs[6][8] shr 1) shr 1) shr 1) - (((hs[6][9] shr 1) shr 1) shr 1) - (((hs[6][10] shr 1) shr 1) shr 1) - (((hs[6][11] shr 1) shr 1) shr 1) + (((hs[6][12] shr 1) shr 1) shr 1) + (((hs[6][13] shr 1) shr 1) shr 1) + (((hs[6][14] shr 1) shr 1) shr 1) + (((hs[6][15] shr 1) shr 1) shr 1) + (((hs[7][0] shr 1) shr 1) shr 1) + (((hs[7][1] shr 1) shr 1) shr 1) - (((hs[7][2] shr 1) shr 1) shr 1) - (((hs[7][3] shr 1) shr 1) shr 1) - (((hs[7][4] shr 1) shr 1) shr 1) - (((hs[7][5] shr 1) shr 1) shr 1) - (((hs[7][6] shr 1) shr 1) shr 1) - (((hs[7][7] shr 1) shr 1) shr 1))
    val c26 = (-(((hs[0][0] shr 1) shr 1) shr 1) - (((hs[0][1] shr 1) shr 1) shr 1) - (((hs[0][2] shr 1) shr 1) shr 1) - (((hs[0][3] shr 1) shr 1) shr 1) - (((hs[0][4] shr 1) shr 1) shr 1) - (((hs[0][5] shr 1) shr 1) shr 1) + (((hs[0][6] shr 1) shr 1) shr 1) + (((hs[0][7] shr 1) shr 1) shr 1) + (((hs[0][8] shr 1) shr 1) shr 1) - (((hs[0][9] shr 1) shr 1) shr 1) - (((hs[0][10] shr 1) shr 1) shr 1) - (((hs[0][11] shr 1) shr 1) shr 1) + (((hs[2][4] shr 1) shr 1) shr 1) + (((hs[2][5] shr 1) shr 1) shr 1) + (((hs[2][6] shr 1) shr 1) shr 1) + (((hs[2][7] shr 1) shr 1) shr 1) + (((hs[2][8] shr 1) shr 1) shr 1) + (((hs[2][9] shr 1) shr 1) shr 1) - (((hs[2][10] shr 1) shr 1) shr 1) - (((hs[2][11] shr 1) shr 1) shr 1) - (((hs[2][12] shr 1) shr 1) shr 1) + (((hs[2][13] shr 1) shr 1) shr 1) + (((hs[2][14] shr 1) shr 1) shr 1) + (((hs[2][15] shr 1) shr 1) shr 1) - (((hs[3][12] shr 1) shr 1) shr 1) - (((hs[3][13] shr 1) shr 1) shr 1) - (((hs[3][14] shr 1) shr 1) shr 1) + (((hs[3][15] shr 1) shr 1) shr 1) + (((hs[4][0] shr 1) shr 1) shr 1) + (((hs[4][1] shr 1) shr 1) shr 1) - (((hs[4][2] shr 1) shr 1) shr 1) - (((hs[4][3] shr 1) shr 1) shr 1) - (((hs[4][4] shr 1) shr 1) shr 1) + (((hs[4][5] shr 1) shr 1) shr 1) + (((hs[4][6] shr 1) shr 1) shr 1) + (((hs[4][7] shr 1) shr 1) shr 1) - (((hs[4][8] shr 1) shr 1) shr 1) - (((hs[4][9] shr 1) shr 1) shr 1) - (((hs[4][10] shr 1) shr 1) shr 1) - (((hs[4][11] shr 1) shr 1) shr 1) - (((hs[4][12] shr 1) shr 1) shr 1) - (((hs[4][13] shr 1) shr 1) shr 1) - (((hs[4][14] shr 1) shr 1) shr 1) - (((hs[4][15] shr 1) shr 1) shr 1) - (((hs[5][0] shr 1) shr 1) shr 1) + (((hs[5][1] shr 1) shr 1) shr 1) + (((hs[5][2] shr 1) shr 1) shr 1) + (((hs[5][3] shr 1) shr 1) shr 1) - (((hs[6][0] shr 1) shr 1) shr 1) - (((hs[6][1] shr 1) shr 1) shr 1) - (((hs[6][2] shr 1) shr 1) shr 1) - (((hs[6][3] shr 1) shr 1) shr 1) - (((hs[6][4] shr 1) shr 1) shr 1) - (((hs[6][5] shr 1) shr 1) shr 1) + (((hs[6][6] shr 1) shr 1) shr 1) + (((hs[6][7] shr 1) shr 1) shr 1) + (((hs[6][8] shr 1) shr 1) shr 1) + (((hs[6][9] shr 1) shr 1) shr 1) + (((hs[6][10] shr 1) shr 1) shr 1) + (((hs[6][11] shr 1) shr 1) shr 1) - (((hs[6][12] shr 1) shr 1) shr 1) - (((hs[6][13] shr 1) shr 1) shr 1) - (((hs[6][14] shr 1) shr 1) shr 1) + (((hs[6][15] shr 1) shr 1) shr 1) + (((hs[7][0] shr 1) shr 1) shr 1) + (((hs[7][1] shr 1) shr 1) shr 1) - (((hs[7][2] shr 1) shr 1) shr 1) - (((hs[7][3] shr 1) shr 1) shr 1) - (((hs[7][4] shr 1) shr 1) shr 1) + (((hs[7][5] shr 1) shr 1) shr 1) + (((hs[7][6] shr 1) shr 1) shr 1) + (((hs[7][7] shr 1) shr 1) shr 1))
    val c27 = ((((hs[0][12] shr 1) shr 1) shr 1) + (((hs[0][13] shr 1) shr 1) shr 1) + (((hs[0][14] shr 1) shr 1) shr 1) - (((hs[0][15] shr 1) shr 1) shr 1) - (((hs[1][0] shr 1) shr 1) shr 1) - (((hs[1][1] shr 1) shr 1) shr 1) + (((hs[1][2] shr 1) shr 1) shr 1) + (((hs[1][3] shr 1) shr 1) shr 1) + (((hs[1][4] shr 1) shr 1) shr 1) + (((hs[1][5] shr 1) shr 1) shr 1) + (((hs[1][6] shr 1) shr 1) shr 1) + (((hs[1][7] shr 1) shr 1) shr 1) - (((hs[1][8] shr 1) shr 1) shr 1) - (((hs[1][9] shr 1) shr 1) shr 1) - (((hs[1][10] shr 1) shr 1) shr 1) + (((hs[1][11] shr 1) shr 1) shr 1) + (((hs[1][12] shr 1) shr 1) shr 1) + (((hs[1][13] shr 1) shr 1) shr 1) + (((hs[1][14] shr 1) shr 1) shr 1) + (((hs[1][15] shr 1) shr 1) shr 1) + (((hs[2][0] shr 1) shr 1) shr 1) - (((hs[2][1] shr 1) shr 1) shr 1) - (((hs[2][2] shr 1) shr 1) shr 1) - (((hs[2][3] shr 1) shr 1) shr 1) - (((hs[3][0] shr 1) shr 1) shr 1) - (((hs[3][1] shr 1) shr 1) shr 1) - (((hs[3][2] shr 1) shr 1) shr 1) + (((hs[3][3] shr 1) shr 1) shr 1) + (((hs[3][4] shr 1) shr 1) shr 1) + (((hs[3][5] shr 1) shr 1) shr 1) - (((hs[3][6] shr 1) shr 1) shr 1) - (((hs[3][7] shr 1) shr 1) shr 1) - (((hs[3][8] shr 1) shr 1) shr 1) - (((hs[3][9] shr 1) shr 1) shr 1) - (((hs[3][10] shr 1) shr 1) shr 1) - (((hs[3][11] shr 1) shr 1) shr 1) - (((hs[5][4] shr 1) shr 1) shr 1) - (((hs[5][5] shr 1) shr 1) shr 1) - (((hs[5][6] shr 1) shr 1) shr 1) - (((hs[5][7] shr 1) shr 1) shr 1) - (((hs[5][8] shr 1) shr 1) shr 1) - (((hs[5][9] shr 1) shr 1) shr 1) - (((hs[5][10] shr 1) shr 1) shr 1) - (((hs[5][11] shr 1) shr 1) shr 1) - (((hs[5][12] shr 1) shr 1) shr 1) + (((hs[5][13] shr 1) shr 1) shr 1) + (((hs[5][14] shr 1) shr 1) shr 1) + (((hs[5][15] shr 1) shr 1) shr 1))
    val c28 = (-hs[1][8] - hs[1][9] - hs[1][10] + hs[1][11] + hs[1][12] + hs[1][13] - hs[1][14] - hs[1][15] - hs[2][0] + hs[2][1] + hs[2][2] + hs[2][3] + hs[2][4] + hs[2][5] + hs[2][6] - hs[2][7] - hs[2][8] - hs[2][9] - hs[2][10] - hs[2][11] - hs[2][12] - hs[2][13] - hs[2][14] - hs[2][15] - hs[3][0] - hs[3][1] - hs[3][2] - hs[3][3] - hs[3][4] - hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] + hs[3][9] + hs[3][10] + hs[3][11] - hs[3][12] - hs[3][13] - hs[3][14] - hs[3][15] - hs[4][0] - hs[4][1] + hs[4][2] + hs[4][3] + hs[4][4] + hs[4][5] + hs[4][6] + hs[4][7] - hs[5][4] - hs[5][5] - hs[5][6] + hs[5][7] + hs[5][8] + hs[5][9] - hs[5][10] - hs[5][11] - hs[5][12] - hs[5][13] - hs[5][14] - hs[5][15] - hs[6][0] - hs[6][1] - hs[6][2] + hs[6][3] + hs[6][4] + hs[6][5] - hs[6][6] - hs[6][7] - hs[6][8] + hs[6][9] + hs[6][10] + hs[6][11])
    val c29 = (-hs[0][0] - hs[0][1] - hs[0][2] + hs[0][3] + hs[0][4] + hs[0][5] - hs[0][6] - hs[0][7] - hs[0][8] - hs[0][9] - hs[0][10] - hs[0][11] - hs[0][12] - hs[0][13] - hs[0][14] + hs[0][15] + hs[1][0] + hs[1][1] - hs[1][2] - hs[1][3] - hs[1][4] - hs[1][5] - hs[1][6] - hs[1][7] - hs[4][8] - hs[4][9] - hs[4][10] - hs[4][11] - hs[4][12] - hs[4][13] + hs[4][14] + hs[4][15] + hs[5][0] - hs[5][1] - hs[5][2] - hs[5][3] - hs[6][12] - hs[6][13] - hs[6][14] - hs[6][15] - hs[7][0] - hs[7][1] - hs[7][2] - hs[7][3] - hs[7][4] - hs[7][5] - hs[7][6] - hs[7][7])
    val c210 = (-hs[1][8] - hs[1][9] - hs[1][10] - hs[1][11] - hs[1][12] - hs[1][13] - hs[1][14] - hs[1][15] - hs[2][0] - hs[2][1] - hs[2][2] - hs[2][3] - hs[2][4] - hs[2][5] - hs[2][6] - hs[2][7] - hs[2][8] - hs[2][9] + hs[2][10] + hs[2][11] + hs[2][12] - hs[2][13] - hs[2][14] - hs[2][15] + hs[3][0] + hs[3][1] + hs[3][2] - hs[3][3] - hs[3][4] - hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] - hs[3][9] - hs[3][10] - hs[3][11] + hs[3][12] + hs[3][13] + hs[3][14] - hs[3][15] - hs[4][0] - hs[4][1] + hs[4][2] + hs[4][3] + hs[4][4] - hs[4][5] - hs[4][6] - hs[4][7] - hs[5][4] - hs[5][5] - hs[5][6] - hs[5][7] - hs[5][8] - hs[5][9] + hs[5][10] + hs[5][11] + hs[5][12] - hs[5][13] - hs[5][14] - hs[5][15] + hs[6][0] + hs[6][1] + hs[6][2] + hs[6][3] + hs[6][4] + hs[6][5] - hs[6][6] - hs[6][7] - hs[6][8] - hs[6][9] - hs[6][10] - hs[6][11])
    val c211 = (hs[0][0] + hs[0][1] + hs[0][2] + hs[0][3] + hs[0][4] + hs[0][5] + hs[0][6] + hs[0][7] + hs[0][8] - hs[0][9] - hs[0][10] - hs[0][11] + hs[0][12] + hs[0][13] + hs[0][14] + hs[0][15] + hs[1][0] + hs[1][1] - hs[1][2] - hs[1][3] - hs[1][4] + hs[1][5] + hs[1][6] + hs[1][7] - hs[3][0] - hs[3][1] - hs[3][2] - hs[3][3] - hs[3][4] - hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] + hs[3][9] + hs[3][10] + hs[3][11] + hs[3][12] + hs[3][13] + hs[3][14] + hs[3][15] + hs[4][0] + hs[4][1] - hs[4][2] - hs[4][3] - hs[4][4] - hs[4][5] - hs[4][6] - hs[4][7] - hs[5][4] - hs[5][5] - hs[5][6] + hs[5][7] + hs[5][8] + hs[5][9] - hs[5][10] - hs[5][11] - hs[5][12] - hs[5][13] - hs[5][14] - hs[5][15] - hs[6][0] - hs[6][1] - hs[6][2] + hs[6][3] + hs[6][4] + hs[6][5] - hs[6][6] - hs[6][7] - hs[6][8] + hs[6][9] + hs[6][10] + hs[6][11])
    val c31 = (-hs[1][8] - hs[1][9] - hs[1][10] + hs[1][11] + hs[1][12] + hs[1][13] + hs[1][14] + hs[1][15] + hs[2][0] - hs[2][1] - hs[2][2] - hs[2][3] - hs[2][4] - hs[2][5] - hs[2][6] - hs[2][7] - hs[2][8] - hs[2][9] - hs[2][10] - hs[2][11] - hs[2][12] + hs[2][13] + hs[2][14] + hs[2][15] + hs[4][8] + hs[4][9] + hs[4][10] + hs[4][11] + hs[4][12] + hs[4][13] - hs[4][14] - hs[4][15] - hs[5][0] + hs[5][1] + hs[5][2] + hs[5][3] - hs[6][12] - hs[6][13] - hs[6][14] - hs[6][15] - hs[7][0] - hs[7][1] - hs[7][2] - hs[7][3] - hs[7][4] - hs[7][5] - hs[7][6] - hs[7][7])
    val c32 = (-hs[0][0] - hs[0][1] - hs[0][2] - hs[0][3] - hs[0][4] - hs[0][5] + hs[0][6] + hs[0][7] + hs[0][8] - hs[0][9] - hs[0][10] - hs[0][11] - hs[0][12] - hs[0][13] - hs[0][14] + hs[0][15] + hs[1][0] + hs[1][1] + hs[1][2] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][6] + hs[1][7] + hs[3][0] + hs[3][1] + hs[3][2] - hs[3][3] - hs[3][4] - hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] - hs[3][9] - hs[3][10] - hs[3][11] - hs[3][12] - hs[3][13] - hs[3][14] + hs[3][15] + hs[4][0] + hs[4][1] - hs[4][2] - hs[4][3] - hs[4][4] + hs[4][5] + hs[4][6] + hs[4][7] + hs[5][4] + hs[5][5] + hs[5][6] + hs[5][7] + hs[5][8] + hs[5][9] - hs[5][10] - hs[5][11] - hs[5][12] + hs[5][13] + hs[5][14] + hs[5][15] - hs[6][0] - hs[6][1] - hs[6][2] - hs[6][3] - hs[6][4] - hs[6][5] + hs[6][6] + hs[6][7] + hs[6][8] + hs[6][9] + hs[6][10] + hs[6][11])
    val c33 = (hs[0][0] + hs[0][1] + hs[0][2] - hs[0][3] - hs[0][4] - hs[0][5] + hs[0][6] + hs[0][7] + hs[0][8] + hs[0][9] + hs[0][10] + hs[0][11] + hs[1][8] + hs[1][9] + hs[1][10] - hs[1][11] - hs[1][12] - hs[1][13] - hs[1][14] - hs[1][15] - hs[2][0] + hs[2][1] + hs[2][2] + hs[2][3] - hs[3][12] - hs[3][13] - hs[3][14] + hs[3][15] + hs[4][0] + hs[4][1] + hs[4][2] + hs[4][3] + hs[4][4] - hs[4][5] - hs[4][6] - hs[4][7] - hs[6][0] - hs[6][1] - hs[6][2] - hs[6][3] - hs[6][4] - hs[6][5] - hs[6][6] - hs[6][7] - hs[6][8] - hs[6][9] - hs[6][10] - hs[6][11])
    val c34 = (-hs[0][12] - hs[0][13] - hs[0][14] - hs[0][15] - hs[1][0] - hs[1][1] + hs[1][2] + hs[1][3] + hs[1][4] - hs[1][5] - hs[1][6] - hs[1][7] - hs[2][4] - hs[2][5] - hs[2][6] + hs[2][7] + hs[2][8] + hs[2][9] + hs[2][10] + hs[2][11] + hs[2][12] + hs[2][13] + hs[2][14] + hs[2][15] + hs[3][0] + hs[3][1] + hs[3][2] + hs[3][3] + hs[3][4] + hs[3][5] + hs[3][6] + hs[3][7] + hs[3][8] - hs[3][9] - hs[3][10] - hs[3][11] - hs[4][8] - hs[4][9] - hs[4][10] + hs[4][11] + hs[4][12] + hs[4][13] - hs[4][14] - hs[4][15] - hs[5][0] - hs[5][1] - hs[5][2] - hs[5][3] + hs[5][4] + hs[5][5] + hs[5][6] - hs[5][7] - hs[5][8] - hs[5][9] + hs[5][10] + hs[5][11] + hs[5][12] + hs[5][13] + hs[5][14] + hs[5][15] + hs[6][12] + hs[6][13] + hs[6][14] - hs[6][15] - hs[7][0] - hs[7][1] - hs[7][2] - hs[7][3] - hs[7][4] + hs[7][5] + hs[7][6] + hs[7][7])
    val c35 = (-hs[0][12] - hs[0][13] - hs[0][14] - hs[0][15] - hs[1][0] - hs[1][1] - hs[1][2] - hs[1][3] - hs[1][4] + hs[1][5] + hs[1][6] + hs[1][7] - hs[2][4] - hs[2][5] - hs[2][6] + hs[2][7] + hs[2][8] + hs[2][9] - hs[2][10] - hs[2][11] - hs[2][12] - hs[2][13] - hs[2][14] - hs[2][15] + hs[3][0] + hs[3][1] + hs[3][2] + hs[3][3] + hs[3][4] + hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] + hs[3][9] + hs[3][10] + hs[3][11] - hs[4][8] - hs[4][9] - hs[4][10] + hs[4][11] + hs[4][12] + hs[4][13] + hs[4][14] + hs[4][15] + hs[5][0] + hs[5][1] + hs[5][2] + hs[5][3] + hs[5][4] + hs[5][5] + hs[5][6] - hs[5][7] - hs[5][8] - hs[5][9] - hs[5][10] - hs[5][11] - hs[5][12] - hs[5][13] - hs[5][14] - hs[5][15] - hs[6][12] - hs[6][13] - hs[6][14] - hs[6][15] - hs[7][0] - hs[7][1] + hs[7][2] + hs[7][3] + hs[7][4] + hs[7][5] + hs[7][6] + hs[7][7])
    val c36 = ((((hs[0][12] shr 1) shr 1) shr 1) + (((hs[0][13] shr 1) shr 1) shr 1) + (((hs[0][14] shr 1) shr 1) shr 1) + (((hs[0][15] shr 1) shr 1) shr 1) + (((hs[1][0] shr 1) shr 1) shr 1) + (((hs[1][1] shr 1) shr 1) shr 1) + (((hs[1][2] shr 1) shr 1) shr 1) + (((hs[1][3] shr 1) shr 1) shr 1) + (((hs[1][4] shr 1) shr 1) shr 1) - (((hs[1][5] shr 1) shr 1) shr 1) - (((hs[1][6] shr 1) shr 1) shr 1) - (((hs[1][7] shr 1) shr 1) shr 1) + (((hs[1][8] shr 1) shr 1) shr 1) + (((hs[1][9] shr 1) shr 1) shr 1) + (((hs[1][10] shr 1) shr 1) shr 1) + (((hs[1][11] shr 1) shr 1) shr 1) + (((hs[1][12] shr 1) shr 1) shr 1) + (((hs[1][13] shr 1) shr 1) shr 1) - (((hs[1][14] shr 1) shr 1) shr 1) - (((hs[1][15] shr 1) shr 1) shr 1) - (((hs[2][0] shr 1) shr 1) shr 1) - (((hs[2][1] shr 1) shr 1) shr 1) - (((hs[2][2] shr 1) shr 1) shr 1) - (((hs[2][3] shr 1) shr 1) shr 1) + (((hs[3][12] shr 1) shr 1) shr 1) + (((hs[3][13] shr 1) shr 1) shr 1) + (((hs[3][14] shr 1) shr 1) shr 1) + (((hs[3][15] shr 1) shr 1) shr 1) + (((hs[4][0] shr 1) shr 1) shr 1) + (((hs[4][1] shr 1) shr 1) shr 1) + (((hs[4][2] shr 1) shr 1) shr 1) + (((hs[4][3] shr 1) shr 1) shr 1) + (((hs[4][4] shr 1) shr 1) shr 1) + (((hs[4][5] shr 1) shr 1) shr 1) + (((hs[4][6] shr 1) shr 1) shr 1) + (((hs[4][7] shr 1) shr 1) shr 1) + (((hs[4][8] shr 1) shr 1) shr 1) + (((hs[4][9] shr 1) shr 1) shr 1) + (((hs[4][10] shr 1) shr 1) shr 1) - (((hs[4][11] shr 1) shr 1) shr 1) - (((hs[4][12] shr 1) shr 1) shr 1) - (((hs[4][13] shr 1) shr 1) shr 1) - (((hs[4][14] shr 1) shr 1) shr 1) - (((hs[4][15] shr 1) shr 1) shr 1) - (((hs[5][0] shr 1) shr 1) shr 1) - (((hs[5][1] shr 1) shr 1) shr 1) - (((hs[5][2] shr 1) shr 1) shr 1) - (((hs[5][3] shr 1) shr 1) shr 1) + (((hs[6][0] shr 1) shr 1) shr 1) + (((hs[6][1] shr 1) shr 1) shr 1) + (((hs[6][2] shr 1) shr 1) shr 1) - (((hs[6][3] shr 1) shr 1) shr 1) - (((hs[6][4] shr 1) shr 1) shr 1) - (((hs[6][5] shr 1) shr 1) shr 1) - (((hs[6][6] shr 1) shr 1) shr 1) - (((hs[6][7] shr 1) shr 1) shr 1) - (((hs[6][8] shr 1) shr 1) shr 1) + (((hs[6][9] shr 1) shr 1) shr 1) + (((hs[6][10] shr 1) shr 1) shr 1) + (((hs[6][11] shr 1) shr 1) shr 1) + (((hs[6][12] shr 1) shr 1) shr 1) + (((hs[6][13] shr 1) shr 1) shr 1) + (((hs[6][14] shr 1) shr 1) shr 1) + (((hs[6][15] shr 1) shr 1) shr 1) + (((hs[7][0] shr 1) shr 1) shr 1) + (((hs[7][1] shr 1) shr 1) shr 1) - (((hs[7][2] shr 1) shr 1) shr 1) - (((hs[7][3] shr 1) shr 1) shr 1) - (((hs[7][4] shr 1) shr 1) shr 1) - (((hs[7][5] shr 1) shr 1) shr 1) - (((hs[7][6] shr 1) shr 1) shr 1) - (((hs[7][7] shr 1) shr 1) shr 1))
    val c37 = ((((hs[0][12] shr 1) shr 1) shr 1) + (((hs[0][13] shr 1) shr 1) shr 1) + (((hs[0][14] shr 1) shr 1) shr 1) - (((hs[0][15] shr 1) shr 1) shr 1) - (((hs[1][0] shr 1) shr 1) shr 1) - (((hs[1][1] shr 1) shr 1) shr 1) - (((hs[1][2] shr 1) shr 1) shr 1) - (((hs[1][3] shr 1) shr 1) shr 1) - (((hs[1][4] shr 1) shr 1) shr 1) - (((hs[1][5] shr 1) shr 1) shr 1) - (((hs[1][6] shr 1) shr 1) shr 1) - (((hs[1][7] shr 1) shr 1) shr 1) - (((hs[1][8] shr 1) shr 1) shr 1) - (((hs[1][9] shr 1) shr 1) shr 1) - (((hs[1][10] shr 1) shr 1) shr 1) - (((hs[1][11] shr 1) shr 1) shr 1) - (((hs[1][12] shr 1) shr 1) shr 1) - (((hs[1][13] shr 1) shr 1) shr 1) - (((hs[1][14] shr 1) shr 1) shr 1) - (((hs[1][15] shr 1) shr 1) shr 1) - (((hs[2][0] shr 1) shr 1) shr 1) - (((hs[2][1] shr 1) shr 1) shr 1) - (((hs[2][2] shr 1) shr 1) shr 1) - (((hs[2][3] shr 1) shr 1) shr 1) + (((hs[3][12] shr 1) shr 1) shr 1) + (((hs[3][13] shr 1) shr 1) shr 1) + (((hs[3][14] shr 1) shr 1) shr 1) - (((hs[3][15] shr 1) shr 1) shr 1) - (((hs[4][0] shr 1) shr 1) shr 1) - (((hs[4][1] shr 1) shr 1) shr 1) + (((hs[4][2] shr 1) shr 1) shr 1) + (((hs[4][3] shr 1) shr 1) shr 1) + (((hs[4][4] shr 1) shr 1) shr 1) - (((hs[4][5] shr 1) shr 1) shr 1) - (((hs[4][6] shr 1) shr 1) shr 1) - (((hs[4][7] shr 1) shr 1) shr 1) + (((hs[4][8] shr 1) shr 1) shr 1) + (((hs[4][9] shr 1) shr 1) shr 1) + (((hs[4][10] shr 1) shr 1) shr 1) + (((hs[4][11] shr 1) shr 1) shr 1) + (((hs[4][12] shr 1) shr 1) shr 1) + (((hs[4][13] shr 1) shr 1) shr 1) + (((hs[4][14] shr 1) shr 1) shr 1) + (((hs[4][15] shr 1) shr 1) shr 1) + (((hs[5][0] shr 1) shr 1) shr 1) - (((hs[5][1] shr 1) shr 1) shr 1) - (((hs[5][2] shr 1) shr 1) shr 1) - (((hs[5][3] shr 1) shr 1) shr 1) + (((hs[6][0] shr 1) shr 1) shr 1) + (((hs[6][1] shr 1) shr 1) shr 1) + (((hs[6][2] shr 1) shr 1) shr 1) + (((hs[6][3] shr 1) shr 1) shr 1) + (((hs[6][4] shr 1) shr 1) shr 1) + (((hs[6][5] shr 1) shr 1) shr 1) - (((hs[6][6] shr 1) shr 1) shr 1) - (((hs[6][7] shr 1) shr 1) shr 1) - (((hs[6][8] shr 1) shr 1) shr 1) - (((hs[6][9] shr 1) shr 1) shr 1) - (((hs[6][10] shr 1) shr 1) shr 1) - (((hs[6][11] shr 1) shr 1) shr 1) - (((hs[6][12] shr 1) shr 1) shr 1) - (((hs[6][13] shr 1) shr 1) shr 1) - (((hs[6][14] shr 1) shr 1) shr 1) + (((hs[6][15] shr 1) shr 1) shr 1) + (((hs[7][0] shr 1) shr 1) shr 1) + (((hs[7][1] shr 1) shr 1) shr 1) - (((hs[7][2] shr 1) shr 1) shr 1) - (((hs[7][3] shr 1) shr 1) shr 1) - (((hs[7][4] shr 1) shr 1) shr 1) + (((hs[7][5] shr 1) shr 1) shr 1) + (((hs[7][6] shr 1) shr 1) shr 1) + (((hs[7][7] shr 1) shr 1) shr 1))
    val c38 = (-(((hs[0][0] shr 1) shr 1) shr 1) - (((hs[0][1] shr 1) shr 1) shr 1) - (((hs[0][2] shr 1) shr 1) shr 1) + (((hs[0][3] shr 1) shr 1) shr 1) + (((hs[0][4] shr 1) shr 1) shr 1) + (((hs[0][5] shr 1) shr 1) shr 1) - (((hs[0][6] shr 1) shr 1) shr 1) - (((hs[0][7] shr 1) shr 1) shr 1) - (((hs[0][8] shr 1) shr 1) shr 1) - (((hs[0][9] shr 1) shr 1) shr 1) - (((hs[0][10] shr 1) shr 1) shr 1) - (((hs[0][11] shr 1) shr 1) shr 1) + (((hs[2][4] shr 1) shr 1) shr 1) + (((hs[2][5] shr 1) shr 1) shr 1) + (((hs[2][6] shr 1) shr 1) shr 1) + (((hs[2][7] shr 1) shr 1) shr 1) + (((hs[2][8] shr 1) shr 1) shr 1) + (((hs[2][9] shr 1) shr 1) shr 1) + (((hs[2][10] shr 1) shr 1) shr 1) + (((hs[2][11] shr 1) shr 1) shr 1) + (((hs[2][12] shr 1) shr 1) shr 1) - (((hs[2][13] shr 1) shr 1) shr 1) - (((hs[2][14] shr 1) shr 1) shr 1) - (((hs[2][15] shr 1) shr 1) shr 1) - (((hs[3][0] shr 1) shr 1) shr 1) - (((hs[3][1] shr 1) shr 1) shr 1) - (((hs[3][2] shr 1) shr 1) shr 1) + (((hs[3][3] shr 1) shr 1) shr 1) + (((hs[3][4] shr 1) shr 1) shr 1) + (((hs[3][5] shr 1) shr 1) shr 1) - (((hs[3][6] shr 1) shr 1) shr 1) - (((hs[3][7] shr 1) shr 1) shr 1) - (((hs[3][8] shr 1) shr 1) shr 1) - (((hs[3][9] shr 1) shr 1) shr 1) - (((hs[3][10] shr 1) shr 1) shr 1) - (((hs[3][11] shr 1) shr 1) shr 1) + (((hs[5][4] shr 1) shr 1) shr 1) + (((hs[5][5] shr 1) shr 1) shr 1) + (((hs[5][6] shr 1) shr 1) shr 1) + (((hs[5][7] shr 1) shr 1) shr 1) + (((hs[5][8] shr 1) shr 1) shr 1) + (((hs[5][9] shr 1) shr 1) shr 1) + (((hs[5][10] shr 1) shr 1) shr 1) + (((hs[5][11] shr 1) shr 1) shr 1) + (((hs[5][12] shr 1) shr 1) shr 1) - (((hs[5][13] shr 1) shr 1) shr 1) - (((hs[5][14] shr 1) shr 1) shr 1) - (((hs[5][15] shr 1) shr 1) shr 1))
    val c39 = (-hs[0][12] - hs[0][13] - hs[0][14] + hs[0][15] + hs[1][0] + hs[1][1] - hs[1][2] - hs[1][3] - hs[1][4] - hs[1][5] - hs[1][6] - hs[1][7] - hs[2][4] - hs[2][5] - hs[2][6] - hs[2][7] - hs[2][8] - hs[2][9] - hs[2][10] - hs[2][11] - hs[2][12] + hs[2][13] + hs[2][14] + hs[2][15] - hs[3][12] - hs[3][13] - hs[3][14] + hs[3][15] + hs[4][0] + hs[4][1] + hs[4][2] + hs[4][3] + hs[4][4] - hs[4][5] - hs[4][6] - hs[4][7] + hs[6][0] + hs[6][1] + hs[6][2] + hs[6][3] + hs[6][4] + hs[6][5] + hs[6][6] + hs[6][7] + hs[6][8] + hs[6][9] + hs[6][10] + hs[6][11])
    val c310 = (hs[0][0] + hs[0][1] + hs[0][2] + hs[0][3] + hs[0][4] + hs[0][5] + hs[0][6] + hs[0][7] + hs[0][8] - hs[0][9] - hs[0][10] - hs[0][11] - hs[1][8] - hs[1][9] - hs[1][10] + hs[1][11] + hs[1][12] + hs[1][13] - hs[1][14] - hs[1][15] - hs[2][0] + hs[2][1] + hs[2][2] + hs[2][3] - hs[3][0] - hs[3][1] - hs[3][2] - hs[3][3] - hs[3][4] - hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] + hs[3][9] + hs[3][10] + hs[3][11] - hs[4][8] - hs[4][9] - hs[4][10] + hs[4][11] + hs[4][12] + hs[4][13] - hs[4][14] - hs[4][15] - hs[5][0] - hs[5][1] - hs[5][2] - hs[5][3] - hs[5][4] - hs[5][5] - hs[5][6] + hs[5][7] + hs[5][8] + hs[5][9] - hs[5][10] - hs[5][11] - hs[5][12] - hs[5][13] - hs[5][14] - hs[5][15] - hs[6][12] - hs[6][13] - hs[6][14] + hs[6][15] + hs[7][0] + hs[7][1] + hs[7][2] + hs[7][3] + hs[7][4] - hs[7][5] - hs[7][6] - hs[7][7])
    val c311 = (hs[0][0] + hs[0][1] + hs[0][2] - hs[0][3] - hs[0][4] - hs[0][5] - hs[0][6] - hs[0][7] - hs[0][8] - hs[0][9] - hs[0][10] - hs[0][11] - hs[1][8] - hs[1][9] - hs[1][10] - hs[1][11] - hs[1][12] - hs[1][13] + hs[1][14] + hs[1][15] + hs[2][0] + hs[2][1] + hs[2][2] + hs[2][3] + hs[3][0] + hs[3][1] + hs[3][2] + hs[3][3] + hs[3][4] + hs[3][5] - hs[3][6] - hs[3][7] - hs[3][8] + hs[3][9] + hs[3][10] + hs[3][11] - hs[4][8] - hs[4][9] - hs[4][10] + hs[4][11] + hs[4][12] + hs[4][13] + hs[4][14] + hs[4][15] + hs[5][0] + hs[5][1] + hs[5][2] + hs[5][3] - hs[5][4] - hs[5][5] - hs[5][6] + hs[5][7] + hs[5][8] + hs[5][9] + hs[5][10] + hs[5][11] + hs[5][12] + hs[5][13] + hs[5][14] + hs[5][15] - hs[6][12] - hs[6][13] - hs[6][14] - hs[6][15] - hs[7][0] - hs[7][1] + hs[7][2] + hs[7][3] + hs[7][4] + hs[7][5] + hs[7][6] + hs[7][7])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c18, c19, c110, c111, c21, c22, c23, c24, c25, c26, c27, c28, c29, c210, c211, c31, c32, c33, c34, c35, c36, c37, c38, c39, c310, c311)
}

fun multiply4By4MatrixAWith4By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 16) { "Matrix A must have 16 elements" }
    require(b.size == 16) { "Matrix B must have 16 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val a31 = a[8]
    val a32 = a[9]
    val a33 = a[10]
    val a34 = a[11]
    val a41 = a[12]
    val a42 = a[13]
    val a43 = a[14]
    val a44 = a[15]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b21 = b[4]
    val b22 = b[5]
    val b23 = b[6]
    val b24 = b[7]
    val b31 = b[8]
    val b32 = b[9]
    val b33 = b[10]
    val b34 = b[11]
    val b41 = b[12]
    val b42 = b[13]
    val b43 = b[14]
    val b44 = b[15]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (a11 + a31),
                    (a11 - a13 + a31),
                    -a13,
                    -a33,
                    -a31,
                    (a11 - a13 + a31 - a33),
                    (-a21 + a22 - a23 - a24),
                    (-a21 + a22 - a23 - a24 - a41 + a42),
                    (a11 - a13),
                    (-a21 + a22 - a41 + a42),
                    (a41 - a42),
                    (-a21 + a22 - a23 - a24 - a41 + a42 - a43 - a44),
                    (-a23 - a24),
                    (a11 - a12 + a21 - a22),
                    (-a12 - a14),
                    (a12 + a14 - a21 + a22 + a23 + a24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a41 - a42),
                    (a12 - a21 + a22 + a32 + a41 - a42),
                    (a14 + a23 + a24),
                    (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a34 + a41 - a42 - a43 - a44),
                    (a32 + a41 - a42),
                    (a12 + a14 + a22 + a24),
                    (a12 + a14 + a22 + a24 + a32 - a42),
                    (a14 + a24),
                    (a12 + a14 + a22 + a24 + a32 + a34 - a42 - a44),
                    (a32 - a42),
                    (a34 - a44),
                    (a34 - a43 - a44),
                    (a14 + a34),
                    (a13 + a14 + a23 + a24 + a33 + a34 - a43 - a44),
                    (a11 - a12 - a13 - a14 + a21 - a22 - a23 - a24 + a31 - a32 - a33 - a34 - a41 + a42 + a43 + a44),
                    -a43,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a14,
                    (a14 - a32),
                    (a13 + a14 + a23 + a24 - a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44),
                    (-a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44),
                    (-a12 - a32),
                    (a32 + a34),
                    (-a13 - a14 - a23 - a24),
                    a32,
                    -a21,
                    (-a21 + a41),
                    (-a21 + a41 - a43),
                    (a12 + a22 + a32 - a42),
                    (-a21 + a23 + a41 - a43),
                    (-a31 + a32 + a41 - a42),
                    (a41 - a43),
                    (-a43 - a44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a23,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b11 + b31),
                    (b11 - b13 + b31),
                    (b11 - b13 + b31 - b33),
                    -b33,
                    -b13,
                    -b31,
                    (-b21 + b22 - b23 - b24),
                    (-b21 + b22 - b23 - b24 - b41 + b42),
                    (b11 - b13),
                    (-b21 + b22 - b41 + b42),
                    (-b23 - b24),
                    (b41 - b42),
                    (-b21 + b22 - b23 - b24 - b41 + b42 - b43 - b44),
                    (-b12 - b14),
                    -b21,
                    (b12 + b14 - b21 + b22 + b23 + b24),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b41 - b42),
                    (b12 - b21 + b22 + b32 + b41 - b42),
                    (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b34 + b41 - b42 - b43 - b44),
                    (b32 + b41 - b42),
                    (b14 + b23 + b24),
                    (b12 + b14 + b22 + b24),
                    (b12 + b14 + b22 + b24 + b32 - b42),
                    (b12 + b14 + b22 + b24 + b32 + b34 - b42 - b44),
                    (b32 - b42),
                    (b14 + b24),
                    (b34 - b44),
                    (b34 - b43 - b44),
                    -b43,
                    (b14 + b34),
                    b14,
                    (b13 + b14 + b23 + b24 + b33 + b34 - b43 - b44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b21 + b41),
                    (-b21 + b41 - b43),
                    (b14 - b32),
                    b32,
                    -b23,
                    (b41 - b43),
                    (b32 + b34),
                    (-b21 + b23 + b41 - b43),
                    (b11 - b12 + b21 - b22),
                    (b11 - b12 - b13 - b14 + b21 - b22 - b23 - b24 + b31 - b32 - b33 - b34 - b41 + b42 + b43 + b44),
                    (b13 + b14 + b23 + b24 - b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44),
                    (b12 + b22 + b32 - b42),
                    (-b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44),
                    (-b12 - b32),
                    (-b13 - b14 - b23 - b24),
                    (-b43 - b44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b31 + b32 + b41 - b42),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3])

    val c11 = (hs[0][0] - hs[0][1] - hs[0][4] + hs[0][8] + hs[0][14] + hs[2][0])
    val c12 = (-hs[0][14] - hs[0][15] + hs[1][0] - hs[1][1] - hs[1][4] + hs[1][5] - hs[1][6] + hs[1][9] - hs[2][0] - hs[2][8] + hs[2][11] + hs[3][0])
    val c13 = (hs[0][1] + hs[0][4] + hs[0][5] - hs[0][8] - hs[1][12] - hs[2][0] + hs[2][1] + hs[2][5])
    val c14 = (-hs[0][15] + hs[1][0] - hs[1][3] - hs[1][4] + hs[1][5] - hs[1][6] + hs[1][8] + hs[1][9] - hs[1][12] - hs[1][15] - hs[2][0] + hs[2][1] + hs[2][5] - hs[2][8] + hs[2][9] + hs[2][10])
    val c21 = (-hs[0][6] + hs[0][7] - hs[0][9] + hs[0][10] - hs[0][13] + hs[0][14] + hs[0][15] - hs[1][0] + hs[1][1] + hs[1][4] - hs[1][14] + hs[2][0] - hs[2][2] - hs[2][3])
    val c22 = (hs[0][6] - hs[0][7] + hs[0][9] - hs[0][10] - hs[0][14] - hs[0][15] + hs[1][0] - hs[1][1] - hs[1][4] + hs[1][5] - hs[1][6] + hs[1][9] - hs[2][0] + hs[2][11])
    val c23 = (-hs[0][6] + hs[0][7] + hs[0][10] + hs[0][11] - hs[0][15] + hs[1][0] - hs[1][3] - hs[1][4] - hs[1][12] - hs[2][0] + hs[2][1] + hs[2][3] + hs[2][5] + hs[2][13])
    val c24 = (-hs[0][6] + hs[0][7] + hs[0][10] + hs[0][11] - hs[0][15] + hs[1][0] - hs[1][3] - hs[1][4] + hs[1][5] - hs[1][6] + hs[1][8] + hs[1][9] - hs[1][12] - hs[2][0] + hs[2][1] + hs[2][5])
    val c31 = (hs[0][0] - hs[0][1] + hs[0][2] - hs[0][4] + hs[2][0] - hs[2][1] + hs[2][4] - hs[2][7])
    val c32 = (hs[1][0] - hs[1][1] - hs[1][2] - hs[1][4] - hs[1][6] + hs[1][7] + hs[1][9] - hs[2][0] + hs[2][1] - hs[2][4] + hs[2][7] - hs[2][10] + hs[2][11] + hs[2][12] - hs[2][14] + hs[3][0])
    val c33 = (hs[0][3] + hs[0][4] - hs[1][12] - hs[2][0] + hs[2][1] + hs[2][7])
    val c34 = (-hs[1][4] + hs[1][9] - hs[1][10] + hs[1][11] - hs[1][12] - hs[1][15] - hs[2][0] + hs[2][1] + hs[2][7] - hs[2][14])
    val c41 = (hs[0][7] - hs[0][9] + hs[0][10] - hs[0][12] + hs[1][0] - hs[1][1] - hs[1][2] - hs[1][4] + hs[1][14] - hs[2][0] + hs[2][1] + hs[2][2] + hs[2][3] - hs[2][4] - hs[2][6] + hs[2][7])
    val c42 = (-hs[0][7] + hs[0][9] - hs[0][10] + hs[0][12] - hs[1][0] + hs[1][1] + hs[1][2] + hs[1][4] + hs[1][6] - hs[1][7] - hs[1][9] + hs[2][0] - hs[2][1] + hs[2][4] - hs[2][7] - hs[2][11])
    val c43 = (hs[0][10] + hs[1][4] - hs[1][11] + hs[1][12] + hs[1][13] + hs[2][0] - hs[2][1] - hs[2][2] - hs[2][3] + hs[2][6] - hs[2][7] + hs[2][15])
    val c44 = (hs[0][10] + hs[1][4] - hs[1][9] + hs[1][10] - hs[1][11] + hs[1][12] + hs[2][0] - hs[2][1] - hs[2][7] + hs[2][15])

    return intArrayOf(c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44)
}

fun multiply4By4MatrixAWith4By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 16) { "Matrix A must have 16 elements" }
    require(b.size == 20) { "Matrix B must have 20 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a21 = a[4]
    val a22 = a[5]
    val a23 = a[6]
    val a24 = a[7]
    val a31 = a[8]
    val a32 = a[9]
    val a33 = a[10]
    val a34 = a[11]
    val a41 = a[12]
    val a42 = a[13]
    val a43 = a[14]
    val a44 = a[15]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a13,
                    (-a11 + a13 - a41),
                    (-a11 + a13 - a31),
                    (-a11 + a13),
                    (-a11 + a13 - a31 + a33 - a34),
                    a12,
                    (a12 + a22 + a24),
                    (-a11 - a14 - a41),
                    (-a11 - a14 - a31),
                    (a22 + a24 + a42),
                    (a22 + a24 + a32),
                    (a22 + a24),
                    (-a11 - a14 - a22 - a24 - a31 - a32 - a34),
                    (-a11 - a14 - a22 - a24 - a31 - a32 - a34 - a41 - a42 - a44),
                    -a34,
                    (-a11 - a12 + a13 - a14 - a22 - a24 - a31 + a33 - a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a21,
                    -a24,
                    (-a11 - a14),
                    (-a11 + a13 - a31 + a32 + a33),
                    (a22 + a32),
                    -a32,
                    (-a21 + a22 + a32),
                    -a42,
                    (-a21 + a23 - a42),
                    (-a11 + a13 - a41 + a42 + a43),
                    (-a21 + a23 - a32),
                    (-a21 + a23),
                    (a12 - a21 + a22 + a23),
                    (-a21 + a23 - a41 + a43),
                    (-a21 + a23 + a33),
                    (-a11 + a13 - a31 + a33 - a34 + a42 + a44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a21 + a23 + a33 + a42 + a43),
                    (-a21 + a23 + a32 + a33 + a43),
                    (-a11 - a21),
                    (-a21 + a22 + a42),
                    (-a11 - a41),
                    a12,
                    (-a11 - a12 - a31),
                    (-a21 + a23 + a43),
                    -a31,
                    (a13 + a21),
                    (-a21 + a22 - a31 + a32),
                    (-a31 + a33 - a34),
                    (a22 - a23 + a32 - a33),
                    a44,
                    (-a11 - a14 - a21),
                    (-a11 - a31),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a11 - a14 - (a21 shl 1) + a23 - a24 - a31 + a33 - a34),
                    (a22 + a42),
                    (-a11 - a14 - a22 - a24 - a34 - a41 - a42 - a44),
                    (-a11 - a14 - a23 - a24 + a41 - a43 - a44),
                    (a22 - a23 + a42 - a43),
                    (-a21 + a23 - a31 + a33 - a34),
                    (a12 + a22),
                    (a41 - a43),
                    (a12 + a13 + a22 + a23),
                    (-a11 - a14 - a22 - a24 - a41 - a42 - a44),
                    (-a11 - a14 - a22 - a24 - a31 - a32 - a34 - a44),
                    (-a21 + a23 + a33 + a43),
                    -a14,
                    -a41,
                    (a12 + a21),
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (-b15 - b35),
                    (-b15 + b33 - b35),
                    (-b15 - b32 - b35),
                    (-b15 - b31),
                    (b31 + b45),
                    (b21 - b25),
                    (b21 - b45),
                    (-b11 - b31 - b43),
                    (-b11 - b31 + b42),
                    (b21 + b43),
                    (b21 - b42),
                    b21,
                    b42,
                    (-b11 + b12 - b13 - b31),
                    (b31 + b41 - b42),
                    b45,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b11 - b14 - b24 - b31 - b34),
                    (-b21 + b41 + b44),
                    (-b11 - b31),
                    (-b32 - b35 - b45),
                    (-b11 + b12 - b14 + b22 - b24 - b31 + b32 - b34 - b42),
                    (b21 - b22 - b25 + b32 + b35 + b45),
                    (-b11 + b12 - b14 - b24 - b31 + b32 - b34),
                    (-b21 - b23 + b25 + b33 - b35 - b45),
                    (b25 + b33 - b35 - b45),
                    (b33 - b35),
                    (b25 - b32 - b35 - b45),
                    (b25 - b35 - b45),
                    (b25 + b31 + b34 - b35 - b45),
                    (-b11 - b13 - b14 - b31 - b44),
                    -b32,
                    b45,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b11 + b12 - b13 + b21 - b22 + b23 - b25 - b31 - b33 + b35 + b45),
                    (-b11 + b12 - b13 + b21 - b22 + b23 - b25 - b31 + b32 + b35 + b45),
                    (-b14 + b15 + b31 + b44),
                    (-b11 - b13 - b14 - b24 - b31 - b33 - b34),
                    (b13 - b15 + b33 - b35 - b43),
                    (b21 + b23 - b25),
                    (-b22 + b23),
                    -b33,
                    (-b11 + b12 - b31),
                    (-b15 - b31 - b34),
                    (-b11 + b12 - b14 - b31 + b44),
                    b31,
                    (-b32 + b34 + b44),
                    (b41 + b43),
                    (-b11 - b31 - b44),
                    (b12 + b15 - b22 + b23 + b32 + b35 - b42),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    b44,
                    (-b11 - b13 - b14 - b23 - b24 - b31 - b33 - b34 + b43),
                    (-b11 + b12 - b13 - b31 + b43),
                    b44,
                    (-b11 - b13 - b14 - b31 - b33 - b34),
                    (b31 - b44),
                    (-b24 + b25 + b31 + b34 - b35 - b45),
                    b31,
                    (b31 + b34 - b35),
                    b43,
                    (-b11 + b12 - b13 - b31 - b42),
                    (-b11 + b12 - b13 + b21 - b22 + b23 - b25 - b31 + b32 - b33 + b35 + b45),
                    (-b11 - b31 + b41 - b45),
                    (-b11 - b13 - b31),
                    -b24,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3])

    val c11 = (hs[0][4] + hs[0][6] - hs[0][11] - hs[0][15] + hs[1][2] - hs[2][11] - hs[3][12])
    val c12 = (hs[0][11] - hs[1][1] - hs[1][2] - hs[2][11] + hs[2][14] + hs[3][0] + hs[3][5])
    val c13 = (hs[0][8] + hs[0][10] - hs[0][11] - hs[0][12] - hs[0][14] - hs[1][2] + hs[2][11])
    val c14 = (hs[0][7] + hs[0][9] - hs[0][11] - hs[1][2] + hs[2][13] - hs[3][7] + hs[3][9])
    val c15 = (hs[0][0] - hs[0][2] - hs[0][5] - hs[0][8] + hs[1][2] + hs[2][5] + hs[2][6] + hs[2][8] - hs[2][15])
    val c21 = (-hs[0][10] + hs[0][11] + hs[1][0] + hs[1][4] - hs[1][5] - hs[1][6] - hs[1][10] + hs[1][11])
    val c22 = (hs[0][8] + hs[0][10] - hs[0][11] - hs[0][12] - hs[1][2] + hs[1][5] + hs[1][10] - hs[1][11] - hs[1][14] - hs[2][8])
    val c23 = (hs[0][12] - hs[0][13] + hs[1][7] + hs[1][14] - hs[2][0] + hs[3][10] + hs[3][11] + hs[3][13])
    val c24 = (-hs[0][0] + hs[0][1] - hs[0][5] + hs[0][7] - hs[1][2] - hs[2][4] + hs[2][5] - hs[3][13])
    val c25 = (hs[0][9] - hs[0][11] - hs[1][0] - hs[1][7] + hs[1][8] - hs[1][11] + hs[2][3] - hs[3][1])
    val c31 = (hs[0][13] + hs[1][5] + hs[2][1] + hs[2][7] - hs[2][8] - hs[3][2] + hs[3][9] - hs[3][11])
    val c32 = (-hs[0][7] - hs[0][9] + hs[0][11] + hs[1][2] + hs[1][7] - hs[1][8] + hs[1][11] - hs[2][7] - hs[3][9] + hs[3][13])
    val c33 = (hs[0][3] - hs[1][0] - hs[1][2] + hs[2][2] - hs[2][9] + hs[2][14] - hs[3][14])
    val c34 = (hs[1][0] + hs[1][2] - hs[1][11] + hs[1][12] + hs[2][11] - hs[2][14] - hs[3][0] - hs[3][5] - hs[3][6] + hs[3][14])
    val c35 = (-hs[1][6] + hs[1][11] - hs[1][12] - hs[1][14] - hs[2][8] + hs[2][10] - hs[2][11] - hs[2][12] + hs[3][5] + hs[3][6] - hs[3][14])
    val c41 = (hs[1][11] - hs[1][12] + hs[1][13] - hs[2][3] + hs[2][7] - hs[2][11] + hs[3][0] - hs[3][3] + hs[3][4] + hs[3][5] + hs[3][6] - hs[3][13] - hs[3][14])
    val c42 = (-hs[0][0] + hs[0][3] + hs[0][4] - hs[0][5] + hs[0][6] - hs[0][11] - hs[0][15] - hs[2][11])
    val c43 = (hs[0][0] + hs[0][5] - hs[0][6] + hs[0][11] - hs[1][11] + hs[1][12] - hs[2][9] - hs[3][8])
    val c44 = (hs[0][2] - hs[0][3] - hs[0][4] - hs[1][3] - hs[1][10] + hs[1][11] + hs[1][14] + hs[2][11])
    val c45 = (hs[0][1] - hs[0][3] - hs[0][4] - hs[1][8] - hs[1][9] + hs[1][11] + hs[1][15] - hs[2][7] + hs[2][11])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25, c31, c32, c33, c34, c35, c41, c42, c43, c44, c45)
}

fun multiply4By5MatrixAWith5By5MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 20) { "Matrix A must have 20 elements" }
    require(b.size == 25) { "Matrix B must have 25 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a21 = a[5]
    val a22 = a[6]
    val a23 = a[7]
    val a24 = a[8]
    val a25 = a[9]
    val a31 = a[10]
    val a32 = a[11]
    val a33 = a[12]
    val a34 = a[13]
    val a35 = a[14]
    val a41 = a[15]
    val a42 = a[16]
    val a43 = a[17]
    val a44 = a[18]
    val a45 = a[19]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b21 = b[5]
    val b22 = b[6]
    val b23 = b[7]
    val b24 = b[8]
    val b25 = b[9]
    val b31 = b[10]
    val b32 = b[11]
    val b33 = b[12]
    val b34 = b[13]
    val b35 = b[14]
    val b41 = b[15]
    val b42 = b[16]
    val b43 = b[17]
    val b44 = b[18]
    val b45 = b[19]
    val b51 = b[20]
    val b52 = b[21]
    val b53 = b[22]
    val b54 = b[23]
    val b55 = b[24]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    a32,
                    (a12 - a13 - a33),
                    (-a35 - a42 - a45),
                    (-a15 - a42 - a45),
                    (a11 - a21 + a22),
                    (-a14 - a32 - a34),
                    (-a25 - a42 - a45),
                    (a24 - a32 - a34),
                    (a42 + a45),
                    (-a12 + a13 - a23),
                    (-a21 + a22 + a31),
                    (a21 - a22 + a41),
                    (-a21 + a22),
                    (a12 - a13),
                    (a32 + a34),
                    -a43,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a44,
                    (-a43 - a44 + a45),
                    (-a24 - a33 + a34),
                    -a31,
                    (-a31 + a33 - a34),
                    a33,
                    (-a11 + a13 + a43),
                    a15,
                    a11,
                    (-a21 + a25 + a31),
                    (-a11 + a13 - a15),
                    a25,
                    -a24,
                    (-a15 + a44 - a45),
                    (-a21 + a24 + a25),
                    -a35,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a23,
                    (-a21 + a24 - a41),
                    (a21 - a24 + a25 + a41 - a43 - a44 + a45),
                    (-a11 + a21 - a24 - a25),
                    -a14,
                    (a13 - a15 - a23),
                    (-a14 - a31 - a34),
                    (a11 - a13 + a15 - a33),
                    (a31 - a33 + a34 + a44),
                    (-a35 + a43 - a45),
                    a22,
                    a12,
                    (a12 - a13 - a22 + a23 - a24 - a25 + a32 + a34 - a42 - a45),
                    a42,
                    (-a11 + a12 + a14 + a15 + a21 - a22 + a32 + a34 + a42 + a45),
                    (a24 - a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a21 - a25 - a31 + a35),
                    (-a13 + a15 + a23 - a25),
                    (-a21 + a31),
                    (a11 - a13 + a14 + a15 + a31 - a33 + a34 + a35),
                    (a14 + a34),
                    (-a23 + a24 + a33 - a34),
                    (a25 + a45),
                    (a33 - a34 - a35 + a43 - a44 - a45),
                    (-a13 + a23),
                    (-a21 + a24 - a41 + a44),
                    (a13 + a33),
                    (a11 - a13 + a14 + a31 - a33 + a34),
                    (-a15 - a45),
                    (-a13 - a14 + a15 - a43 - a44 + a45),
                    (-a13 - a43),
                    (-a11 + a13 - a41 + a43),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a31 + a33 - a34 - a35 - a41 + a43 - a44 - a45),
                    a41,
                    (-a32 - a34 - a44),
                    (-a12 + a13 + a21 - a22 - a32 - a34 + a41 - a42 + a43 - a44),
                    (-a11 + a21),
                    (a21 + a41),
                    (-a12 + a13 - a21 + a22 + a31 - a32 + a33 - a35 - a42 - a45),
                    (-a12 + a13 + a43),
                    (-a23 - a24 + a25 - a43 - a44 + a45),
                    (-a34 - a44),
                    (-a31 + a33 - a34 - a41 + a43 - a44),
                    (-a11 - a13 + a14 + a15 + a21 + a23 - a24 - a25),
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b22 + b24 - b42),
                    (-b24 + b32),
                    (b24 + b52),
                    (-b21 - b52),
                    (b12 - b21),
                    (-b21 - b42),
                    (b25 - b52),
                    (b25 - b42),
                    -b52,
                    (b25 + b32),
                    (b12 - b24),
                    (-b12 - b23),
                    b12,
                    b32,
                    b42,
                    (-b32 + b33 + b53),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b42 - b43 - b53),
                    b53,
                    (-b32 - b34 + b35 - b44),
                    (b12 + b14 - b44),
                    b44,
                    (-b32 - b34 - b44),
                    (b11 + b12 - b13 + b31),
                    (b31 + b51 + b52),
                    (b11 + b12 + b31),
                    (b15 - b52 - b54 + b55),
                    b31,
                    (b15 - b52 + b55),
                    (b15 - b42 + b45),
                    (b41 + b42 - b43 - b53),
                    -b15,
                    (b52 + b54),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b32 + b35),
                    (-b15 + b42 - b43 - b45 - b53),
                    (-b32 + b33 + b35 + b53),
                    (b15 - b41 - b42 + b45),
                    (-b41 - b42),
                    (-b31 + b41 + b42 - b45 - b51 - b52 + b55),
                    (-b11 - b12 - b14 + b44 + b51 + b52 + b54),
                    (-b31 - b51 - b52 - b54),
                    (-b12 + b13 - b14 + b44),
                    (-b12 + b13 - b14 - b32 + b33 - b34 + b53),
                    (b12 + b22 - b25),
                    (-b21 - b22 - b32),
                    b25,
                    (b22 - b23 - b52),
                    -b21,
                    (-b25 - b32 - b34 + b35 - b44 + b45),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b52 - b54 + b55),
                    (b41 + b42 - b45 - b51 - b52 + b55),
                    (b14 - b15 + b24 + b52 + b54 - b55),
                    (b51 + b52 + b54),
                    (-b11 - b12 - b14 - b21 + b41 + b44),
                    (b32 + b34 - b35),
                    (b25 + b32 - b33 - b35 - b53 - b55),
                    (-b12 + b13 - b14 - b32 + b33 - b34),
                    (-b25 + b31 - b35 + b51 + b52 - b55),
                    (b32 - b33 - b35 - b42 + b43 + b45),
                    (b11 + b12 + b14 + b24 + b31 + b34),
                    (b11 + b12 + b14 - b51 - b52 - b54),
                    (b21 - b41 - b42 + b43 - b51 + b53),
                    (b41 + b42 - b43),
                    (-b11 - b12 + b13 + b23 - b31 + b33 + b41 + b42 - b43),
                    (b11 + b12 - b13),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b12 - b13 + b14 - b24 + b32 - b33 + b34 - b53 + b54),
                    (-b12 + b13),
                    (-b23 + b42),
                    -b23,
                    (-b11 + b15 - b21 - b41 - b42 + b45),
                    (-b13 - b15 - b23 - b32 + b33 + b35 + b42 - b43 - b45),
                    -b24,
                    (b23 + b32),
                    (b32 - b33 - b35),
                    (-b23 - b32 + b33 - b34 + b43 - b44),
                    (-b24 + b32 - b33 + b34 - b53 + b54),
                    (b41 + b42 - b45),
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3], lefts[4] * rights[4])

    val c11 = (-hs[0][3] - hs[0][4] - hs[0][5] - hs[0][8] + hs[0][12] + hs[0][14] + hs[1][7] + hs[1][8] + hs[1][10] + hs[2][4] - hs[2][14])
    val c12 = (-hs[0][4] + hs[0][12] + hs[1][8] + hs[1][10] + hs[1][11] - hs[1][12] + hs[2][3] + hs[2][5] + hs[3][1] - hs[4][4])
    val c13 = (hs[0][5] - hs[0][14] + hs[1][3] - hs[1][10] + hs[1][15] - hs[2][4] + hs[2][6] + hs[2][7] + hs[3][3] + hs[3][4])
    val c14 = (hs[0][3] + hs[0][8] - hs[1][0] + hs[1][6] - hs[1][7] - hs[1][10] + hs[1][13] + hs[3][12] - hs[3][15] + hs[4][1])
    val c15 = (hs[0][3] + hs[0][4] + hs[0][5] + hs[0][8] - hs[0][12] - hs[0][13] - hs[0][14] - hs[2][11] + hs[2][14])
    val c21 = (hs[0][6] - hs[0][7] + hs[0][8] - hs[0][9] - hs[0][12] - hs[0][13] + hs[0][14] + hs[2][10] - hs[2][12])
    val c22 = (hs[0][0] - hs[0][1] - hs[0][2] + hs[0][8] + hs[0][10] - hs[0][12] + hs[0][13] + hs[0][14] - hs[4][6])
    val c23 = (-hs[0][8] - hs[0][11] + hs[0][12] + hs[0][13] - hs[0][14] + hs[2][13] - hs[4][2] + hs[4][3] + hs[4][7])
    val c24 = (-hs[0][13] + hs[0][15] - hs[1][1] + hs[1][6] + hs[1][8] + hs[1][13] + hs[2][4] + hs[3][13] - hs[3][14] - hs[4][7])
    val c25 = (hs[0][11] - hs[0][12] - hs[1][1] + hs[1][12] - hs[2][0] - hs[2][1] + hs[2][2] - hs[4][1] - hs[4][5] + hs[4][8])
    val c31 = (hs[0][14] + hs[1][0] - hs[1][1] - hs[1][3] - hs[1][5] + hs[2][8] - hs[2][9] + hs[3][7] + hs[4][2] - hs[4][9])
    val c32 = (-hs[0][11] + hs[0][12] + hs[0][13] - hs[0][14] - hs[0][15] - hs[1][0] + hs[1][1] + hs[4][1] - hs[4][2] + hs[4][3] + hs[4][7])
    val c33 = (-hs[0][1] + hs[0][13] + hs[1][4] + hs[1][5] - hs[1][7] - hs[1][8] - hs[2][6] - hs[2][7] + hs[3][10] + hs[3][11])
    val c34 = (-hs[0][10] + hs[0][12] + hs[1][2] - hs[1][3] - hs[1][4] - hs[1][9] + hs[1][11] + hs[2][0] - hs[3][2] - hs[3][5])
    val c35 = (hs[0][1] + hs[0][2] - hs[0][8] - hs[0][10] + hs[0][12] - hs[0][13] - hs[1][3] - hs[1][4] - hs[1][5] - hs[1][15] + hs[4][6])
    val c41 = (-hs[0][2] + hs[0][8] - hs[0][15] + hs[1][4] + hs[1][15] + hs[2][8] - hs[2][9] - hs[4][0] + hs[4][1] + hs[4][10])
    val c42 = (-hs[0][9] - hs[0][13] + hs[1][7] + hs[1][14] + hs[2][0] - hs[2][3] + hs[2][4] - hs[2][5] + hs[3][8] - hs[4][11])
    val c43 = (hs[0][6] - hs[0][7] + hs[0][8] - hs[0][9] - hs[0][13] + hs[0][14] + hs[1][11] - hs[1][12] + hs[1][14] + hs[2][0] - hs[2][12])
    val c44 = (-hs[0][7] + hs[0][14] - hs[1][2] - hs[1][5] + hs[1][9] - hs[1][12] + hs[1][14] - hs[1][15] - hs[2][15] + hs[3][0])
    val c45 = (-hs[0][6] - hs[0][8] + hs[0][15] + hs[1][0] - hs[1][11] - hs[1][14] + hs[2][1] - hs[2][2] - hs[3][6] + hs[3][9])

    return intArrayOf(c11, c12, c13, c14, c15, c21, c22, c23, c24, c25, c31, c32, c33, c34, c35, c41, c42, c43, c44, c45)
}

fun multiply4By5MatrixAWith5By9MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 20) { "Matrix A must have 20 elements" }
    require(b.size == 45) { "Matrix B must have 45 elements" }

    val a11 = a[0]
    val a12 = a[1]
    val a13 = a[2]
    val a14 = a[3]
    val a15 = a[4]
    val a21 = a[5]
    val a22 = a[6]
    val a23 = a[7]
    val a24 = a[8]
    val a25 = a[9]
    val a31 = a[10]
    val a32 = a[11]
    val a33 = a[12]
    val a34 = a[13]
    val a35 = a[14]
    val a41 = a[15]
    val a42 = a[16]
    val a43 = a[17]
    val a44 = a[18]
    val a45 = a[19]
    val b11 = b[0]
    val b12 = b[1]
    val b13 = b[2]
    val b14 = b[3]
    val b15 = b[4]
    val b16 = b[5]
    val b17 = b[6]
    val b18 = b[7]
    val b19 = b[8]
    val b21 = b[9]
    val b22 = b[10]
    val b23 = b[11]
    val b24 = b[12]
    val b25 = b[13]
    val b26 = b[14]
    val b27 = b[15]
    val b28 = b[16]
    val b29 = b[17]
    val b31 = b[18]
    val b32 = b[19]
    val b33 = b[20]
    val b34 = b[21]
    val b35 = b[22]
    val b36 = b[23]
    val b37 = b[24]
    val b38 = b[25]
    val b39 = b[26]
    val b41 = b[27]
    val b42 = b[28]
    val b43 = b[29]
    val b44 = b[30]
    val b45 = b[31]
    val b46 = b[32]
    val b47 = b[33]
    val b48 = b[34]
    val b49 = b[35]
    val b51 = b[36]
    val b52 = b[37]
    val b53 = b[38]
    val b54 = b[39]
    val b55 = b[40]
    val b56 = b[41]
    val b57 = b[42]
    val b58 = b[43]
    val b59 = b[44]

    val lefts =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    -a32,
                    (-a12 + a13 + a33),
                    (a35 + a42 + a45),
                    (a15 + a42 + a45),
                    (-a11 + a21 - a22),
                    (a14 + a32 + a34),
                    (a25 + a42 + a45),
                    (-a24 + a32 + a34),
                    (-a42 - a45),
                    (a12 - a13 + a23),
                    (a21 - a22 - a31),
                    (-a21 + a22 - a41),
                    (a21 - a22),
                    (-a12 + a13),
                    (-a32 - a34),
                    a43,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a44,
                    (a43 + a44 - a45),
                    (a24 + a33 - a34),
                    a31,
                    (a31 - a33 + a34),
                    -a33,
                    (a11 - a13 - a43),
                    -a15,
                    -a11,
                    (a21 - a25 - a31),
                    (a11 - a13 + a15),
                    -a25,
                    a24,
                    (a15 - a44 + a45),
                    (a21 - a24 - a25),
                    a35,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a23,
                    (a21 - a24 + a41),
                    (-a21 + a24 - a25 - a41 + a43 + a44 - a45),
                    (a11 - a21 + a24 + a25),
                    a14,
                    (-a13 + a15 + a23),
                    (a14 + a31 + a34),
                    (-a11 + a13 - a15 + a33),
                    (-a31 + a33 - a34 - a44),
                    (a35 - a43 + a45),
                    -a22,
                    -a12,
                    (-a12 + a13 + a22 - a23 + a24 + a25 - a32 - a34 + a42 + a45),
                    -a42,
                    (a11 - a12 - a14 - a15 - a21 + a22 - a32 - a34 - a42 - a45),
                    (-a24 + a34),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-a21 + a25 + a31 - a35),
                    (a13 - a15 - a23 + a25),
                    (a21 - a31),
                    (-a11 + a13 - a14 - a15 - a31 + a33 - a34 - a35),
                    (-a14 - a34),
                    (a23 - a24 - a33 + a34),
                    (-a25 - a45),
                    (-a33 + a34 + a35 - a43 + a44 + a45),
                    (a13 - a23),
                    (a21 - a24 + a41 - a44),
                    (-a13 - a33),
                    (-a11 + a13 - a14 - a31 + a33 - a34),
                    (a15 + a45),
                    (a13 + a14 - a15 + a43 + a44 - a45),
                    (a13 + a43),
                    (a11 - a13 + a41 - a43),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a31 - a33 + a34 + a35 + a41 - a43 + a44 + a45),
                    -a41,
                    (a32 + a34 + a44),
                    (a12 - a13 - a21 + a22 + a32 + a34 - a41 + a42 - a43 + a44),
                    (a11 - a21),
                    (-a21 - a41),
                    (a12 - a13 + a21 - a22 - a31 + a32 - a33 + a35 + a42 + a45),
                    (a12 - a13 - a43),
                    (a23 + a24 - a25 + a43 + a44 - a45),
                    (a34 + a44),
                    (a31 - a33 + a34 + a41 - a43 + a44),
                    (a11 + a13 - a14 - a15 - a21 - a23 + a24 + a25),
                    (a12 - a13 - a15 + a25),
                    (a13 + a45),
                    (-a12 + a35),
                    (a14 + a15 - a35 - a45),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (a11 + a15 - a35 - a45),
                    (-a12 - a13 - a15 + a25),
                    (a11 + a15 - a25),
                    (a13 + a41 - a43),
                    (-a12 + a31 + a32),
                    (a23 + a41 - a43),
                    (-a22 + a31 + a32),
                    (-a11 - a15 + a21 + a22 - a23 + a25 - a31 - a32 - a41 + a43),
                    (-a31 - a32 + a42),
                    (a33 - a42),
                    -a31,
                    (-a11 - a15),
                    (-a14 + a22 - a23 + a24),
                    -a21,
                    (a11 + a12 - a13 - a14 - a21 + a24 - a31 - a32 - a41 + a43),
                    -a35,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a22,
                    (-a22 + a32 + a33),
                    (-a22 - a34),
                    (-a23 + a42 + a43),
                    (a23 - a43 - a45),
                    -a45,
                    (-a22 + a32 - a35),
                    (a22 - a23 - a24 - a25 - a32 + a34 + a35 + a43 + a44 + a45),
                    (a24 + a25 - a34 - a44),
                    a44,
                    (-a32 - a34 + a35 + a42),
                    a45,
                    -a42,
                    a33,
                    a14,
                    (a23 - a44),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    -a13,
                    (a12 + a13),
                    a12,
                    (a33 - a43 + a44 - a45),
                    (a12 - a32 - a33 - a34),
                    (-a14 - a25),
                    a34,
                    (-a11 - a15 - a21 + a24 + a31 - a34 + a35 - a44 + a45),
                    -a34,
                    a41,
                    (a14 + a21 - a24),
                    -a12,
                    (a21 - a24 + a44),
                    -a23,
                    -a33,
                    -a44,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    a44,
                    (a21 - a24 + a34 + a44),
                    (-a24 + a34 + a44),
                    -a41,
                    -a25,
                    (a33 + a41 - a43),
                    a42,
                    (-a33 + a42),
                    -a11,
                    (-a13 + a42 + a43 - a44),
                    (-a14 + a24 - a34 - a44),
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val rights =
        arrayOf(
            Int32x16.from(
                intArrayOf(
                    (b26 + b28 - b46),
                    (-b28 + b36),
                    (b28 + b56),
                    (-b25 - b56),
                    (b16 - b25),
                    (-b25 - b46),
                    (b29 - b56),
                    (b29 - b46),
                    -b56,
                    (b29 + b36),
                    (b16 - b28),
                    (-b16 - b27),
                    b16,
                    b36,
                    b46,
                    (-b36 + b37 + b57),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b46 - b47 - b57),
                    b57,
                    (-b36 - b38 + b39 - b48),
                    (b16 + b18 - b48),
                    b48,
                    (-b36 - b38 - b48),
                    (b15 + b16 - b17 + b35),
                    (b35 + b55 + b56),
                    (b15 + b16 + b35),
                    (b19 - b56 - b58 + b59),
                    b35,
                    (b19 - b56 + b59),
                    (b19 - b46 + b49),
                    (b45 + b46 - b47 - b57),
                    -b19,
                    (b56 + b58),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b36 + b39),
                    (-b19 + b46 - b47 - b49 - b57),
                    (-b36 + b37 + b39 + b57),
                    (b19 - b45 - b46 + b49),
                    (-b45 - b46),
                    (-b35 + b45 + b46 - b49 - b55 - b56 + b59),
                    (-b15 - b16 - b18 + b48 + b55 + b56 + b58),
                    (-b35 - b55 - b56 - b58),
                    (-b16 + b17 - b18 + b48),
                    (-b16 + b17 - b18 - b36 + b37 - b38 + b57),
                    (b16 + b26 - b29),
                    (-b25 - b26 - b36),
                    b29,
                    (b26 - b27 - b56),
                    -b25,
                    (-b29 - b36 - b38 + b39 - b48 + b49),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b56 - b58 + b59),
                    (b45 + b46 - b49 - b55 - b56 + b59),
                    (b18 - b19 + b28 + b56 + b58 - b59),
                    (b55 + b56 + b58),
                    (-b15 - b16 - b18 - b25 + b45 + b48),
                    (b36 + b38 - b39),
                    (b29 + b36 - b37 - b39 - b57 - b59),
                    (-b16 + b17 - b18 - b36 + b37 - b38),
                    (-b29 + b35 - b39 + b55 + b56 - b59),
                    (b36 - b37 - b39 - b46 + b47 + b49),
                    (b15 + b16 + b18 + b28 + b35 + b38),
                    (b15 + b16 + b18 - b55 - b56 - b58),
                    (b25 - b45 - b46 + b47 - b55 + b57),
                    (b45 + b46 - b47),
                    (-b15 - b16 + b17 + b27 - b35 + b37 + b45 + b46 - b47),
                    (b15 + b16 - b17),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b16 - b17 + b18 - b28 + b36 - b37 + b38 - b57 + b58),
                    (-b16 + b17),
                    (-b27 + b46),
                    -b27,
                    (-b15 + b19 - b25 - b45 - b46 + b49),
                    (-b17 - b19 - b27 - b36 + b37 + b39 + b46 - b47 - b49),
                    -b28,
                    (b27 + b36),
                    (b36 - b37 - b39),
                    (-b27 - b36 + b37 - b38 + b47 - b48),
                    (-b28 + b36 - b37 + b38 - b57 + b58),
                    (b45 + b46 - b49),
                    (-b51 - b53),
                    (b33 - b51 - b53),
                    (-b23 - b51 - b53),
                    (-b13 - b51),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b13 + b54),
                    (b12 - b52),
                    (b12 - b54),
                    (-b11 - b13 - b34),
                    (-b11 - b13 + b24),
                    (b12 + b34),
                    (b12 - b24),
                    b12,
                    b24,
                    (-b11 - b13 + b21 - b31),
                    (b13 + b14 - b24),
                    b54,
                    (-b11 - b13 - b41 - b42 - b43),
                    (-b12 + b14 + b44),
                    (-b11 - b13),
                    (-b23 - b53 - b54),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b11 - b13 + b21 + b22 + b23 - b24 - b41 - b42 - b43),
                    (b12 - b22 + b23 - b52 + b53 + b54),
                    (-b11 - b13 + b21 + b23 - b41 - b42 - b43),
                    (-b12 - b32 + b33 + b52 - b53 - b54),
                    (b33 + b52 - b53 - b54),
                    (b33 - b53),
                    (-b23 + b52 - b53 - b54),
                    (b52 - b53 - b54),
                    (b13 + b43 + b52 - b53 - b54),
                    (-b11 - b13 - b31 - b41 - b44),
                    -b23,
                    b54,
                    (-b11 + b12 - b13 + b21 - b22 - b31 + b32 - b33 - b52 + b53 + b54),
                    (-b11 + b12 - b13 + b21 - b22 + b23 - b31 + b32 - b52 + b53 + b54),
                    (b13 - b41 + b44 + b51),
                    (-b11 - b13 - b31 - b33 - b41 - b42 - b43),
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (b31 + b33 - b34 - b51 - b53),
                    (b12 + b32 - b52),
                    (-b22 + b32),
                    -b33,
                    (-b11 - b13 + b21),
                    (-b13 - b43 - b51),
                    (-b11 - b13 + b21 - b41 + b44),
                    b13,
                    (-b23 + b43 + b44),
                    (b14 + b34),
                    (-b11 - b13 - b44),
                    (b21 - b22 + b23 - b24 + b32 + b51 + b53),
                    b44,
                    (-b11 - b13 - b31 - b32 - b33 + b34 - b41 - b42 - b43),
                    (-b11 - b13 + b21 - b31 + b34),
                    b44,
                ),
            ),
            Int32x16.from(
                intArrayOf(
                    (-b11 - b13 - b31 - b33 - b41 - b43),
                    (b13 - b44),
                    (b13 - b42 + b43 + b52 - b53 - b54),
                    b13,
                    (b13 + b43 - b53),
                    b34,
                    (-b11 - b13 + b21 - b24 - b31),
                    (-b11 + b12 - b13 + b21 - b22 + b23 - b31 + b32 - b33 - b52 + b53 + b54),
                    (-b11 - b13 + b14 - b54),
                    (-b11 - b13 - b31),
                    -b42,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        )

    val hs = arrayOf(lefts[0] * rights[0], lefts[1] * rights[1], lefts[2] * rights[2], lefts[3] * rights[3], lefts[4] * rights[4], lefts[5] * rights[5], lefts[6] * rights[6], lefts[7] * rights[7], lefts[8] * rights[8])

    val c11 = (-hs[4][13] - hs[4][14] - hs[4][15] - hs[5][0] - hs[5][3] - hs[5][4] - hs[5][8] - hs[5][9] - hs[5][11] - hs[5][14] - hs[5][15] - hs[6][5] - hs[6][11] - hs[6][14] - hs[7][0] - hs[7][2] - hs[7][10] - hs[7][11] - hs[7][12] - hs[7][14] - hs[7][15] - hs[8][5] - hs[8][6])
    val c12 = (-hs[5][12] - hs[6][2] - hs[6][4] - hs[6][6] - hs[6][7] - hs[6][8] - hs[6][9] - hs[6][10] - hs[6][12] - hs[6][13] - hs[6][14] - hs[6][15] - hs[7][3] + hs[7][5] - hs[7][6] - hs[7][10] - (hs[7][12] shl 1) - hs[8][1] - hs[8][7] + hs[8][10])
    val c13 = (-hs[4][14] - hs[5][0] - hs[5][4] - hs[5][8] - hs[5][9] - hs[5][11] - hs[5][15] - hs[6][11] - hs[7][2] - hs[7][4] - hs[7][6] - hs[7][7] - hs[7][11] - hs[7][12] - hs[8][1] - hs[8][6])
    val c14 = (-hs[4][13] - hs[5][3] - hs[5][9] - hs[6][5] - hs[6][9] - hs[7][0] - hs[7][14] + hs[7][15] + hs[8][3] - hs[8][5] - hs[8][9])
    val c15 = (hs[5][1] + hs[5][2] - hs[5][11] + hs[6][8] + hs[7][1] - hs[7][2] + hs[8][2] + hs[8][4] + hs[8][10])
    val c16 = (hs[5][2] + hs[5][5] + hs[5][6] + hs[5][7] - hs[5][8] - hs[5][9] - hs[5][11] + hs[6][0] + hs[6][2] + hs[6][8] + hs[6][15] + hs[7][6] + hs[7][8] + hs[7][13] - hs[7][14] + hs[8][0] + hs[8][2] + hs[8][4] - hs[8][5] - hs[8][6])
    val c17 = (hs[5][6] - hs[5][8] - hs[5][9] + hs[5][15] + hs[6][0] - hs[6][1] + hs[6][2] - hs[6][6] + hs[6][13] + hs[7][6] + hs[7][8] - hs[8][6])
    val c18 = (hs[5][5] - hs[5][9] - hs[6][3] - hs[6][4] + hs[6][5] + hs[6][11] + hs[6][12] + hs[6][15] + hs[7][13] - hs[7][14] + hs[8][0] - hs[8][5])
    val c19 = (hs[4][12] + hs[4][13] + hs[4][14] + hs[4][15] + hs[5][0] + hs[5][11] + hs[5][15] + hs[6][5] + hs[6][11] + hs[7][5] + hs[8][4])
    val c21 = (hs[6][4] + hs[6][6] + hs[6][7] + hs[6][8] + hs[6][9] + hs[6][10] + hs[6][12] + hs[6][13] + hs[7][3] - hs[7][8] + hs[7][12] - hs[7][15] - hs[8][0] + hs[8][1] + hs[8][4] + hs[8][7])
    val c22 = (hs[5][0] + hs[5][11] + hs[5][15] + hs[6][10] + hs[6][11] + hs[6][12] + hs[6][13] + hs[7][7] - hs[7][8] + hs[7][12] + hs[8][1] + hs[8][7])
    val c23 = (hs[6][5] + hs[6][9] + hs[6][12] + hs[6][13] + hs[7][3] - hs[7][15] - hs[8][0] - hs[8][3] + hs[8][7])
    val c24 = (-hs[5][3] - hs[5][4] - hs[5][8] - hs[5][9] - hs[5][11] - hs[5][14] - hs[7][10] - hs[7][12] - hs[7][14] - hs[7][15] - hs[8][5] - hs[8][6] - hs[8][8])
    val c25 = (hs[5][2] + hs[5][5] + hs[5][6] + hs[5][7] - hs[5][8] - hs[5][9] - hs[5][11] - hs[5][13] - hs[7][12] - hs[7][14] - hs[7][15] - hs[8][5] - hs[8][6])
    val c26 = (-hs[5][0] - hs[5][8] - hs[5][9] - hs[5][10] - hs[5][11] - hs[6][11] - hs[7][7] - hs[7][12] - hs[7][14] - hs[8][1] - hs[8][6])
    val c27 = (-hs[5][9] + hs[6][11] + hs[7][9] - hs[7][14] - hs[7][15] - hs[8][5] - hs[8][6])
    val c28 = (hs[0][3] + hs[0][4] + hs[0][5] + hs[0][8] - hs[0][12] - hs[0][14] - hs[1][7] - hs[1][8] - hs[1][10] - hs[2][4] + hs[2][14])
    val c29 = (hs[0][4] - hs[0][12] - hs[1][8] - hs[1][10] - hs[1][11] + hs[1][12] - hs[2][3] - hs[2][5] - hs[3][1] + hs[4][4])
    val c31 = (-hs[0][5] + hs[0][14] - hs[1][3] + hs[1][10] - hs[1][15] + hs[2][4] - hs[2][6] - hs[2][7] - hs[3][3] - hs[3][4])
    val c32 = (-hs[0][3] - hs[0][8] + hs[1][0] - hs[1][6] + hs[1][7] + hs[1][10] - hs[1][13] - hs[3][12] + hs[3][15] - hs[4][1])
    val c33 = (-hs[0][3] - hs[0][4] - hs[0][5] - hs[0][8] + hs[0][12] + hs[0][13] + hs[0][14] + hs[2][11] - hs[2][14])
    val c34 = (-hs[0][6] + hs[0][7] - hs[0][8] + hs[0][9] + hs[0][12] + hs[0][13] - hs[0][14] - hs[2][10] + hs[2][12])
    val c35 = (-hs[0][0] + hs[0][1] + hs[0][2] - hs[0][8] - hs[0][10] + hs[0][12] - hs[0][13] - hs[0][14] + hs[4][6])
    val c36 = (hs[0][8] + hs[0][11] - hs[0][12] - hs[0][13] + hs[0][14] - hs[2][13] + hs[4][2] - hs[4][3] - hs[4][7])
    val c37 = (hs[0][13] - hs[0][15] + hs[1][1] - hs[1][6] - hs[1][8] - hs[1][13] - hs[2][4] - hs[3][13] + hs[3][14] + hs[4][7])
    val c38 = (-hs[0][11] + hs[0][12] + hs[1][1] - hs[1][12] + hs[2][0] + hs[2][1] - hs[2][2] + hs[4][1] + hs[4][5] - hs[4][8])
    val c39 = (-hs[0][14] - hs[1][0] + hs[1][1] + hs[1][3] + hs[1][5] - hs[2][8] + hs[2][9] - hs[3][7] - hs[4][2] + hs[4][9])
    val c41 = (hs[0][11] - hs[0][12] - hs[0][13] + hs[0][14] + hs[0][15] + hs[1][0] - hs[1][1] - hs[4][1] + hs[4][2] - hs[4][3] - hs[4][7])
    val c42 = (hs[0][1] - hs[0][13] - hs[1][4] - hs[1][5] + hs[1][7] + hs[1][8] + hs[2][6] + hs[2][7] - hs[3][10] - hs[3][11])
    val c43 = (hs[0][10] - hs[0][12] - hs[1][2] + hs[1][3] + hs[1][4] + hs[1][9] - hs[1][11] - hs[2][0] + hs[3][2] + hs[3][5])
    val c44 = (-hs[0][1] - hs[0][2] + hs[0][8] + hs[0][10] - hs[0][12] + hs[0][13] + hs[1][3] + hs[1][4] + hs[1][5] + hs[1][15] - hs[4][6])
    val c45 = (hs[0][2] - hs[0][8] + hs[0][15] - hs[1][4] - hs[1][15] - hs[2][8] + hs[2][9] + hs[4][0] - hs[4][1] - hs[4][10])
    val c46 = (hs[0][9] + hs[0][13] - hs[1][7] - hs[1][14] - hs[2][0] + hs[2][3] - hs[2][4] + hs[2][5] - hs[3][8] + hs[4][11])
    val c47 = (-hs[0][6] + hs[0][7] - hs[0][8] + hs[0][9] + hs[0][13] - hs[0][14] - hs[1][11] + hs[1][12] - hs[1][14] - hs[2][0] + hs[2][12])
    val c48 = (hs[0][7] - hs[0][14] + hs[1][2] + hs[1][5] - hs[1][9] + hs[1][12] - hs[1][14] + hs[1][15] + hs[2][15] - hs[3][0])
    val c49 = (hs[0][6] + hs[0][8] - hs[0][15] - hs[1][0] + hs[1][11] + hs[1][14] - hs[2][1] + hs[2][2] + hs[3][6] - hs[3][9])

    return intArrayOf(c11, c12, c13, c14, c15, c16, c17, c18, c19, c21, c22, c23, c24, c25, c26, c27, c28, c29, c31, c32, c33, c34, c35, c36, c37, c38, c39, c41, c42, c43, c44, c45, c46, c47, c48, c49)
}

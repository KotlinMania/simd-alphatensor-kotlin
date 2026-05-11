@file:Suppress("unused", "LocalVariableName", "FunctionName")

package com.kotlinmania.simd

/**
 * SIMD AlphaTensor Matrix Multiplication Algorithms
 * 
 * This library contains cutting-edge matrix multiplication algorithms discovered by AlphaTensor.
 * These are novel, machine-discovered algorithms that perform matrix multiplication with fewer
 * operations than traditional methods.
 * 
 * Original Rust implementation by: drbh (david.richard.holtz@gmail.com)
 * Original repository: https://github.com/drbh/simd-alphatensor-rs
 * Kotlin translation by: @sydneyrenee / sydney@solace.ofharmony.ai (The Solace Project) / KotlinMania
 * 
 * Note: Since Kotlin/JVM lacks native SIMD support like Rust, this implementation uses
 * regular arrays. Performance characteristics will differ from the Rust implementation.
 */

/**
 * Helper class to simulate SIMD operations with 16 Int32 elements.
 * This simulates Rust's i32x16 SIMD type using regular arrays.
 */
internal class Int32x16(private val data: IntArray) {
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

/**
 * Multiply a 2x2 matrix A with a 2x2 matrix B using AlphaTensor algorithm.
 * 
 * This uses the novel AlphaTensor-discovered algorithm that requires only 7 multiplications
 * instead of the traditional 8 multiplications for 2x2 matrix multiplication.
 * 
 * @param a Matrix A as a flattened array [a11, a12, a21, a22] (row-major order)
 * @param b Matrix B as a flattened array [b11, b12, b21, b22] (row-major order)
 * @return Result matrix C as a flattened array [c11, c12, c21, c22] (row-major order)
 */
fun multiply2By2MatrixAWith2By2MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 4) { "Matrix B must have 4 elements" }
    
    val (a11, a12, a21, a22) = a
    val (b11, b12, b21, b22) = b
    
    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
            (a21 - a22),
            (a11 + a21 - a22),
            (a11 - a12 + a21 - a22),
            a12,
            (a11 + a21),
            a11,
            a22,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )
    
    val rights = arrayOf(
        Int32x16.from(intArrayOf(
            b12,
            (b12 + b21 + b22),
            (b21 + b22),
            b21,
            (b11 + b12 + b21 + b22),
            b11,
            (b12 + b22),
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )
    
    val hs = arrayOf(lefts[0] * rights[0])
    
    val c11 = (hs[0][3] + hs[0][5])
    val c12 = (-hs[0][1] + hs[0][4] - hs[0][5] - hs[0][6])
    val c21 = (-hs[0][0] + hs[0][1] - hs[0][2] - hs[0][3])
    val c22 = (hs[0][0] + hs[0][6])
    
    return intArrayOf(c11, c12, c21, c22)
}

/**
 * Multiply a 2x2 matrix A with a 2x3 matrix B using AlphaTensor algorithm.
 *
 * Uses 11 multiplications instead of the traditional 12.
 *
 * @param a Matrix A as a flattened array of 4 elements (row-major order)
 * @param b Matrix B as a flattened array of 6 elements (row-major order)
 * @return Result matrix C as a flattened array of 6 elements (row-major order)
 */
fun multiply2By2MatrixAWith2By3MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 6) { "Matrix B must have 6 elements" }

    val (a11, a12, a21, a22) = a
    val (b11, b12, b13, b21, b22, b23) = b

    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
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
            0, 0, 0, 0, 0
        ))
    )

    val rights = arrayOf(
        Int32x16.from(intArrayOf(
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
            0, 0, 0, 0, 0
        ))
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

/**
 * Multiply a 2x2 matrix A with a 2x4 matrix B using AlphaTensor algorithm.
 *
 * Uses 14 multiplications instead of the traditional 16.
 *
 * @param a Matrix A as a flattened array of 4 elements (row-major order)
 * @param b Matrix B as a flattened array of 8 elements (row-major order)
 * @return Result matrix C as a flattened array of 8 elements (row-major order)
 */
fun multiply2By2MatrixAWith2By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 8) { "Matrix B must have 8 elements" }

    val (a11, a12, a21, a22) = a
    val (b11, b12, b13, b14, b21, b22, b23, b24) = b

    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
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
            0
        ))
    )

    val rights = arrayOf(
        Int32x16.from(intArrayOf(
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
            0
        ))
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

// Component operators for destructuring IntArray (up to 16 elements)
private operator fun IntArray.component1() = this[0]
private operator fun IntArray.component2() = this[1]
private operator fun IntArray.component3() = this[2]
private operator fun IntArray.component4() = this[3]
private operator fun IntArray.component5() = this[4]
private operator fun IntArray.component6() = this[5]
private operator fun IntArray.component7() = this[6]
private operator fun IntArray.component8() = this[7]
private operator fun IntArray.component9() = this[8]
private operator fun IntArray.component10() = this[9]
private operator fun IntArray.component11() = this[10]
private operator fun IntArray.component12() = this[11]
private operator fun IntArray.component13() = this[12]
private operator fun IntArray.component14() = this[13]
private operator fun IntArray.component15() = this[14]
private operator fun IntArray.component16() = this[15]

/**
 * Multiply a 3x3 matrix A with a 3x3 matrix B using AlphaTensor algorithm.
 *
 * Uses 23 multiplications, matching the best known bound for 3x3 matrix
 * multiplication (Laderman 1976 / Smirnov 1986).
 *
 * @param a Matrix A as a flattened array of 9 elements (row-major order)
 * @param b Matrix B as a flattened array of 9 elements (row-major order)
 * @return Result matrix C as a flattened array of 9 elements (row-major order)
 */
fun multiply3By3MatrixAWith3By3MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 9) { "Matrix A must have 9 elements" }
    require(b.size == 9) { "Matrix B must have 9 elements" }

    val (a11, a12, a13, a21, a22, a23, a31, a32, a33) = a
    val (b11, b12, b13, b21, b22, b23, b31, b32, b33) = b

    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
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
            (a22 - a23)
        )),
        Int32x16.from(intArrayOf(
            (a21 + a22 - a31 - a32),
            (a21 + (a31 shl 1) + (a33 shl 1)),
            (a22 - a32),
            (a11 - a31),
            (a12 - a21 + a22),
            a13,
            (a21 + a23 + (a31 shl 1) + (a33 shl 1)),
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )

    val rights = arrayOf(
        Int32x16.from(intArrayOf(
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
            b21
        )),
        Int32x16.from(intArrayOf(
            b12,
            (b11 - b21 - b31),
            (b12 - b22 - b32),
            (b11 - b12 - b13 + (b21 shl 1) - b22 - b23 - b31 + b33),
            b13,
            (b32 - b33),
            (b21 + b31),
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )

    val hs = arrayOf(
        lefts[0] * rights[0],
        lefts[1] * rights[1]
    )

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

/**
 * Multiply a 4x4 matrix A with a 4x4 matrix B using AlphaTensor algorithm.
 * 
 * This is the breakthrough AlphaTensor algorithm that performs 4x4 matrix multiplication
 * with fewer operations than any previously known method. This is considered one of the
 * most significant algorithms discovered by the AlphaTensor system.
 * 
 * @param a Matrix A as a flattened array of 16 elements (row-major order)
 * @param b Matrix B as a flattened array of 16 elements (row-major order)
 * @return Result matrix C as a flattened array of 16 elements (row-major order)
 */
fun multiply4By4MatrixAWith4By4MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 16) { "Matrix A must have 16 elements" }
    require(b.size == 16) { "Matrix B must have 16 elements" }
    
    val (a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44) = a
    val (b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44) = b
    
    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
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
            (a12 + a14 - a21 + a22 + a23 + a24)
        )),
        Int32x16.from(intArrayOf(
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
            -a43
        )),
        Int32x16.from(intArrayOf(
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
            (-a43 - a44)
        )),
        Int32x16.from(intArrayOf(-a23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    )
    
    val rights = arrayOf(
        Int32x16.from(intArrayOf(
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
            (b12 + b14 - b21 + b22 + b23 + b24)
        )),
        Int32x16.from(intArrayOf(
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
            (b13 + b14 + b23 + b24 + b33 + b34 - b43 - b44)
        )),
        Int32x16.from(intArrayOf(
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
            (-b43 - b44)
        )),
        Int32x16.from(intArrayOf((-b31 + b32 + b41 - b42), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    )
    
    val hs = arrayOf(
        lefts[0] * rights[0],
        lefts[1] * rights[1],
        lefts[2] * rights[2],
        lefts[3] * rights[3]
    )
    
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
    
    return intArrayOf(
        c11, c12, c13, c14,
        c21, c22, c23, c24,
        c31, c32, c33, c34,
        c41, c42, c43, c44
    )
}

package com.kotlinmania.simd

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * Multiplies two row-major matrices and returns the row-major result matrix.
 *
 * - [a] is an `m x k` matrix encoded in row-major order.
 * - [b] is a `k x n` matrix encoded in row-major order.
 * - The returned array is an `m x n` matrix encoded in row-major order.
 *
 * @param a left operand matrix data (`m x k`, row-major)
 * @param b right operand matrix data (`k x n`, row-major)
 * @param m number of rows in [a] and in the result
 * @param k number of columns in [a] / rows in [b]
 * @param n number of columns in [b] and in the result
 */
private fun naiveMultiply(a: IntArray, b: IntArray, m: Int, k: Int, n: Int): IntArray {
    require(a.size == m * k)
    require(b.size == k * n)
    val c = IntArray(m * n)
    for (i in 0 until m) {
        for (j in 0 until n) {
            var sum = 0
            for (p in 0 until k) {
                sum += a[i * k + p] * b[p * n + j]
            }
            c[i * n + j] = sum
        }
    }
    return c
}

/**
 * Generates a flattened row-major matrix as an [IntArray] of [size] elements for test inputs.
 *
 * Each generated value is sampled uniformly from the half-open range `[-bound, bound)`.
 *
 * @param rng random source used to generate values.
 * @param size total number of elements in the flattened matrix.
 * @param bound exclusive absolute bound for generated values (default: 100).
 */
private fun randomMatrix(rng: Random, size: Int, bound: Int = 100): IntArray =
    IntArray(size) { rng.nextInt(-bound, bound) }

private const val RANDOM_TEST_ITERATIONS = 64

/**
 * Upstream AlphaTensor algorithms emit their result as `reshape(n, m).T`. Reinterpret the
 * raw `[c11, c12, ...]` output as an `m x n` row-major matrix matching the naive product.
 */
private fun reorderResult(arr: IntArray, m: Int, n: Int): IntArray {
    require(arr.size == m * n)
    val result = IntArray(m * n)
    for (i in 0 until m) {
        for (j in 0 until n) {
            result[i * n + j] = arr[j * m + i]
        }
    }
    return result
}

class MatrixMultiplicationTest {
    @Test
    fun matrixMultiply_2x2_by_2x2_matchesNaive() {
        val rng = Random(0x222A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 4)
            val got = reorderResult(multiply2By2MatrixAWith2By2MatrixB(a, b), 2, 2)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 2), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x3_matchesNaive() {
        val rng = Random(0x223A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 6)
            val got = reorderResult(multiply2By2MatrixAWith2By3MatrixB(a, b), 2, 3)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 3), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x4_matchesNaive() {
        val rng = Random(0x224A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 8)
            val got = reorderResult(multiply2By2MatrixAWith2By4MatrixB(a, b), 2, 4)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 4), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x5_matchesNaive() {
        val rng = Random(0x225A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 10)
            val got = reorderResult(multiply2By2MatrixAWith2By5MatrixB(a, b), 2, 5)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 5), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x6_matchesNaive() {
        val rng = Random(0x226A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 12)
            val got = reorderResult(multiply2By2MatrixAWith2By6MatrixB(a, b), 2, 6)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 6), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x7_matchesNaive() {
        val rng = Random(0x227A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 14)
            val got = reorderResult(multiply2By2MatrixAWith2By7MatrixB(a, b), 2, 7)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 7), got)
        }
    }

    @Test
    fun matrixMultiply_2x2_by_2x8_matchesNaive() {
        val rng = Random(0x228A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 16)
            val got = reorderResult(multiply2By2MatrixAWith2By8MatrixB(a, b), 2, 8)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 8), got)
        }
    }

    @Test
    fun matrixMultiply_2x3_by_3x3_matchesNaive() {
        val rng = Random(0x233A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 6)
            val b = randomMatrix(rng, 9)
            val got = reorderResult(multiply2By3MatrixAWith3By3MatrixB(a, b), 2, 3)
            assertContentEquals(naiveMultiply(a, b, 2, 3, 3), got)
        }
    }

    @Test
    fun matrixMultiply_2x3_by_3x4_matchesNaive() {
        val rng = Random(0x234A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 6)
            val b = randomMatrix(rng, 12)
            val got = reorderResult(multiply2By3MatrixAWith3By4MatrixB(a, b), 2, 4)
            assertContentEquals(naiveMultiply(a, b, 2, 3, 4), got)
        }
    }

    @Test
    fun matrixMultiply_2x3_by_3x5_matchesNaive() {
        val rng = Random(0x235A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 6)
            val b = randomMatrix(rng, 15)
            val got = reorderResult(multiply2By3MatrixAWith3By5MatrixB(a, b), 2, 5)
            assertContentEquals(naiveMultiply(a, b, 2, 3, 5), got)
        }
    }

    @Test
    fun matrixMultiply_2x4_by_4x4_matchesNaive() {
        val rng = Random(0x244A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 8)
            val b = randomMatrix(rng, 16)
            val got = reorderResult(multiply2By4MatrixAWith4By4MatrixB(a, b), 2, 4)
            assertContentEquals(naiveMultiply(a, b, 2, 4, 4), got)
        }
    }

    @Test
    fun matrixMultiply_2x4_by_4x5_matchesNaive() {
        val rng = Random(0x245A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 8)
            val b = randomMatrix(rng, 20)
            val got = reorderResult(multiply2By4MatrixAWith4By5MatrixB(a, b), 2, 5)
            assertContentEquals(naiveMultiply(a, b, 2, 4, 5), got)
        }
    }

    @Test
    fun matrixMultiply_2x5_by_5x5_matchesNaive() {
        val rng = Random(0x255A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 10)
            val b = randomMatrix(rng, 25)
            val got = reorderResult(multiply2By5MatrixAWith5By5MatrixB(a, b), 2, 5)
            assertContentEquals(naiveMultiply(a, b, 2, 5, 5), got)
        }
    }

    @Test
    fun matrixMultiply_3x3_by_3x3_matchesNaive() {
        val rng = Random(0x333A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 9)
            val b = randomMatrix(rng, 9)
            val got = reorderResult(multiply3By3MatrixAWith3By3MatrixB(a, b), 3, 3)
            assertContentEquals(naiveMultiply(a, b, 3, 3, 3), got)
        }
    }

    @Test
    fun matrixMultiply_3x3_by_3x4_matchesNaive() {
        val rng = Random(0x334A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 9)
            val b = randomMatrix(rng, 12)
            val got = reorderResult(multiply3By3MatrixAWith3By4MatrixB(a, b), 3, 4)
            assertContentEquals(naiveMultiply(a, b, 3, 3, 4), got)
        }
    }

    @Test
    fun matrixMultiply_3x3_by_3x5_matchesNaive() {
        val rng = Random(0x335A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 9)
            val b = randomMatrix(rng, 15)
            val got = reorderResult(multiply3By3MatrixAWith3By5MatrixB(a, b), 3, 5)
            assertContentEquals(naiveMultiply(a, b, 3, 3, 5), got)
        }
    }

    @Test
    fun matrixMultiply_3x4_by_4x4_matchesNaive() {
        val rng = Random(0x344A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 12)
            val b = randomMatrix(rng, 16)
            val got = reorderResult(multiply3By4MatrixAWith4By4MatrixB(a, b), 3, 4)
            assertContentEquals(naiveMultiply(a, b, 3, 4, 4), got)
        }
    }

    @Test
    fun matrixMultiply_3x4_by_4x5_matchesNaive() {
        val rng = Random(0x345A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 12)
            val b = randomMatrix(rng, 20)
            val got = reorderResult(multiply3By4MatrixAWith4By5MatrixB(a, b), 3, 5)
            assertContentEquals(naiveMultiply(a, b, 3, 4, 5), got)
        }
    }

    @Test
    fun matrixMultiply_3x5_by_5x5_matchesNaive() {
        val rng = Random(0x355A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 15)
            val b = randomMatrix(rng, 25)
            val got = reorderResult(multiply3By5MatrixAWith5By5MatrixB(a, b), 3, 5)
            assertContentEquals(naiveMultiply(a, b, 3, 5, 5), got)
        }
    }

    @Test
    fun matrixMultiply_3x5_by_5x9_matchesNaive() {
        val rng = Random(0x359A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 15)
            val b = randomMatrix(rng, 45)
            val got = reorderResult(multiply3By5MatrixAWith5By9MatrixB(a, b), 3, 9)
            assertContentEquals(naiveMultiply(a, b, 3, 5, 9), got)
        }
    }

    @Test
    fun matrixMultiply_4x4_by_4x4_matchesNaive() {
        val rng = Random(0x444A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 16)
            val b = randomMatrix(rng, 16)
            val got = reorderResult(multiply4By4MatrixAWith4By4MatrixB(a, b), 4, 4)
            assertContentEquals(naiveMultiply(a, b, 4, 4, 4), got)
        }
    }

    @Test
    fun matrixMultiply_4x4_by_4x5_matchesNaive() {
        val rng = Random(0x445A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 16)
            val b = randomMatrix(rng, 20)
            val got = reorderResult(multiply4By4MatrixAWith4By5MatrixB(a, b), 4, 5)
            assertContentEquals(naiveMultiply(a, b, 4, 4, 5), got)
        }
    }

    @Test
    fun matrixMultiply_4x5_by_5x5_matchesNaive() {
        val rng = Random(0x455A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 20)
            val b = randomMatrix(rng, 25)
            val got = reorderResult(multiply4By5MatrixAWith5By5MatrixB(a, b), 4, 5)
            assertContentEquals(naiveMultiply(a, b, 4, 5, 5), got)
        }
    }

    @Test
    fun matrixMultiply_4x5_by_5x9_matchesNaive() {
        val rng = Random(0x459A)
        repeat(RANDOM_TEST_ITERATIONS) {
            val a = randomMatrix(rng, 20)
            val b = randomMatrix(rng, 45)
            val got = reorderResult(multiply4By5MatrixAWith5By9MatrixB(a, b), 4, 9)
            assertContentEquals(naiveMultiply(a, b, 4, 5, 9), got)
        }
    }

    @Test
    fun matrixMultiply_3x4_by_4x11_zerosAndOnes() {
        val zerosA = IntArray(12)
        val zerosB = IntArray(44)
        assertContentEquals(IntArray(33), multiply3By4MatrixAWith4By11MatrixB(zerosA, zerosB))

        val onesA = IntArray(12) { 1 }
        val onesB = IntArray(44) { 1 }
        val expectedOnes = IntArray(33) { 4 }
        assertContentEquals(expectedOnes, multiply3By4MatrixAWith4By11MatrixB(onesA, onesB))
    }

    @Test
    fun matrixMultiply_3x9_by_9x11_zerosAndRegression() {
        val zerosA = IntArray(27)
        val zerosB = IntArray(99)
        assertContentEquals(IntArray(33), multiply3By9MatrixAWith9By11MatrixB(zerosA, zerosB))

        val onesA = IntArray(27) { 1 }
        val onesB = IntArray(99) { 1 }
        val expectedOnes = intArrayOf(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 9, 0, 0, 0)
        assertContentEquals(expectedOnes, multiply3By9MatrixAWith9By11MatrixB(onesA, onesB))
    }

    @Test
    fun matrixMultiply_3x3_identityIsIdentity() {
        val identity = intArrayOf(1, 0, 0, 0, 1, 0, 0, 0, 1)
        val a = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9)
        assertContentEquals(a, reorderResult(multiply3By3MatrixAWith3By3MatrixB(a, identity), 3, 3))
        assertContentEquals(a, reorderResult(multiply3By3MatrixAWith3By3MatrixB(identity, a), 3, 3))
    }
}

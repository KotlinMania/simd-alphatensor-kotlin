package com.kotlinmania.simd

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals

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

private fun randomMatrix(rng: Random, size: Int, bound: Int = 100): IntArray =
    IntArray(size) { rng.nextInt(-bound, bound) }

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
    fun `2x2 by 2x2 matches naive`() {
        val rng = Random(0xA1FA)
        repeat(64) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 4)
            val got = reorderResult(multiply2By2MatrixAWith2By2MatrixB(a, b), 2, 2)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 2), got)
        }
    }

    @Test
    fun `2x2 by 2x3 matches naive`() {
        val rng = Random(0xA1FB)
        repeat(64) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 6)
            val got = reorderResult(multiply2By2MatrixAWith2By3MatrixB(a, b), 2, 3)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 3), got)
        }
    }

    @Test
    fun `2x2 by 2x4 matches naive`() {
        val rng = Random(0xA1FC)
        repeat(64) {
            val a = randomMatrix(rng, 4)
            val b = randomMatrix(rng, 8)
            val got = reorderResult(multiply2By2MatrixAWith2By4MatrixB(a, b), 2, 4)
            assertContentEquals(naiveMultiply(a, b, 2, 2, 4), got)
        }
    }

    @Test
    fun `3x3 by 3x3 matches naive`() {
        val rng = Random(0xA1FD)
        repeat(64) {
            val a = randomMatrix(rng, 9)
            val b = randomMatrix(rng, 9)
            val got = reorderResult(multiply3By3MatrixAWith3By3MatrixB(a, b), 3, 3)
            assertContentEquals(naiveMultiply(a, b, 3, 3, 3), got)
        }
    }

    @Test
    fun `4x4 by 4x4 matches naive`() {
        val rng = Random(0xA1FE)
        repeat(64) {
            val a = randomMatrix(rng, 16)
            val b = randomMatrix(rng, 16)
            val got = reorderResult(multiply4By4MatrixAWith4By4MatrixB(a, b), 4, 4)
            assertContentEquals(naiveMultiply(a, b, 4, 4, 4), got)
        }
    }

    @Test
    fun `3x3 identity is identity`() {
        val identity = intArrayOf(1, 0, 0, 0, 1, 0, 0, 0, 1)
        val a = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9)
        assertContentEquals(a, reorderResult(multiply3By3MatrixAWith3By3MatrixB(a, identity), 3, 3))
        assertContentEquals(a, reorderResult(multiply3By3MatrixAWith3By3MatrixB(identity, a), 3, 3))
    }
}

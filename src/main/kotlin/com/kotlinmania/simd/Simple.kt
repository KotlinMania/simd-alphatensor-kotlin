package com.kotlinmania.simd

/**
 * Simple example demonstrating AlphaTensor matrix multiplication algorithms.
 * 
 * Original Rust implementation by: drbh (david.richard.holtz@gmail.com)
 * Original repository: https://github.com/drbh/simd-alphatensor-rs
 * Kotlin translation by: @sydneyrenee / sydney@solace.ofharmony.ai (The Solace Project) / KotlinMania
 */

fun main() {
    // Example 2x2 matrix multiplication
    val result2x2 = multiply2By2MatrixAWith2By2MatrixB(
        intArrayOf(1000, 2000, 3000, 4000),
        intArrayOf(3000, 4000, 5000, 6000)
    )
    println("Example of 2x2 * 2x2: ${result2x2.contentToString()}")
    
    // Example 4x4 matrix multiplication  
    val result4x4 = multiply4By4MatrixAWith4By4MatrixB(
        intArrayOf(
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000
        ),
        intArrayOf(
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000,
            1000, 2000, 3000, 4000
        )
    )
    println("Example of 4x4 * 4x4: ${result4x4.contentToString()}")
}

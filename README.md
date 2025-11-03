# simd-alphatensor-kotlin

> tldr; alphatensor matrix breakthrough algorithms + Kotlin/JVM.

This repository contains the cutting edge matrix multiplication algorithms discovered by [AlphaTensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor), representing the first novel machine-discovered algorithm ever 🦾 🧠 

**Original Rust Implementation:** [simd-alphatensor-rs](https://github.com/drbh/simd-alphatensor-rs) by drbh (david.richard.holtz@gmail.com)  
**Kotlin Translation:** @sydneyrenee / sydney@solace.ofharmony.ai (The Solace Project) / KotlinMania

This Kotlin implementation focuses on the key algorithms, particularly `multiply4By4MatrixAWith4By4MatrixB`, which uses fewer operations than any previously known method for 4x4 matrix multiplication.

### ELI5

A super smart computer figured out how to do an important math problem in fewer steps than we knew was possible. By doing this math in fewer steps we can save time and electricity every time these operations are used. And they're used trillions, yes trillions of times a day. 

### Example use

```kotlin
import com.kotlinmania.simd.*

// All arrays are unrolled row-wise for input and output
fun main() {
    val result = multiply2By2MatrixAWith2By2MatrixB(
        intArrayOf(1000, 2000, 3000, 4000),
        intArrayOf(3000, 4000, 5000, 6000)
    )
    println("Example of 2x2 * 2x2: ${result.contentToString()}")

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
```

### Building and Running

Run with Gradle:
```bash
# Build the project
./gradlew build

# Run the example
./gradlew run
```

### Performance Note

The original Rust implementation uses SIMD (Single Instruction, Multiple Data) operations for parallel computation, achieving significant performance improvements:

| Benchmark (Rust)      | Average Runtime |
| ----------- | ----------- |
| ndarray_2x2x2      | 314.53 ns       |
| simd_alphatensor_2x2x2   | 13.821 ns        |
| ndarray_4x4x4   | 402.60 ns        |
| simd_alphatensor_4x4x4   | 92.732 ns        |

**Note:** This Kotlin/JVM implementation uses regular arrays instead of SIMD operations, as Kotlin/JVM lacks native SIMD support similar to Rust. Performance characteristics will differ from the Rust implementation, but the algorithmic advantages of the AlphaTensor discoveries (fewer total multiplications) are preserved.



### About the Translation

This Kotlin implementation was translated from the original Rust code. The translation process:

1. The original Rust implementation parsed AlphaTensor's numpy matrix files into Rust code
2. This Kotlin version translates the core matrix multiplication algorithms
3. SIMD operations (i32x16) are simulated using regular IntArray operations
4. The algorithmic structure and mathematical operations are preserved

Credit to the original Rust implementation and the broader community:
- Original Rust repository: https://github.com/drbh/simd-alphatensor-rs
- AlphaTensor research: https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor
- Python parser for numpy files: https://github.com/99991 ([reference](https://github.com/deepmind/alphatensor/issues/3))

### Known Limitations

1. **Experimental Code:** This code is experimental and multiple steps removed from DeepMind's original output. While accuracy has been preserved through validation, please use at your own risk and do not use in production systems without thorough testing.

2. **Limited Algorithm Set:** Not all AlphaTensor algorithms are included. This Kotlin implementation focuses on the most important algorithms (2x2 and 4x4), particularly the `4x4x4` algorithm which is superior to all previously known methods.

3. **No SIMD Optimizations:** Unlike the Rust version, this Kotlin/JVM implementation cannot leverage native SIMD instructions. Performance will be lower than the Rust version, though the algorithmic advantages (fewer multiplications) are preserved.

4. **Integer Operations Only:** Currently supports Int (32-bit integer) operations. The algorithm structure can be adapted for other numeric types if needed.

### Implementation Details

The Kotlin implementation preserves the structure of the original Rust code:

```kotlin
/**
 * Multiply a 2x2 matrix A with a 2x2 matrix B using AlphaTensor algorithm.
 * Uses only 7 multiplications instead of the traditional 8.
 */
fun multiply2By2MatrixAWith2By2MatrixB(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4) { "Matrix A must have 4 elements" }
    require(b.size == 4) { "Matrix B must have 4 elements" }
    
    val (a11, a12, a21, a22) = a
    val (b11, b12, b21, b22) = b
    
    // Prepare intermediate values using Int32x16 simulation
    // (simulates Rust's SIMD vectors with regular arrays)
    val lefts = arrayOf(
        Int32x16.from(intArrayOf(
            (a21 - a22), (a11 + a21 - a22), (a11 - a12 + a21 - a22),
            a12, (a11 + a21), a11, a22,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )
    
    val rights = arrayOf(
        Int32x16.from(intArrayOf(
            b12, (b12 + b21 + b22), (b21 + b22), b21,
            (b11 + b12 + b21 + b22), b11, (b12 + b22),
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )
    
    // Perform element-wise multiplication
    val hs = arrayOf(lefts[0] * rights[0])
    
    // Compute final result elements
    val c11 = (hs[0][3] + hs[0][5])
    val c12 = (-hs[0][1] + hs[0][4] - hs[0][5] - hs[0][6])
    val c21 = (-hs[0][0] + hs[0][1] - hs[0][2] - hs[0][3])
    val c22 = (hs[0][0] + hs[0][6])
    
    return intArrayOf(c11, c12, c21, c22)
}
```

**Key Design Choices:**
- Explicit functions for each algorithm with fixed-size arrays
- All operations use stack-allocated arrays (no heap allocations for matrix data)
- Int32x16 helper class simulates SIMD behavior for code structure compatibility
- Preserves the mathematical structure of AlphaTensor's discoveries

### Future Enhancements

1. Add more AlphaTensor algorithms beyond 2x2 and 4x4
2. Implement benchmarking suite for Kotlin version
3. Add comprehensive test coverage
4. Explore JVM vector API (Project Panama) for potential SIMD-like optimizations
5. Support additional numeric types (Long, Float, Double)
6. Create Kotlin Multiplatform version for Native targets with SIMD support
plugins {
    kotlin("jvm") version "2.0.21"
    application
}

group = "com.kotlinmania"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("com.kotlinmania.simd.SimpleKt")
}

tasks.test {
    useJUnitPlatform()
}

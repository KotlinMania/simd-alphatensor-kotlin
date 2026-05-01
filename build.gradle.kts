plugins {
    kotlin("jvm") version "2.3.20"
    application
}

group = "com.kotlinmania"
version = "0.1.1"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
}

application {
    mainClass.set("com.kotlinmania.simd.SimpleKt")
}

tasks.test {
    useJUnitPlatform()
}

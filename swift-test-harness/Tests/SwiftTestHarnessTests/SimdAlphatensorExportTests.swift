#if canImport(Testing)
import Testing
import SimdAlphatensor

@Suite("SimdAlphatensor Swift Export Suite")
struct SimdAlphatensorExportTests {
    @Test("Swift module loads cleanly")
    func swiftModuleLoads() {
        #expect(Bool(true), "SimdAlphatensor swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import SimdAlphatensor

final class SimdAlphatensorExportTests: XCTestCase {
    func testSwiftModuleLoads() {
        XCTAssertTrue(true, "SimdAlphatensor swift module imported cleanly")
    }
}
#endif

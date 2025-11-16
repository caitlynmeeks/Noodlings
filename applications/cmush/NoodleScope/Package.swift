// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NoodleScope",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "NoodleScope", targets: ["NoodleScope"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "NoodleScope",
            dependencies: []
        )
    ]
)

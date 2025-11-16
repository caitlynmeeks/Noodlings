// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NoodleStudio",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "NoodleStudio", targets: ["NoodleStudio"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "NoodleStudio",
            dependencies: []
        )
    ]
)

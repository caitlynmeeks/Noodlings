#!/bin/bash
# Create macOS .app bundle for NoodleStudio

APP_NAME="NoodleStudio"
BUNDLE_DIR="$APP_NAME.app"
CONTENTS_DIR="$BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean old bundle
rm -rf "$BUNDLE_DIR"

# Create bundle structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create launcher script with logging
cat > "$MACOS_DIR/$APP_NAME" << 'EOF'
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../../../" && pwd)"
cd "$DIR"

# Log for debugging
LOG_FILE="$HOME/.noodlestudio/launch.log"
mkdir -p "$HOME/.noodlestudio"
echo "=== Launch $(date) ===" >> "$LOG_FILE"
echo "Dir: $DIR" >> "$LOG_FILE"

source venv/bin/activate >> "$LOG_FILE" 2>&1
python run_studio.py >> "$LOG_FILE" 2>&1
EOF

chmod +x "$MACOS_DIR/$APP_NAME"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>NoodleStudio</string>
    <key>CFBundleIdentifier</key>
    <string>com.noodlings.noodlestudio</string>
    <key>CFBundleName</key>
    <string>NoodleStudio</string>
    <key>CFBundleDisplayName</key>
    <string>NoodleStudio</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Copy icon if it exists
if [ -f "NoodleStudio.iconset/icon.icns" ]; then
    cp "NoodleStudio.iconset/icon.icns" "$RESOURCES_DIR/AppIcon.icns"
    echo "✓ Icon copied"
elif [ -d "NoodleStudio.iconset" ]; then
    # Generate .icns from iconset
    iconutil -c icns "NoodleStudio.iconset" -o "$RESOURCES_DIR/AppIcon.icns"
    echo "✓ Icon generated from iconset"
fi

echo "✓ NoodleStudio.app created!"
echo "To install: drag NoodleStudio.app to /Applications or double-click to run"

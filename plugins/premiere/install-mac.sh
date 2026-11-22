#!/bin/sh
set -eu
destination="$HOME/Library/Application Support/Adobe/CEP/extensions/com.alx8g.fakethermal.panel"
if [ "${1:-}" = "--uninstall" ]; then
    rm -rf "$destination"
    echo "Fake Thermal uninstalled. Shared PlayerDebugMode settings were retained."
    exit 0
fi
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$destination"
cp -R "$source_dir/CSXS" "$source_dir/css" "$source_dir/js" "$source_dir/jsx" "$source_dir/index.html" "$destination/"
for version in 9 10 11 12; do
    defaults write "com.adobe.CSXS.$version" PlayerDebugMode -string 1
done
echo "Installed. Restart Premiere, then open Window > Extensions (Legacy) > Fake Thermal."

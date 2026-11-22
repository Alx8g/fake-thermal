#!/bin/sh
set -eu
root=${FAKE_THERMAL_HOME:-"$HOME/.fake-thermal"}
if [ "$(uname -s)" = Darwin ]; then root=${FAKE_THERMAL_HOME:-"$HOME/Library/Application Support/fake-thermal"}; fi
if [ "${1:-}" = --uninstall ] || [ "${FAKE_THERMAL_UNINSTALL:-0}" = 1 ]; then
 if [ ! -d "$root" ]; then echo 'fake-thermal is not installed.'; exit 0; fi
 "$root/venv/bin/python" "$root/installer/scripts/install_user.py" --root "$root" --uv "$root/bootstrap/uv" --source "$root/installer" --uninstall
 exit $?
fi
mkdir -p "$root/bootstrap"
case "$(uname -s)-$(uname -m)" in
 Darwin-arm64) target=aarch64-apple-darwin;;
 Darwin-x86_64) target=x86_64-apple-darwin;;
 Linux-x86_64) target=x86_64-unknown-linux-gnu;;
 Linux-aarch64) target=aarch64-unknown-linux-gnu;;
 *) echo 'Unsupported platform' >&2; exit 1;;
esac
curl -fLsS "https://github.com/astral-sh/uv/releases/download/0.9.18/uv-$target.tar.gz" -o "$root/bootstrap/uv.tar.gz"
tar -xzf "$root/bootstrap/uv.tar.gz" -C "$root/bootstrap"
uv="$root/bootstrap/uv-$target/uv"
export UV_PYTHON_INSTALL_DIR="$root/python"
"$uv" python install 3.11
"$uv" venv --python 3.11 --managed-python --allow-existing "$root/venv"
source_dir=${FAKE_THERMAL_SOURCE:-}
if [ -z "$source_dir" ]; then
 curl -fLsS "https://codeload.github.com/Alx8g/fake-thermal/tar.gz/${FAKE_THERMAL_REF:-main}" -o "$root/bootstrap/source.tar.gz"
 rm -rf "$root/source"
 mkdir -p "$root/source"
 tar -xzf "$root/bootstrap/source.tar.gz" --strip-components=1 -C "$root/source"
 source_dir="$root/source"
fi
"$root/venv/bin/python" "$source_dir/scripts/install_user.py" --root "$root" --uv "$uv" --source "$source_dir" "$@"

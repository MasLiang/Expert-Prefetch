#!/usr/bin/env bash
set -euo pipefail
exec make -C "$(cd "$(dirname "$0")/.." && pwd)" rtl-test

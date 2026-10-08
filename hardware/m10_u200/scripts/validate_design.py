"""Run reproducible RTL/golden/lint/Python checks and record honest sign-off status."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def main() -> None:
    """Write logs and a machine-readable functional baseline validation report."""
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    checks = []
    commands = (
        ("rtl", ["make", "-C", str(ROOT), "rtl-test"]),
        ("golden", ["make", "-C", str(ROOT), "golden"]),
        ("lint", ["make", "-C", str(ROOT), "lint"]),
        ("python", ["pytest", "-o", "addopts=", "-q"]),
    )
    for name, command in commands:
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        output = result.stdout + result.stderr
        (reports / f"validation_{name}.log").write_text(output, encoding="utf-8")
        # 位宽和变量遮蔽也作为验收失败项，禁止新增隐式截断后继续报告 PASS。
        structural = re.findall(
            r"%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT|BLKANDNBLK|WIDTHEXPAND|WIDTHTRUNC|VARHIDDEN)",
            output,
        )
        passed = result.returncode == 0 and not structural
        checks.append({"name": name, "command": command, "passed": passed,
                       "returncode": result.returncode, "pass_messages": output.count("PASS:"),
                       "structural_warnings": structural,
                       "warning_count": output.count("%Warning-")})
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    incomplete = [
        "FullExpert/BudgetProportional policies are not integrated in numeric top",
        "four physical AXI banks integrated, but DDR PHY/shell/CDC and measured bandwidth pending",
        "complete freeze-gate concurrency and out-of-order matrix not covered",
        "single-variable always coding style not fully applied",
        "board XPFM/host/XRT/CDC/PnR/timing sign-off outstanding",
    ]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((ROOT / "rtl").glob("*.sv"))}
    passed = all(check["passed"] for check in checks)
    report = {"functional_baseline_passed": passed, "complete_spec_signoff": False,
              "checks": checks, "remaining_signoff_items": incomplete,
              "rtl_sha256": source_hashes,
              "claim_boundary": "Functional checks and installed-device OOC synthesis do not imply board deployment or timing closure."}
    (reports / "design_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Design validation failed; see reports/validation_*.log")


if __name__ == "__main__":
    main()

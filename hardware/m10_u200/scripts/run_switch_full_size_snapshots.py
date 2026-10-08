"""隔离运行已验收源 hash 的完整尺寸 snapshot，避免共享 runtime 文件冲突。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from run_switch_full_size import CASES, PROJECT, command_run, parse_result


def run_case(case: str, origin: Path, output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """每个 case 独立复制编译产物，不改变 RTL、snapshot 参数或 checker。"""
    whole, prefetch = CASES[case]
    directory = output/case
    build = directory/'runtime'
    build.mkdir(parents=True, exist_ok=True)
    snapshot = f'full_switch_whole_{whole}'
    source = origin/'build/xsim.dir'/snapshot
    if not (source/'xsimk').is_file():
        raise ValueError('compiled full-size snapshot missing')
    target = build/'xsim.dir'/snapshot
    shutil.copytree(source, target)
    work = origin/'build/xsim.dir/work'
    if work.exists():
        shutil.copytree(work, build/'xsim.dir/work')
    ini = origin/'build/xsim.ini'
    if ini.exists():
        shutil.copy2(ini, build/'xsim.ini')
    source_kernel = hashlib.sha256((source/'xsimk').read_bytes()).hexdigest()
    if source_kernel != hashlib.sha256((target/'xsimk').read_bytes()).hexdigest():
        raise ValueError('copied kernel changed')
    batch = directory/'run.tcl'
    batch.write_text('run 100 ms\nquit\n', encoding='utf-8')
    log = directory/'simulation.log'
    responder = manifest['responder']
    wall = command_run(['xsim', snapshot, '--tclbatch', str(batch), '--onerror', 'quit',
        '--onfinish', 'quit', '--log', str(log), '--testplusarg', f'PREFETCH={prefetch}',
        '--testplusarg', f'PATTERN={manifest["predictor_pattern"]}',
        '--testplusarg', f'STARTUP={responder["startup_cycles"]}',
        '--testplusarg', f'BEAT_GAP={responder["beat_gap_cycles"]}',
        '--testplusarg', f'ALIGNMENT_DEBUG={int(manifest.get("alignment_debug", False))}',
        '--testplusarg', f'TIMELINE={directory/"timeline.csv"}'], build, directory/'console.log')
    result = dict(case=case, wall_seconds=wall, snapshot_sha256=source_kernel,
                  **parse_result(log, manifest['geometry'], whole, prefetch))
    (directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'{case}: PASS cycles={result["metrics"]["cycles"]} wall={wall:.1f}s', flush=True)
    return result


def main() -> None:
    """在有限并发数下复用两种编译配置；源/产物 hash 和功能验收不放宽。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    origin, output = args.origin.resolve(), args.output.resolve()
    manifest = json.loads((origin/'summary.json').read_text(encoding='utf-8'))
    for path, expected in manifest['source_sha256'].items():
        if hashlib.sha256((PROJECT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'compiled-source provenance mismatch: {path}')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, case, origin, output, manifest) for case in args.cases]
        for future in futures:
            results.append(future.result())
    for path, expected in manifest['source_sha256'].items():
        if hashlib.sha256((PROJECT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source changed during simulation: {path}')
    (output/'summary.json').write_text(json.dumps(dict(status='PASS',
        configuration=manifest, results=results), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

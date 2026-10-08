"""按显式白名单生成可移植源码快照；不删除原文件、不提交或上传 Git。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('src', 'tests', 'configs', 'examples', 'docs', 'scripts',
               'hardware/m10_u200/rtl', 'hardware/m10_u200/tb',
               'hardware/m10_u200/scripts', 'hardware/m10_u200/configs',
               'hardware/m10_u200/traces')
FIXTURE_DIRS = ('m72_policy_phase', 'm72a_paper_figure_cleanup',
                'm72b_opportunity_refinement', 'm73_fidelity_audit',
                'm8_tile_scheduler', 'm9_hardware_scheduler',
                'm91_realistic_evaluation', 'm92_criticality_scheduler',
                'm93_predictor_sensitivity')
SUFFIXES = frozenset(('.py', '.toml', '.yaml', '.yml', '.json', '.jsonl', '.csv',
                      '.sv', '.v', '.vh', '.svh', '.tcl', '.xdc', '.sh', '.md',
                      '.txt', '.mem', '.hex', '.coe', '.png', '.pdf'))


def selected_files(root: Path) -> list[Path]:
    """返回排序后的源码与有限测试夹具，拒绝符号链接与大文件。"""
    paths = [root/'README.md', root/'pyproject.toml', root/'.gitignore',
             root/'hardware/m10_u200/Makefile', root/'hardware/m10_u200/README.md',
             root/'results/prefetch_dense/inputs.json']
    paths += list(root.glob('M*.md'))
    for name in SOURCE_DIRS + tuple('results/'+s for s in FIXTURE_DIRS):
        for path in (root/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in SUFFIXES:
                paths.append(path)
    result = sorted(set(paths), key=lambda p: str(p.relative_to(root)))
    for path in result:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'无法打包非普通文件：{path}')
        if path.stat().st_size >= 100*1024*1024:
            raise ValueError(f'文件超过100MiB，需另行决定发布方式：{path}')
    return result


def prepare(root: Path, output: Path) -> dict[str, Any]:
    """复制白名单快照，生成逐文件SHA256清单与压缩包；拒绝覆盖。"""
    archive = output.with_name(output.name+'.tar.gz')
    if output.exists() or archive.exists():
        raise ValueError('输出目录/压缩包已存在，拒绝覆盖')
    paths = selected_files(root)
    output.mkdir(parents=True)
    records = []
    for path in paths:
        relative = path.relative_to(root)
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append(dict(path=str(relative), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    manifest = dict(release_id=output.name, scope='single-token simulator + U200 RTL source snapshot',
        files=records, file_count=len(records), total_bytes=sum(r['bytes'] for r in records),
        git_history_included=False, validated_hardware='behavioral simulation, not board deployment',
        experimental_batch_features='preserved for regression only; not included in single-token claims',
        excluded=['all Vivado generated projects/builds/reports', 'all credentials/tool-private directories',
                  'experiment outputs except explicit regression fixtures'],
        license='not selected; no new license assigned')
    (output/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(output, arcname=output.name)
    return dict(directory=str(output), archive=str(archive), file_count=len(records),
                source_bytes=manifest['total_bytes'], archive_bytes=archive.stat().st_size)


def main() -> None:
    """准备快照；不执行 git init/add/commit/push 或破坏性清理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/edge-moe-single-token-20261008')
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.output.resolve()), indent=2))


if __name__ == '__main__':
    main()

"""Execute notebooks in fresh kernels and retain outputs separately from learner files."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='*.ipynb', help='Notebook filename glob, e.g. 17_*.ipynb')
    parser.add_argument('--cuda', action='store_true', help='Enable the optional CUDA AMP experiment')
    parser.add_argument('--retry-failed', action='store_true', help='Rerun failed entries in the full report and update it')
    args = parser.parse_args()
    os.environ['PYTORCH_COURSE_CUDA'] = '1' if args.cuda else '0'
    # A dedicated kernel spec always uses the interpreter running this tool.
    output_dir = ROOT / 'validation' / 'executed'
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir = ROOT / 'validation' / 'kernel'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Course validation', 'language': 'python',
    }), encoding='utf-8')
    manager = KernelSpecManager()
    manager.install_kernel_spec(str(kernel_dir), kernel_name='pytorch-course-validation', user=True)
    previous_report = None
    if args.retry_failed:
        previous_report = json.loads((ROOT / 'validation' / 'report.json').read_text(encoding='utf-8'))
        if previous_report['cuda_experiment_enabled'] != args.cuda:
            raise SystemExit('Use the same --cuda setting as the original full validation run.')
        notebooks = [ROOT / 'notebooks' / row['notebook'] for row in previous_report['results'] if row['status'] != 'passed']
    else:
        notebooks = sorted((ROOT / 'notebooks').glob(args.pattern))
    if not notebooks:
        raise SystemExit('No notebooks matched.')
    results = []
    for path in notebooks:
        started = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        status, error = 'passed', None
        try:
            nbformat.validate(notebook)
            NotebookClient(notebook, timeout=240, kernel_name='pytorch-course-validation',
                           resources={'metadata': {'path': str(ROOT / 'notebooks')}},
                           allow_errors=False).execute()
        except Exception:
            status, error = 'failed', traceback.format_exc()
        nbformat.write(notebook, output_dir / path.name)
        result = {'notebook': path.name, 'status': status,
                  'seconds': round(time.perf_counter() - started, 2)}
        if error:
            result['error'] = error
        results.append(result)
        print(f"{status.upper():6} {path.name} ({result['seconds']}s)", flush=True)
        if error:
            print(error, flush=True)
    if previous_report is not None:
        updates = {row['notebook']: row for row in results}
        results = [updates.get(row['notebook'], row) for row in previous_report['results']]
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'numpy', 'matplotlib', 'scikit-learn', 'nbformat', 'nbclient', 'jupyterlab']},
        'cuda_experiment_enabled': args.cuda,
        'inductor_experiment_enabled': False,
        'results': results,
    }
    report_name = 'report.json' if args.pattern == '*.ipynb' else 'report_selected.json'
    (ROOT / 'validation' / report_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    failed = sum(r['status'] != 'passed' for r in results)
    print(f'{len(results) - failed}/{len(results)} notebooks passed.', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

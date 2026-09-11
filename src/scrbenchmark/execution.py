"""Record complete and partial benchmark outcomes for automation."""

import json
from pathlib import Path


def write_run_status(output: Path, algorithms: list[str], n_repeats: int, results, failures=(), *, final: bool = True) -> int:
    """Persist missing runs and return a nonzero exit status for incomplete work."""
    completed = {(result.algorithm_name, result.run_id) for result in results}
    errors = {(item['algorithm'], item['run_id']): item['error'] for item in failures}
    missing = [
        {'algorithm': name, 'run_id': run_id, 'error': errors.get((name, run_id), 'No result produced')}
        for name in algorithms for run_id in range(n_repeats) if (name, run_id) not in completed
    ]
    payload = {
        'status': ('partial_failure' if completed else 'failed') if missing else ('completed' if final else 'running'),
        'requested_runs': len(algorithms) * n_repeats,
        'completed_runs': len(completed),
        'failures': missing,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_status.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return int(bool(missing))

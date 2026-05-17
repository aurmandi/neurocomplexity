import subprocess
import sys


def test_benchmark_help_lists_subcommand():
    out = subprocess.run(
        [sys.executable, "-m", "neurocomplexity", "benchmark", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "benchmark" in out.stdout.lower() or "case" in out.stdout.lower()


def test_benchmark_runs_single_case(tmp_path):
    out_csv = tmp_path / "bench.csv"
    out = subprocess.run(
        [sys.executable, "-m", "neurocomplexity", "benchmark",
         "--case", "pid.atoms_xor", "--reps", "2",
         "-o", str(out_csv)],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out_csv.exists()

"""CPU-only controller checks using synthetic producers and checkers."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(all(shutil.which(name) for name in ("bash", "timeout", "sha256sum")), "Requires shell utilities")
class TmemControlTests(unittest.TestCase):
    def fixture(self, root, corrupt=False):
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / "lib").mkdir()
        (bundle / "sanitizer-new").mkdir()
        source = Path(__file__).parents[1] / "benchmarks/diagnostics/run_tmem_init.sh"
        shutil.copy2(source, bundle / source.name)
        (bundle / "source.cubin").write_bytes(b"\x7fELF synthetic tmem_init_allocation tmem_init_ordinary fixture")
        helpers = root / "helpers"
        helpers.mkdir()
        # Lock scheduling is not simulated: this test exercises collection/status logic only.
        scripts = {
            helpers / "flock": "import sys; sys.exit(0)\n",
            helpers / "setsid": "import os,sys; os.setsid(); os.execvp(sys.argv[1],sys.argv[1:])\n",
            bundle
            / "tmem_init": """import json,sys
from pathlib import Path
producer, initialization, barrier, capture, output = sys.argv[1:]
p = Path(output)
(p / 'report.json').write_text(json.dumps({'all_match':True,'captured_modules':1},separators=(',',':')))
(p / 'launches.jsonl').write_text(json.dumps({'symbol':'tmem_init_'+producer},separators=(',',':')))
(p / 'loaded-0.cubin').write_bytes(b'CORRUPT' if CORRUPT else Path('source.cubin').read_bytes())
""".replace(
                "CORRUPT else", str(corrupt) + " else"
            ),
        }
        checker = """import json,subprocess,sys
from pathlib import Path
if sys.argv[1:] == ['--version']:
 print('synthetic-new' if 'sanitizer-new' in __file__ else 'synthetic-old'); sys.exit(0)
args=sys.argv[1:]
with Path('checker-calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
i=args.index('--error-exitcode')+2
status=subprocess.call(args[i:])
if status: sys.exit(status)
print('========= ERROR SUMMARY: 0 errors')
sys.exit(99 if args[i+1] == 'allocation' else 0)
"""
        scripts[bundle / "sanitizer-new/compute-sanitizer"] = checker
        scripts[helpers / "compute-sanitizer"] = checker
        for path, code in scripts.items():
            path.write_text("#!" + sys.executable + "\n" + code)
            path.chmod(0o755)
        (bundle / "SHA256SUMS").write_text(
            "".join(
                hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.relative_to(bundle)) + "\n"
                for path in sorted(bundle.rglob("*"))
                if path.is_file()
            )
        )
        return bundle, dict(os.environ, PATH=str(helpers) + os.pathsep + os.environ["PATH"])

    def collect(self, root, bundle, env, mode="controls"):
        result = root / "results"
        status = subprocess.run(
            [
                "bash",
                "-c",
                'exec 9>"$1"; exec bash "$2" "$3" "$4"',
                "controller",
                str(root / "lock"),
                str(bundle / "run_tmem_init.sh"),
                str(result),
                mode,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        return status, result

    def test_complete_factorial_preserves_nonzero_status_and_unlimited_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, env = self.fixture(root)
            status, result = self.collect(root, bundle, env)
            self.assertEqual(status.returncode, 0, status.stderr.decode())
            rows = [row.split("\t") for row in (result / "status.tsv").read_text().splitlines()]
            self.assertEqual(len(rows), 18)
            self.assertEqual(sum(row[1] == "99" for row in rows), 5)
            calls = [json.loads(line) for line in (bundle / "checker-calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 10)
            self.assertTrue(all(call[call.index("--print-limit") + 1] == "0" for call in calls))
            self.assertTrue(all(call[-2] == "disabled" for call in calls))
            # Both tool versions invoke precisely the same executable path.
            self.assertEqual({call[call.index("--error-exitcode") + 2] for call in calls}, {str(bundle / "tmem_init")})
            self.assertIn("not_attention_qualification", (result / "final-status.txt").read_text())

    def test_default_comparison_retains_six_processes_and_original_print_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, env = self.fixture(root)
            status, result = self.collect(root, bundle, env, "baseline")
            self.assertEqual(status.returncode, 0, status.stderr.decode())
            labels = [row.split("\t")[0] for row in (result / "status.tsv").read_text().splitlines()]
            self.assertEqual(
                labels,
                [
                    "physical-allocation",
                    "physical-ordinary",
                    "old-allocation",
                    "old-ordinary",
                    "new-allocation",
                    "new-ordinary",
                ],
            )
            calls = [json.loads(line) for line in (bundle / "checker-calls.jsonl").read_text().splitlines()]
            self.assertTrue(all(call[call.index("--print-limit") + 1] == "6" for call in calls))

    def test_wrong_loaded_image_stops_before_any_instrumented_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, env = self.fixture(root, corrupt=True)
            status, result = self.collect(root, bundle, env)
            self.assertNotEqual(status.returncode, 0)
            self.assertFalse((bundle / "checker-calls.jsonl").exists())
            self.assertFalse((result / "final-status.txt").exists())


if __name__ == "__main__":
    unittest.main()

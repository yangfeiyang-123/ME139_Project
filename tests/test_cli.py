"""Regression checks for the launcher without Isaac Sim or GPU dependencies."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="me139 cli ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        self.root.mkdir()
        shutil.copy2(ROOT / "project.sh", self.root)
        (self.root / "scripts").mkdir()
        for name in ("env.sh", "python.sh", "setup.sh", "replay_remote.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)

    def run_cli(self, *args, extra_env=None):
        env = os.environ.copy()
        env.pop("ISAAC_STREAM_HOST", None)
        env.update(extra_env or {})
        return subprocess.run(
            [str(self.root / "project.sh"), *args],
            cwd=self.temp.name, env=env, text=True, capture_output=True,
        )

    def install_probe(self):
        env_bin = self.root / ".venv" / "bin"
        env_bin.mkdir(parents=True)
        (env_bin / "activate").write_text("# Test-only activation stub.\n")
        probe = self.root / "probe.py"
        probe.write_text(
            "import json, os, sys\n"
            "print(json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd(), "
            "root=os.environ['ME139_PROJECT_ROOT'], "
            "tmp=os.environ['TMPDIR'])))\n"
            "sys.exit(int(os.environ.get('PROBE_EXIT', '0')))\n"
        )
        interpreter = env_bin / "python"
        interpreter.write_text(
            "#!/usr/bin/env bash\nexec "
            + shlex.quote(sys.executable) + " " + shlex.quote(str(probe)) + ' "$@"\n'
        )
        interpreter.chmod(0o755)

    def test_help_works_without_environment(self):
        for args in ((), ("help",), ("--help",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def test_invalid_command_fails_before_loading_environment(self):
        result = self.run_cli("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown command", result.stderr)

    def test_missing_environment_explains_setup(self):
        result = self.run_cli("doctor")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ISAAC_BASE_PYTHON", result.stderr)
        self.assertIn("./project.sh setup", result.stderr)

    def test_dispatch_preserves_paths_arguments_and_exit_code(self):
        self.install_probe()
        result = self.run_cli(
            "play", "--motion", "data/motions/path with spaces.npz",
            "--checkpoint", "logs/model.pt", extra_env={"PROBE_EXIT": "7"},
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"], [
            "scripts/train.py", "--play",
            "--motion", "data/motions/path with spaces.npz",
            "--checkpoint", "logs/model.pt",
        ])
        self.assertEqual(payload["cwd"], str(self.root))
        self.assertEqual(payload["root"], str(self.root))
        self.assertEqual(payload["tmp"], str(self.root / ".cache" / "tmp"))

    def test_remote_replay_requires_explicit_host(self):
        result = self.run_cli("replay-remote", "--motion", "data/motions/example.npz")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ISAAC_STREAM_HOST", result.stderr)

    def test_setup_rejects_arguments_before_installing(self):
        result = self.run_cli("setup", "--unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertFalse((self.root / ".venv").exists())


if __name__ == "__main__":
    unittest.main()

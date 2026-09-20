"""Host-stdlib launcher contracts; no Docker daemon needed for these tests."""

import ast
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("benchmark_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
IMAGE = "sha256:" + "a" * 64
SOURCE = {"package_sha256": "b" * 64, "helpers_sha256": "c" * 64}
# Independent oracle for an artifact inside private container tmpfs, not host /tmp.
EXPECTED_SCORE_PATH = "/tmp/scores.json"  # noqa: S108


class FakeDocker:
    def __init__(self, fail=None):
        self.calls = []
        self.alive = {}
        self.fail = fail
        self.bundle = {
            "format": "lm-eval-offline-bundle-v1",
            "source": SOURCE,
            "sha256": "d" * 64,
            "provenance": {"request_cache": False, "response_cache": None},
            "tasks": {
                "humaneval": {
                    "version": 3,
                    "repeats": 1,
                    "dataset": {"fingerprint": "f"},
                    "doc_ids": [0],
                }
            },
        }
        self.result = {
            "format": "lm-eval-offline-results-v1",
            "source": SOURCE,
            "bundle_sha256": "d" * 64,
        }

    def call(self, argv, timeout=60, log=None, binary_output=None, max_bytes=None):
        self.calls.append(argv[:])
        if log is not None:
            launcher.private_text(log, "bounded phase log\n")
        operation = argv[0]
        if binary_output is not None:
            container, source = argv[1], argv[-2]
            if container not in self.alive:
                raise RuntimeError("tmpfs lost after container removal")
            if self.fail == "copy" and source == EXPECTED_SCORE_PATH:
                raise RuntimeError("injected copy")
            value = self.result if source == EXPECTED_SCORE_PATH else self.bundle
            Path(binary_output).write_text(
                "{}\n" if source.endswith("jsonl") else json.dumps(value)
            )
            return ""
        if operation == "image":
            return json.dumps(
                [
                    {
                        "Id": IMAGE,
                        "RepoDigests": ["repo@" + IMAGE],
                        "Config": {
                            "User": "65532:65532",
                            "Env": ["PATH=/opt/venv/bin:/usr/bin"],
                        },
                    }
                ]
            )
        if operation == "create":
            container = str(len([c for c in self.calls if c[0] == "create"])) * 64
            self.alive[container] = argv[:]
            if self.fail == "create":
                self.fail = None
                raise subprocess.TimeoutExpired("docker create", timeout)
            return container
        if operation == "inspect":
            container = argv[1]
            if container not in self.alive:
                container = next(
                    key
                    for key, value in self.alive.items()
                    if value[value.index("--name") + 1] == argv[1]
                )
            create = self.alive[container]
            label = create[create.index("--label") + 1].split("=", 1)[1]
            return json.dumps(
                [
                    {
                        "Id": container,
                        "Image": IMAGE,
                        "Config": {
                            "User": "65532:65532",
                            "Labels": {"lm-eval.run": label},
                            "Env": ["HF_ALLOW_CODE_EVAL=1"],
                        },
                        "HostConfig": {
                            "NetworkMode": "none",
                            "ReadonlyRootfs": True,
                            "CapDrop": ["ALL"],
                            "SecurityOpt": ["no-new-privileges"],
                            "Init": True,
                            "PidsLimit": 128,
                            "Memory": 536870912,
                            "MemorySwap": 536870912,
                            "NanoCpus": 1000000000,
                            # Docker inspection oracle; no host file access.
                            "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=128m,mode=1777"},  # noqa: S108
                        },
                        "Mounts": [
                            {
                                "Type": "bind",
                                "Destination": "/input/score-input.json",
                                "RW": False,
                            }
                        ],
                    }
                ]
            )
        if operation == "exec":
            if "-c" in argv:
                return json.dumps(
                    {"source": SOURCE, "python": "3.13.15", "lock_sha256": "e" * 64}
                )
            phase = "generate" if "generate" in argv else "score"
            if self.fail == phase:
                raise RuntimeError("injected " + phase)
            if self.fail == "interrupt":
                raise KeyboardInterrupt
            if self.fail == "timeout":
                raise subprocess.TimeoutExpired("docker exec", timeout)
            return ""
        if operation == "rm":
            if self.fail == "cleanup" and argv[-1] == "2" * 64:
                self.fail = None
                raise RuntimeError("injected cleanup")
            del self.alive[argv[-1]]
        return ""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "tasks": ["humaneval"],
                    "model": "fixture",
                    "base_url": "http://host.docker.internal:8888/v1/chat/completions",
                    "gen_kwargs": {"temperature": 0.2, "max_gen_toks": 128},
                }
            )
        )
        self.args = launcher.parser().parse_args(
            [
                "--config",
                str(self.config),
                "--output-dir",
                str(self.root / "run"),
                "--image",
                "lm-eval-fork:c1",
            ]
        )

    def test_success_same_image_strict_isolation_and_live_storage(self):
        docker = FakeDocker()
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "private-openai", "HF_TOKEN": "private-hf"}
        ):
            completed = launcher.run(self.args, docker)
        self.assertEqual(completed, self.root / "run" / "completed")
        self.assertFalse(docker.alive)
        creates = [call for call in docker.calls if call[0] == "create"]
        self.assertEqual(len(creates), 2)
        generation, score = creates
        for call in creates:
            self.assertIn(IMAGE, call)
            self.assertNotIn("lm-eval-fork:c1", call)
            self.assertEqual(call[call.index("--user") + 1], "65532:65532")
            self.assertIn("--read-only", call)
            self.assertIn("--init", call)
            self.assertEqual(call[call.index("--cap-drop") + 1], "ALL")
            self.assertEqual(
                call[call.index("--security-opt") + 1], "no-new-privileges"
            )
            self.assertEqual(call[call.index("--memory") + 1], "512m")
            self.assertEqual(call[call.index("--memory-swap") + 1], "512m")
            self.assertEqual(
                call[call.index("--tmpfs") + 1],
                # Independent oracle for Docker's isolated tmpfs mount.
                "/tmp:rw,nosuid,nodev,size=128m,mode=1777",  # noqa: S108
            )
            self.assertEqual(call.count("--mount"), 1)
            self.assertTrue(call[call.index("--mount") + 1].endswith(",readonly"))
        self.assertEqual(generation[generation.index("--network") + 1], "bridge")
        self.assertIn("OPENAI_API_KEY", generation)
        self.assertIn("HF_TOKEN", generation)
        self.assertIn("HF_ALLOW_CODE_EVAL=", generation)
        self.assertEqual(score[score.index("--network") + 1], "none")
        self.assertIn("HF_ALLOW_CODE_EVAL=1", score)
        self.assertNotIn("OPENAI_API_KEY", score)
        self.assertNotIn("HF_TOKEN", score)
        self.assertNotIn("private-openai", str(docker.calls))
        self.assertNotIn("private-hf", str(docker.calls))
        self.assertNotIn("docker.sock", str(docker.calls))
        manifest = json.loads((completed / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["image"]["id"], IMAGE)
        self.assertEqual(manifest["server"], "unknown")
        self.assertEqual(
            manifest["isolation"]["mounts"],
            [{"Type": "bind", "Destination": "/input/score-input.json", "RW": False}],
        )
        self.assertEqual(
            manifest["generation"], {"request_cache": False, "response_cache": None}
        )
        self.assertEqual((self.root / "run").stat().st_mode & 0o777, 0o700)
        self.assertEqual((completed / "scores.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            (self.root / "run" / ".partial" / "score-input.json").stat().st_mode
            & 0o777,
            0o444,
        )

    def test_failures_never_publish_and_cleanup_only_owned_ids(self):
        for failure in (
            "generate",
            "score",
            "copy",
            "timeout",
            "interrupt",
            "create",
            "cleanup",
        ):
            with self.subTest(failure=failure):
                self.args.output_dir = self.root / failure
                docker = FakeDocker(failure)
                with self.assertRaises(
                    (RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt)
                ):
                    launcher.run(self.args, docker)
                self.assertFalse(docker.alive)
                self.assertFalse((self.args.output_dir / "completed").exists())
                for call in docker.calls:
                    if call[0] == "rm":
                        self.assertIn(call[-1], ["1" * 64, "2" * 64])

    def test_publication_failure_leaves_no_completed_run(self):
        docker = FakeDocker()
        with (
            mock.patch.object(Path, "rename", side_effect=OSError("publish failure")),
            self.assertRaises(OSError),
        ):
            launcher.run(self.args, docker)
        self.assertFalse(docker.alive)
        self.assertFalse((self.args.output_dir / "completed").exists())

    def test_no_overwrite_existing_directory(self):
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / "keep.txt"
        sentinel.write_text("original")
        docker = FakeDocker()
        with self.assertRaises(FileExistsError):
            launcher.run(self.args, docker)
        self.assertEqual(sentinel.read_text(), "original")
        self.assertFalse(any(call[0] == "create" for call in docker.calls))

    def test_unsafe_inputs_fail_before_container_creation(self):
        original = json.loads(self.config.read_text())
        for url in (
            "http://user:pass@server/v1/chat/completions",
            "http://server/v1/chat/completions?key=secret",
            "file:///tmp/data",
        ):
            with self.subTest(url=url):
                self.config.write_text(json.dumps(dict(original, base_url=url)))
                docker = FakeDocker()
                with self.assertRaises(ValueError):
                    launcher.run(self.args, docker)
                self.assertFalse(docker.calls)
        self.config.write_text(json.dumps(original))
        for image in ("--privileged", "image,foo", "image\nother"):
            with self.subTest(image=image):
                self.args.image = image
                with self.assertRaises(ValueError):
                    launcher.run(self.args, FakeDocker())

    def test_credential_fields_rejected_before_copy(self):
        for field in ("headers", "api_key", "password", "HF_TOKEN"):
            config = json.loads(self.config.read_text())
            config["gen_kwargs"] = {field: "secret"}
            self.config.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                launcher.run(self.args, FakeDocker())
            self.assertFalse(self.args.output_dir.exists())

    def test_path_symlink_and_comma_rejected(self):
        for name in ("unsafe,field", "symlink"):
            path = self.root / name
            if name == "symlink":
                path.symlink_to(self.root / "missing")
            with self.assertRaises(ValueError):
                launcher.safe_path(path)

    def test_malformed_result_identity_never_publishes(self):
        docker = FakeDocker()
        docker.result["bundle_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            launcher.run(self.args, docker)
        self.assertFalse(docker.alive)
        self.assertFalse((self.args.output_dir / "completed").exists())

    def test_unexpected_image_environment_is_rejected(self):
        docker = FakeDocker()
        with (
            mock.patch.object(
                docker,
                "call",
                return_value=json.dumps(
                    [
                        {
                            "Id": IMAGE,
                            "Config": {
                                "User": "65532:65532",
                                "Env": ["OPENAI_API_KEY=baked-secret"],
                            },
                        }
                    ]
                ),
            ),
            self.assertRaises(ValueError),
        ):
            launcher.run(self.args, docker)
        self.assertFalse(self.args.output_dir.exists())

    def test_resource_bounds(self):
        for key, value in (
            ("memory_mib", 0),
            ("tmpfs_mib", 512),
            ("cpus", float("nan")),
            ("generation_timeout", 0),
            ("pids_limit", 10000),
        ):
            with (
                self.subTest(key=key),
                mock.patch.object(self.args, key, value),
                self.assertRaises(ValueError),
            ):
                launcher.run(self.args, FakeDocker())

    def test_python39_syntax_and_standard_library_only(self):
        tree = ast.parse(SCRIPT.read_text(), feature_version=(3, 9))
        imports = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertLessEqual(imports, {"__future__", "pathlib", "urllib"})

    def test_runner_bounded_log_and_environment_secret_redaction(self):
        process = mock.Mock()
        process.stdout = io.BytesIO(
            b"private-openai\n" + b"x" * (launcher.LOG_LIMIT * 2)
        )
        process.returncode = 0
        process.poll.return_value = 0
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "private-openai"}),
            mock.patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                launcher.subprocess, "Popen", return_value=process
            ) as popen,
        ):
            docker = launcher.Docker()
            output = docker.call(["image", "inspect", "image"], log=self.root / "log")
        self.assertLessEqual(len(output), launcher.LOG_LIMIT)
        self.assertNotIn("private-openai", output)
        self.assertIn("[REDACTED]", output)
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(
            popen.call_args.args[0], ["/usr/bin/docker", "image", "inspect", "image"]
        )
        self.assertEqual((self.root / "log").stat().st_mode & 0o777, 0o600)
        self.assertLessEqual((self.root / "log").stat().st_size, launcher.LOG_LIMIT)

    def test_actual_isolation_mismatch_fails_before_scoring(self):
        for field, value in (
            ("NetworkMode", "bridge"),
            ("ReadonlyRootfs", False),
            ("CapDrop", []),
            ("Init", False),
            ("MemorySwap", -1),
        ):
            with self.subTest(field=field):
                self.args.output_dir = self.root / field
                docker = FakeDocker()
                original = docker.call

                def altered(
                    argv, original=original, field=field, value=value, **kwargs
                ):
                    result = original(argv, **kwargs)
                    if argv[0] == "inspect":
                        record = json.loads(result)
                        record[0]["HostConfig"][field] = value
                        return json.dumps(record)
                    return result

                with (
                    mock.patch.object(docker, "call", side_effect=altered),
                    self.assertRaises(ValueError),
                ):
                    launcher.run(self.args, docker)
                self.assertFalse(docker.alive)
                self.assertFalse(
                    any(call[0] == "exec" and "score" in call for call in docker.calls)
                )
                self.assertFalse((self.args.output_dir / "completed").exists())

    def test_missing_or_weakened_actual_tmpfs_never_scores(self):
        scratch = str(Path(EXPECTED_SCORE_PATH).parent)
        bad_tmpfs = (
            None,
            {},
            {scratch: "rw,size=1024m,mode=1777"},
            {scratch: "rw,nosuid,nodev,size=1024m,mode=1777"},
            {scratch: "rw,nodev,size=128m,mode=1777"},
            {scratch: "rw,nosuid,size=128m,mode=1777"},
            {scratch: "rw,nosuid,nodev,size=128m,mode=0777"},
            {scratch: "rw,nosuid,nodev,size=128m,mode=1777", "/extra": "rw"},
        )
        for index, options in enumerate(bad_tmpfs):
            with self.subTest(options=options):
                self.args.output_dir = self.root / f"tmpfs-{index}"
                docker = FakeDocker()
                original = docker.call

                def altered(argv, original=original, options=options, **kwargs):
                    result = original(argv, **kwargs)
                    if argv[0] == "inspect":
                        records = json.loads(result)
                        records[0]["HostConfig"]["Tmpfs"] = options
                        return json.dumps(records)
                    return result

                with (
                    mock.patch.object(docker, "call", side_effect=altered),
                    self.assertRaises(ValueError),
                ):
                    launcher.run(self.args, docker)
                self.assertFalse(docker.alive)
                self.assertFalse(
                    any(call[0] == "exec" and "score" in call for call in docker.calls)
                )
                self.assertFalse((self.args.output_dir / "completed").exists())

    def test_log_bound_applies_after_redaction_expansion(self):
        process = mock.Mock()
        process.stdout = io.BytesIO(b"x" * launcher.LOG_LIMIT)
        process.poll.return_value = 0
        process.returncode = 0
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}),
            mock.patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(launcher.subprocess, "Popen", return_value=process),
        ):
            launcher.Docker().call(
                ["exec", "container", "python"], log=self.root / "expanded.log"
            )
        self.assertLessEqual(
            (self.root / "expanded.log").stat().st_size, launcher.LOG_LIMIT
        )
        self.assertNotIn("x", (self.root / "expanded.log").read_text())

    def test_copied_symlink_does_not_touch_host_target(self):
        target = self.root / "untouched"
        target.write_text("keep")
        target.chmod(0o644)
        docker = FakeDocker()
        original = docker.call

        def altered(argv, **kwargs):
            if kwargs.get("binary_output") is not None and argv[-2].endswith(
                "scores.json"
            ):
                Path(kwargs["binary_output"]).symlink_to(target)
                return ""
            return original(argv, **kwargs)

        with (
            mock.patch.object(docker, "call", side_effect=altered),
            self.assertRaises(ValueError),
        ):
            launcher.run(self.args, docker)
        self.assertEqual(target.read_text(), "keep")
        self.assertEqual(target.stat().st_mode & 0o777, 0o644)
        self.assertFalse(docker.alive)
        self.assertFalse((self.args.output_dir / "completed").exists())

    def test_runner_timeout_kills_client_and_keeps_bounded_diagnostic(self):
        process = mock.Mock()
        process.stdout = io.BytesIO(b"partial output")
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("docker", 1), 0]
        with (
            mock.patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(launcher.subprocess, "Popen", return_value=process),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            launcher.Docker().call(
                ["exec", "container", "python"],
                timeout=1,
                log=self.root / "timeout.log",
            )
        process.kill.assert_called_once()
        self.assertEqual((self.root / "timeout.log").read_text(), "partial output")

    def test_no_docker_operations_after_atomic_publication(self):
        docker = FakeDocker()
        original = Path.rename

        def publish(path, destination):
            self.assertFalse(docker.alive)
            docker.call = mock.Mock(
                side_effect=AssertionError("post-publication Docker operation")
            )
            return original(path, destination)

        with mock.patch.object(Path, "rename", publish):
            completed = launcher.run(self.args, docker)
        self.assertTrue((completed / "manifest.json").is_file())
        docker.call.assert_not_called()

    def test_binary_transfer_is_exact_and_new_only(self):
        payload = bytes(range(256)) * 400
        process = mock.Mock()
        process.stdout = io.BytesIO(payload)
        process.poll.return_value = 0
        process.returncode = 0
        destination = self.root / "artifact"
        with (
            mock.patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(launcher.subprocess, "Popen", return_value=process),
        ):
            docker = launcher.Docker()
            docker.call(
                ["exec", "container", "python"],
                binary_output=destination,
                max_bytes=len(payload),
            )
            with self.assertRaises(FileExistsError):
                docker.call(
                    ["exec", "container", "python"],
                    binary_output=destination,
                    max_bytes=len(payload),
                )
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_transfer_limit_failure_and_timeout_remove_incomplete_file(self):
        for failure in ("limit", "exit", "timeout"):
            with self.subTest(failure=failure):
                process = mock.Mock()
                process.stdout = io.BytesIO(b"incomplete payload")
                process.poll.return_value = 0
                process.returncode = 1 if failure == "exit" else 0
                if failure == "timeout":
                    process.wait.side_effect = [
                        subprocess.TimeoutExpired("docker", 1),
                        0,
                    ]
                    process.poll.return_value = None
                destination = self.root / failure
                with (
                    mock.patch.object(
                        launcher.shutil, "which", return_value="/usr/bin/docker"
                    ),
                    mock.patch.object(
                        launcher.subprocess, "Popen", return_value=process
                    ),
                    self.assertRaises(
                        (RuntimeError, ValueError, subprocess.TimeoutExpired)
                    ),
                ):
                    launcher.Docker().call(
                        ["exec", "container", "python"],
                        binary_output=destination,
                        max_bytes=4 if failure == "limit" else 100,
                    )
                self.assertFalse(destination.exists())


def verify_real_transfers(image):
    """Optional explicit Docker acceptance: real live tmpfs, not a fake oracle."""
    docker = launcher.Docker()
    image_id = launcher.image_identity(docker, image)["id"]
    container = docker.call(
        [
            "create",
            "--user",
            "65532:65532",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--pids-limit",
            "32",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--tmpfs",
            f"{launcher.CONTAINER_TMP}:rw,nosuid,nodev,size=16m,mode=1777",
            image_id,
            "python",
            "-c",
            launcher.KEEPALIVE,
        ]
    ).strip()
    try:
        docker.call(["start", container])
        directory = launcher.CONTAINER_TMP
        docker.call(
            [
                "exec",
                container,
                "python",
                "-c",
                (
                    "import os,pathlib,sys; p=pathlib.Path(sys.argv[1]);"
                    "(p/'regular').write_bytes(bytes(range(256)));"
                    "(p/'link').symlink_to(p/'regular');os.mkfifo(p/'fifo')"
                ),
                directory,
            ]
        )
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch) / "artifact"
            docker.call(
                [
                    "exec",
                    container,
                    "python",
                    "-c",
                    launcher.STREAM_ARTIFACT,
                    f"{directory}/regular",
                    "256",
                ],
                binary_output=destination,
                max_bytes=256,
            )
            assert destination.read_bytes() == bytes(range(256))
            for source, cap in (("link", 256), ("fifo", 256), ("regular", 4)):
                target = Path(scratch) / f"reject-{source}"
                try:
                    docker.call(
                        [
                            "exec",
                            container,
                            "python",
                            "-c",
                            launcher.STREAM_ARTIFACT,
                            f"{directory}/{source}",
                            str(cap),
                        ],
                        binary_output=target,
                        max_bytes=cap,
                        timeout=10,
                    )
                except (RuntimeError, ValueError):
                    assert not target.exists()
                else:
                    raise AssertionError(f"Unsafe source accepted: {source}")
        print(
            "Real transfer acceptance: exact binary bytes, symlink rejection, FIFO rejection, cap rejection passed"
        )
    finally:
        docker.call(["rm", "--force", container])


if __name__ == "__main__":
    unittest.main()

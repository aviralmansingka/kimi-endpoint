"""Local regressions; run with the Python environment that has Modal installed."""

import contextlib
import io
import subprocess
from unittest.mock import Mock, patch

import requests

import serve


def test_argv():
    cmd = serve.build_server_cmd(8123)
    assert "" not in cmd and "None" not in cmd
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert serve.GPU == "B300:8"
    for flag, value in {
        "--tp": "8",
        "--dcp-size": "8",
        "--max-mamba-cache-size": "160",
        "--chunked-prefill-size": "16384",
        "--cuda-graph-max-bs": "32",
    }.items():
        assert cmd.count(flag) == 1
        assert cmd[cmd.index(flag) + 1] == value
    for flag in ("--trust-remote-code", "--enable-linear-replayssm-spec"):
        assert cmd.count(flag) == 1
        i = cmd.index(flag)
        assert i == len(cmd) - 1 or cmd[i + 1].startswith("--")
    assert "--max-running-requests" not in cmd
    for flags in (serve.SERVER_ARGS, serve.speculative_config):
        for key, value in flags.items():
            flag = "--" + key.removeprefix("--")
            assert cmd.count(flag) == 1
            i = cmd.index(flag)
            if value is None:
                assert i == len(cmd) - 1 or cmd[i + 1].startswith("--")
            else:
                assert cmd[i + 1] == str(value)


def test_readiness():
    process = Mock(args=["python"], poll=Mock(return_value=17))
    with patch("serve.time.sleep") as sleep, patch("requests.get") as get:
        try:
            serve.wait_ready(process)
        except subprocess.CalledProcessError as exc:
            assert exc.returncode == 17
        else:
            raise AssertionError("Dead child did not fail immediately")
        sleep.assert_not_called()
        get.assert_not_called()

    process.poll.return_value = None
    with patch("serve.time.sleep") as sleep, patch(
        "requests.get", side_effect=[requests.exceptions.ConnectionError(), Mock()]
    ):
        serve.wait_ready(process)
        sleep.assert_called_once_with(5)


def test_verdicts():
    for returncode, output, expected in (
        (2, "", "FAIL"),
        (1, "usage: launch_server\nerror: unrecognized arguments: bad", "FAIL"),
        (1, "error: argument --port: invalid int value", "FAIL"),
        (1, "ImportError: torch", "INCONCLUSIVE"),
        (-11, "", "INCONCLUSIVE"),
        (1, "server_args=ServerArgs(tp_size=8)\nCUDA unavailable", "PASS"),
        (None, "server_args=ServerArgs(tp_size=8)", "INCONCLUSIVE"),
    ):
        def run(cmd, **kwargs):
            assert cmd == serve.build_server_cmd(serve.PORT)
            assert kwargs["timeout"] == 180
            kwargs["stdout"].write(output)
            if returncode is None:
                raise subprocess.TimeoutExpired(cmd, 180)
            return subprocess.CompletedProcess(cmd, returncode)

        with patch("serve.subprocess.run", side_effect=run):
            assert serve.check_argv.local()["verdict"] == expected


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        test_argv()
        test_readiness()
        test_verdicts()
    print("PASS: argv shape, fail-fast stderr tail, connection retries, CPU verdicts")

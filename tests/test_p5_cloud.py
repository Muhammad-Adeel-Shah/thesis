"""P5 AWS auto-shutdown and checklists, tested with a recorder in place of `shutdown`
(user rules, 2026-11-21: trap on exit and on error, idle watchdog, hard cap at start,
one instance, data volume DeleteOnTermination=false, projection within the stage's share)."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PY = Path(sys.executable).as_posix()


def recorder(tmp_path: Path) -> tuple[str, Path]:
    """A stand-in for `sudo shutdown -h`: appends its arguments to a file."""
    calls = tmp_path / "shutdown_calls.txt"
    stub = tmp_path / "stub_shutdown.py"
    stub.write_text("import sys\nopen(sys.argv[1], 'a').write(' '.join(sys.argv[2:]) + '\\n')\n",
                    encoding="utf-8")
    return f"{PY} {stub.as_posix()} {calls.as_posix()}", calls


# ------------------------------------------------------------------- the guard ---

def guard(tmp_path, stop_cmd, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p5_cloud_guard.py"), "--state",
         str(tmp_path / "state"), "--rate", "1.21", "--volume-created", dt.date.today().isoformat(),
         "--ebs-usd-per-day", "0.5333", "--poll", "0.05", "--grace", "0",
         "--stop-command", f"{stop_cmd} now", *extra],   # as the wrapper passes it
        capture_output=True, text=True, timeout=120, check=False)


def test_spend_counts_instance_hours_and_volume_days():
    import p5_cloud_guard as G

    today = dt.date(2026, 11, 25)
    assert G.spend_usd(10, 1.21, "2026-11-22", 0.5333, today) == pytest.approx(12.1 + 4 * 0.5333)


def test_guard_stops_at_the_cap(tmp_path):
    stop_cmd, calls = recorder(tmp_path)
    done = guard(tmp_path, stop_cmd, "--cap-usd", "0.1", "--max-polls", "5")
    assert done.returncode == 3, done.stdout + done.stderr
    assert calls.read_text().split() == ["now"]
    assert "SPEND CAP REACHED" in (tmp_path / "state" / "ALERT_CAP.txt").read_text()


def test_guard_stops_when_idle(tmp_path):
    stop_cmd, calls = recorder(tmp_path)
    done = guard(tmp_path, stop_cmd, "--cap-usd", "75", "--idle-minutes", "0.002",
                 "--max-polls", "200")
    assert done.returncode == 0 and calls.read_text().split() == ["now"], done.stdout
    ledger = json.loads((tmp_path / "state" / "ledger.json").read_text())
    assert "idle" in ledger["sessions"][-1]["stopped"]


def test_guard_stops_when_the_watched_job_ends(tmp_path):
    stop_cmd, calls = recorder(tmp_path)
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    done = guard(tmp_path, stop_cmd, "--cap-usd", "75", "--idle-minutes", "60",
                 "--watch-pid", str(job.pid), "--max-polls", "200")
    job.wait()
    assert done.returncode == 0 and calls.read_text().split() == ["now"], done.stdout


# --------------------------------------------------------- the session wrapper ---

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def session(tmp_path, command: str, estimate: str = "0.01"):
    stop_cmd, calls = recorder(tmp_path)
    env = dict(os.environ, P5_STATE=(tmp_path / "session").as_posix(), P5_SHUTDOWN=stop_cmd,
               P5_PYTHON=PY, P5_VOLUME_CREATED=dt.date.today().isoformat(), P5_POLL="0.2")
    done = subprocess.run(["bash", (ROOT / "env" / "aws_p5_session.sh").as_posix(), "--label",
                           "t", "--estimate-hours", estimate, "--", "bash", "-c", command],
                          capture_output=True, text=True, env=env, timeout=120, check=False)
    records = [json.loads(line) for line in
               (tmp_path / "session" / "sessions.jsonl").read_text().splitlines()]
    return done, calls.read_text().split("\n"), records


@needs_bash
def test_session_schedules_the_cap_first_and_stops_on_success(tmp_path):
    done, calls, records = session(tmp_path, "echo working; sleep 0.5")
    assert done.returncode == 0, done.stdout + done.stderr
    assert calls[0] == "+1"            # ceil(0.01 h x 1.5 x 60) = 1 min, scheduled first
    assert "now" in calls[1:]          # the exit trap stopped the instance
    assert records[-1]["status"] == "ok" and records[-1]["cap_min"] == 1


@needs_bash
def test_session_backs_up_before_it_stops(tmp_path, monkeypatch):
    marker = (tmp_path / "backup_ran.txt").as_posix()
    monkeypatch.setenv("P5_BACKUP_CMD", f"echo backed-up > {marker}")
    done, calls, records = session(tmp_path, "echo working")
    assert done.returncode == 0 and Path(marker).read_text().strip() == "backed-up"
    assert "now" in calls[1:]


@needs_bash
def test_session_stops_on_error_too(tmp_path):
    done, calls, records = session(tmp_path, "echo failing; exit 7", estimate="2")
    assert done.returncode != 0
    assert calls[0] == "+180" and "now" in calls[1:]
    assert records[-1]["status"] == "failed" and records[-1]["exit"] != 0


# ----------------------------------------------------------- the PC checklists ---

def fake_aws(responses):
    def call(*args):
        return responses[args[0]](args)
    return call


SSH_ONE_32 = [{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": "198.51.100.7/32"}], "Ipv6Ranges": []}]


def base_responses(delete_on_termination=False, behaviour="stop", others=(), zone="us-east-2a",
                   iops=3000, root_gb=75, lifecycle=None, ingress=None, groups=1,
                   instance_sgs=("sg-1",), root_type="gp3", root_size=None, root_dot=True,
                   encryption_default=False, data_encrypted=False, root_encrypted=False,
                   ami_encrypted=False):
    volumes = {
        "vol-data": {"VolumeId": "vol-data", "Size": 200, "VolumeType": "gp3", "State": "in-use",
                     "AvailabilityZone": zone, "Iops": iops, "Throughput": 125,
                     "Encrypted": data_encrypted},
        "vol-root": {"VolumeId": "vol-root", "Size": root_size or root_gb,
                     "VolumeType": root_type, "State": "in-use", "AvailabilityZone": zone,
                     "Iops": 3000, "Throughput": 125, "Encrypted": root_encrypted},
    }

    def instances(args):
        if "--instance-ids" in args:
            blocks = [{"DeviceName": "/dev/sda1",
                       "Ebs": {"VolumeId": "vol-root", "DeleteOnTermination": root_dot}},
                      {"DeviceName": "/dev/sdf",
                       "Ebs": {"VolumeId": "vol-data",
                               "DeleteOnTermination": delete_on_termination}}]
            inst = {"InstanceId": "i-1", "InstanceType": "g5.2xlarge",
                    "RootDeviceName": "/dev/sda1", "BlockDeviceMappings": blocks,
                    "SecurityGroups": [{"GroupId": g} for g in instance_sgs]}
            if lifecycle:
                inst["InstanceLifecycle"] = lifecycle
            return {"Reservations": [{"Instances": [inst]}]}
        return {"Reservations": [{"Instances": [{"InstanceId": i, "State": {"Name": "stopped"}}
                                                for i in ("i-1", *others)]}]}

    def describe_volumes(args):
        if "--volume-ids" in args:
            return {"Volumes": [volumes[args[args.index("--volume-ids") + 1]]]}
        return {"Volumes": list(volumes.values())}

    sg = {"GroupId": "sg-1", "GroupName": "dbp-p5",
          "IpPermissions": SSH_ONE_32 if ingress is None else ingress}
    return {"describe-instances": instances,
            "describe-security-groups": lambda a: {"SecurityGroups": [sg] * groups},
            "describe-images": lambda a: {"Images": [{
                "ImageId": "ami-x", "RootDeviceName": "/dev/sda1",
                "BlockDeviceMappings": [{"DeviceName": "/dev/sda1",
                                         "Ebs": {"VolumeSize": root_gb,
                                                 "Encrypted": ami_encrypted}}]}]},
            "get-ebs-encryption-by-default": lambda a: {
                "EbsEncryptionByDefault": encryption_default},
            "describe-instance-type-offerings": lambda a: {"InstanceTypeOfferings": [
                {"Location": "us-east-2a"}, {"Location": "us-east-2b"}]},
            "describe-instance-types": lambda a: {"InstanceTypes": [
                {"VCpuInfo": {"DefaultVCpus": 8}}]},
            "describe-volumes": describe_volumes,
            "describe-instance-attribute": lambda a: {
                "InstanceInitiatedShutdownBehavior": {"Value": behaviour}},
            "describe-addresses": lambda a: {"Addresses": []},
            "describe-snapshots": lambda a: {"Snapshots": []}}


@pytest.mark.parametrize("change, hours, ok", [
    ({}, 10.0, True),
    ({"delete_on_termination": True}, 10.0, False),
    ({"behaviour": "terminate"}, 10.0, False),
    ({"others": ("i-2",)}, 10.0, False),
    ({}, 37.0, False),                  # 37 h x $1.217 = $45.03 > the $45 share
    ({"quota": 0.0}, 10.0, False),      # a new account's G/VT quota may be 0
    ({"quota": None}, 10.0, False),     # the quota could not be found
    ({"zone": "us-east-2c"}, 10.0, False),   # g5.2xlarge not offered where the volume is
    ({"iops": 6000}, 10.0, False),      # provisioned IOPS are billed extra
    ({"root_gb": 120}, 10.0, True),     # the root follows the AMI's size, whatever it is
    ({"lifecycle": "spot"}, 10.0, False),
    # the SSH group (user, 2026-11-22): tcp 22 from exactly one /32, nothing else
    ({"ingress": [{**SSH_ONE_32[0], "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]}, 10.0, False),
    ({"ingress": [{**SSH_ONE_32[0], "Ipv6Ranges": [{"CidrIpv6": "::/0"}]}]}, 10.0, False),
    ({"ingress": [{**SSH_ONE_32[0], "IpRanges": [{"CidrIp": "198.51.100.0/24"}]}]}, 10.0, False),
    ({"ingress": [{**SSH_ONE_32[0], "IpRanges": [{"CidrIp": "198.51.100.7/32"},
                                                 {"CidrIp": "203.0.113.9/32"}]}]}, 10.0, False),
    ({"ingress": [{**SSH_ONE_32[0], "FromPort": 0, "ToPort": 65535}]}, 10.0, False),
    ({"ingress": [{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "198.51.100.7/32"}]}]}, 10.0,
     False),
    ({"ingress": []}, 10.0, False),     # no SSH rule at all
    ({"groups": 0}, 10.0, False),       # the group does not exist
    ({"instance_sgs": ("sg-1", "sg-default")}, 10.0, False),
    # the root volume (user, 2026-11-22): gp3 explicitly, at the AMI's size, deleted with it
    ({"root_type": "gp2"}, 10.0, False),
    ({"root_size": 100}, 10.0, False),
    ({"root_dot": False}, 10.0, False),
    # 2026-11-23: encryption by default ON made AWS delete the data volume right after
    # CreateVolume (kms:GenerateDataKeyWithoutPlaintext AccessDenied for dbp-launch)
    ({"encryption_default": True}, 10.0, False),
    ({"data_encrypted": True}, 10.0, False),
    ({"root_encrypted": True}, 10.0, False),
    ({"ami_encrypted": True}, 10.0, False),
])
def test_prelaunch_refuses_every_unsafe_condition(monkeypatch, capsys, change, hours, ok):
    import argparse

    import p5_aws_check as C

    change = dict(change)
    quota = change.pop("quota", 8.0)
    monkeypatch.setattr(C, "gvt_quota_vcpus", lambda: quota)
    monkeypatch.setattr(C, "aws", fake_aws(base_responses(**change)))
    args = argparse.Namespace(ami="ami-x", data_volume="vol-data", instance_id="i-1",
                              stage="reconstruction", hours=hours, security_group="dbp-p5")
    assert (C.prelaunch(args) == 0) is ok, capsys.readouterr().out


def test_prelaunch_never_prints_the_ssh_address(monkeypatch, capsys):
    import argparse

    import p5_aws_check as C

    monkeypatch.setattr(C, "gvt_quota_vcpus", lambda: 8.0)
    monkeypatch.setattr(C, "aws", fake_aws(base_responses()))
    args = argparse.Namespace(ami="ami-x", data_volume="vol-data", instance_id="i-1",
                              stage="setup_pexels", hours=2.0, security_group="dbp-p5")
    assert C.prelaunch(args) == 0
    out = capsys.readouterr().out
    assert "198.51.100.7" not in out and "tcp 22 from 1 IPv4 /32" in out
    assert "VolumeSize=75,VolumeType=gp3" in out and "$1.217/h" in out


def test_rate_includes_the_public_ipv4_address():
    import p5_aws_check as C

    assert pytest.approx(1.217) == C.RATE
    session = (ROOT / "env" / "aws_p5_session.sh").read_text(encoding="utf-8")
    assert '"${P5_RATE:-1.217}"' in session


def test_runbook_ssh_is_one_32_and_root_is_gp3():
    text = (ROOT / "env" / "cloud_p5_runbook.md").read_text(encoding="utf-8")
    rules = [line for line in text.splitlines() if "authorize-security-group-ingress" in line]
    assert rules, "no SSH rule in the runbook"
    for line in rules:
        assert "--port 22" in line and line.rstrip().endswith('/32"'), line
        assert "0.0.0.0/0" not in line and "::/0" not in line and "ipv6" not in line.lower()
    for start in [i for i in range(len(text)) if text.startswith("run-instances", i)]:
        cmd = text[start:start + 700]
        if "--image-id" in cmd[:80]:
            assert "VolumeType=gp3" in cmd and "VolumeSize=<ROOT_GB>" in cmd, cmd[:200]


def test_postsession_reports_clean_and_spend(monkeypatch, capsys, tmp_path):
    import argparse

    import p5_aws_check as C

    monkeypatch.setattr(C, "aws", fake_aws(base_responses()))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"hours": 5.0}))
    args = argparse.Namespace(ledger=str(ledger), volume_created=dt.date.today().isoformat(),
                              hours=None)
    assert C.postsession(args) == 0
    out = capsys.readouterr().out
    assert "running instances = 0" in out and "Elastic IPs = 0" in out and "$6.82" in out


def test_postsession_takes_hours_without_a_ledger(monkeypatch, capsys):
    import argparse

    import p5_aws_check as C

    monkeypatch.setattr(C, "aws", fake_aws(base_responses()))
    args = argparse.Namespace(ledger="", volume_created=dt.date.today().isoformat(), hours=5.0)
    assert C.postsession(args) == 0 and "$6.82" in capsys.readouterr().out


def test_runbook_gpu_jobs_name_the_device():
    """The reconstruction command once lacked --device cuda: p5_pilot.py defaults to CPU."""
    text = (ROOT / "env" / "cloud_p5_runbook.md").read_text(encoding="utf-8")
    for phase in ("smoke", "reconstruct"):
        start = text.index(f"--phase {phase} ")
        assert "--device cuda" in text[start:start + 300], phase


def test_runbook_reconstruction_skips_threshold_reals_and_uses_the_fp16_rule():
    """User, 2026-11-22: the 200 threshold-held-out reals are never reconstructed, and fp16
    follows the pilot's records."""
    text = (ROOT / "env" / "cloud_p5_runbook.md").read_text(encoding="utf-8")
    start = text.index("--phase reconstruct ")
    cmd = text[start:start + 500]
    assert "--splits train,validation" in cmd
    assert "--fp16-evidence reports/kaggle_20261121/p5_pilot" in cmd
    assert (ROOT / "reports" / "kaggle_20261121" / "p5_pilot" / "smoke_wan21.json").is_file()


def test_check_script_only_describes():
    import p5_aws_check as C

    with pytest.raises(SystemExit, match="only describes"):
        C.aws("run-instances")
    assert {"get-ebs-encryption-by-default"} == C.READ_ONLY_GETS
    for verb in ("enable-ebs-encryption-by-default", "disable-ebs-encryption-by-default"):
        with pytest.raises(SystemExit, match="only describes"):
            C.aws(verb)


# ------------------------------------------------------- IAM policies and the key helper ---

IAM = ROOT / "env" / "aws_iam"


def statements(name):
    return json.loads((IAM / name).read_text(encoding="utf-8"))["Statement"]


def as_list(value):
    return [value] if isinstance(value, str) else list(value)


def test_agent_policy_allows_only_stop_and_terminate_in_us_east_2():
    allows = [st for st in statements("dbp-agent-inline.json") if st["Effect"] == "Allow"]
    assert [sorted(as_list(st["Action"])) for st in allows] == [
        ["ec2:StopInstances", "ec2:TerminateInstances"]]
    assert allows[0]["Resource"].startswith("arn:aws:ec2:us-east-2:")
    denied = {a for st in statements("dbp-agent-inline.json") if st["Effect"] == "Deny"
              for a in as_list(st["Action"])}
    assert {"ec2:RunInstances", "ec2:StartInstances"} <= denied


def test_launch_policy_is_closed_and_g5_2xlarge_only():
    sts = statements("dbp-launch.json")
    allowed = {a for st in sts if st["Effect"] == "Allow" for a in as_list(st["Action"])}
    closed = next(st for st in sts if st["Sid"] == "DenyEverythingNotListed")
    assert allowed | {"sts:GetCallerIdentity"} == set(closed["NotAction"])   # no gap either way
    # user, 2026-11-22: revoking the old /32 when the public IP changes; it only removes access
    assert "ec2:RevokeSecurityGroupIngress" in allowed
    ssh = next(st for st in sts if st["Sid"] == "SshSecurityGroupAndKeyPair")
    assert "ec2:RevokeSecurityGroupIngress" in ssh["Action"]
    assert not any(a.startswith(("s3:", "iam:")) or a in ("ec2:AllocateAddress",
                   "ec2:CreateSnapshot", "ec2:CreateSnapshots", "ec2:ModifyInstanceAttribute")
                   for a in allowed)
    by_sid = {st["Sid"]: st for st in sts}
    assert by_sid["DenyOutsideUsEast2"]["Condition"] == {
        "StringNotEquals": {"aws:RequestedRegion": "us-east-2"}}
    assert by_sid["DenyAnyInstanceTypeButG52xlarge"]["Condition"] == {
        "StringNotEquals": {"ec2:InstanceType": "g5.2xlarge"}}
    for st in sts:
        if st["Effect"] == "Allow" and st["Resource"] != "*":
            assert all(r.startswith("arn:aws:ec2:us-east-2:") for r in as_list(st["Resource"]))


needs_powershell = pytest.mark.skipif(shutil.which("powershell") is None,
                                      reason="Windows PowerShell not on PATH")


@needs_powershell
def test_key_helper_parses_writes_no_file_and_clears():
    script = ROOT / "env" / "aws_p5_env.ps1"
    raw = script.read_bytes()
    assert max(raw) < 128                      # Windows PowerShell 5.1 reads BOM-less files as ANSI
    text = raw.decode("ascii")
    for forbidden in ("Out-File", "Set-Content", "Add-Content", "Export-Clixml",
                      "aws configure", "Start-Transcript", "Write-Host $env:AWS_SECRET"):
        assert forbidden not in text
    reads = [line for line in text.splitlines() if "= Read-Host" in line]
    assert reads and all("-AsSecureString" in line for line in reads), reads
    probe = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
             f"'{script}', [ref]$null, [ref]$e); 'errors=' + $e.Count; "
             "$env:AWS_ACCESS_KEY_ID='x'; $env:AWS_SECRET_ACCESS_KEY='y'; "
             f"& '{script}' -Clear | Out-Null; "
             "'left=' + [bool]($env:AWS_ACCESS_KEY_ID -or $env:AWS_SECRET_ACCESS_KEY)")
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                           probe], capture_output=True, text=True, timeout=120, check=False)
    assert "errors=0" in done.stdout and "left=False" in done.stdout, done.stdout + done.stderr

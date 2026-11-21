"""P5 AWS checklists, run on the user's PC with the AWS CLI (user rules, 2026-11-21).

``prelaunch``  -- before anything that STARTS spending (launch / start): region us-east-2,
                  type g5.2xlarge, the AMI id, the data volume (attached at launch,
                  DeleteOnTermination=false), shutdown behaviour = stop, no other P5
                  instance, and the projected hours and cost against the stage's budget.
                  Exit 1 if any check fails: do not launch.
``postsession`` -- after every session: running instances = 0, volumes = root + data only,
                  Elastic IPs = 0, snapshots = 0, and the spend so far (from the guard's
                  ledger, copied from the instance) against the budget.

Read-only: it calls only ``aws ec2 describe-*``. It never launches, starts or modifies.
Credentials come from the shell's environment variables; nothing is printed of them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REGION, TYPE, RATE = "us-east-2", "g5.2xlarge", 1.21
EBS_PER_GB_MONTH, DATA_GB = 0.08, 200          # user, 2026-11-21 (to be confirmed in console)
BUDGET = {"setup_pexels": 5, "reconstruction": 45, "training": 15, "features_heads": 3,
          "ebs": 5, "reserve": 27}
HARD_STOP = 75


def aws(*args: str) -> Any:
    """One read-only ``aws ec2 describe-*`` call, JSON out."""
    if not args or not args[0].startswith("describe-"):
        raise SystemExit("p5_aws_check only describes; it never changes anything")
    done = subprocess.run(["aws", "ec2", *args, "--region", REGION, "--output", "json"],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"aws ec2 {args[0]} failed: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout or "{}")


def p5_instances() -> list[dict[str, Any]]:
    out = aws("describe-instances", "--filters", "Name=tag:Name,Values=dbp-p5",
              "Name=instance-state-name,Values=pending,running,stopping,stopped")
    return [i for r in out.get("Reservations", []) for i in r["Instances"]]


def ebs_usd(days: float) -> float:
    return DATA_GB * EBS_PER_GB_MONTH / 30 * days


def prelaunch(args: argparse.Namespace) -> int:
    problems: list[str] = []
    others = [i["InstanceId"] for i in p5_instances() if i["InstanceId"] != args.instance_id]
    if others:
        problems.append(f"another P5 instance exists: {others} (one at a time)")
    image = aws("describe-images", "--image-ids", args.ami).get("Images", [])
    if not image:
        problems.append(f"AMI {args.ami} not found in {REGION}")
    vol = aws("describe-volumes", "--volume-ids", args.data_volume).get("Volumes", [])
    if not vol:
        problems.append(f"data volume {args.data_volume} not found")
    elif vol[0]["Size"] != DATA_GB or vol[0]["VolumeType"] != "gp3":
        problems.append(f"data volume is {vol[0]['Size']} GB {vol[0]['VolumeType']}, "
                        f"expected {DATA_GB} GB gp3")
    if args.instance_id:
        attr = aws("describe-instance-attribute", "--instance-id", args.instance_id,
                   "--attribute", "instanceInitiatedShutdownBehavior")
        if attr.get("InstanceInitiatedShutdownBehavior", {}).get("Value") != "stop":
            problems.append("shutdown behaviour is not 'stop'")
        mapping = aws("describe-instances", "--instance-ids", args.instance_id)
        inst = mapping["Reservations"][0]["Instances"][0]
        if inst["InstanceType"] != TYPE:
            problems.append(f"instance type {inst['InstanceType']}, expected {TYPE}")
        data = [b for b in inst.get("BlockDeviceMappings", [])
                if b["Ebs"]["VolumeId"] == args.data_volume]
        if not data:
            problems.append("the data volume is not attached to this instance")
        elif data[0]["Ebs"]["DeleteOnTermination"]:
            problems.append("the data volume has DeleteOnTermination=true (must be false)")
    cost = args.hours * RATE
    share = BUDGET.get(args.stage)
    print(f"PRE-LAUNCH CHECKLIST ({dt.datetime.now():%Y-%m-%d %H:%M})")
    print(f"  region {REGION} | type {TYPE} | AMI {args.ami} | data volume {args.data_volume}")
    print("  shutdown behaviour = stop | watchdog + hard cap: armed by env/aws_p5_session.sh")
    print(f"  stage {args.stage}: projected {args.hours:.2f} h x ${RATE} = ${cost:.2f} "
          f"(share ${share}); hard cap at launch: {args.hours * 1.5:.2f} h")
    if share is not None and cost > share:
        problems.append(f"projection ${cost:.2f} exceeds the {args.stage} share ${share}: ask")
    for p in problems:
        print("  FAIL:", p)
    print("  RESULT:", "OK to launch AFTER the user's explicit OK" if not problems else
          "DO NOT LAUNCH")
    return 0 if not problems else 1


def postsession(args: argparse.Namespace) -> int:
    running = [i["InstanceId"] for i in p5_instances() if i["State"]["Name"] in
               ("pending", "running", "stopping")]
    vols = aws("describe-volumes").get("Volumes", [])
    eips = aws("describe-addresses").get("Addresses", [])
    snaps = aws("describe-snapshots", "--owner-ids", "self").get("Snapshots", [])
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8")) if args.ledger else {}
    hours = float(ledger.get("hours", 0.0))
    days = (dt.date.today() - dt.date.fromisoformat(args.volume_created)).days + 1
    spend = hours * RATE + ebs_usd(days)
    print(f"POST-SESSION ({dt.datetime.now():%Y-%m-%d %H:%M})")
    print(f"  running instances = {len(running)} {running}")
    print(f"  volumes = {len(vols)}: " + ", ".join(
        f"{v['VolumeId']} {v['Size']} GB {v['State']}" for v in vols))
    print(f"  Elastic IPs = {len(eips)} | snapshots = {len(snaps)}")
    print(f"  spend so far: {hours:.2f} instance-h x ${RATE} + {days} d EBS "
          f"(${ebs_usd(days):.2f}) = ${spend:.2f} of hard stop ${HARD_STOP} "
          f"(budget shares {BUDGET})")
    bad = running or len(vols) > 2 or eips or snaps or spend >= HARD_STOP
    print("  RESULT:", "CLEAN" if not bad else "ACTION NEEDED")
    return 0 if not bad else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    pre = sub.add_parser("prelaunch")
    pre.add_argument("--ami", required=True)
    pre.add_argument("--data-volume", required=True)
    pre.add_argument("--instance-id", default="", help="for a START of a stopped instance")
    pre.add_argument("--stage", required=True, choices=sorted(BUDGET))
    pre.add_argument("--hours", type=float, required=True)
    post = sub.add_parser("postsession")
    post.add_argument("--ledger", default="", help="ledger.json copied from the instance")
    post.add_argument("--volume-created", required=True)
    args = parser.parse_args()
    return prelaunch(args) if args.cmd == "prelaunch" else postsession(args)


if __name__ == "__main__":
    sys.exit(main())

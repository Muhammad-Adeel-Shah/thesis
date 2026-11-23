"""P5 AWS checklists, run on the user's PC with the AWS CLI (user rules, 2026-11-21).

``prelaunch``  -- before anything that STARTS spending (launch / start): region us-east-2,
                  type g5.2xlarge, the AMI id and its root size, the data volume (attached
                  at launch, DeleteOnTermination=false), the SSH security group (tcp 22 from
                  one /32 only; never 0.0.0.0/0 or ::/0), shutdown behaviour = stop, the
                  root volume gp3 at the AMI's size, EBS encryption by default OFF and no
                  encrypted volume or AMI snapshot (dbp-launch has no KMS permission: an
                  encrypted volume is deleted right after it is created; 2026-11-23), no other
                  P5 instance, and the projected hours and cost against the stage's budget.
                  Exit 1 if any check fails: do not launch.
``postsession`` -- after every session: running instances = 0, volumes = root + data only,
                  Elastic IPs = 0, snapshots = 0, and the spend so far (from the guard's
                  ledger, copied from the instance) against the budget.

Read-only: it calls only ``aws ec2 describe-*``, ``aws ec2 get-ebs-encryption-by-default``
and ``aws service-quotas list-service-quotas``. It never launches, starts or modifies.
Credentials come from the shell's environment variables; nothing is printed of them, and
the SSH rule's address is not printed either (only its prefix length).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# $1.212/h on-demand g5.2xlarge + $0.005/h for the public IPv4 address (user, 2026-11-22)
INSTANCE_RATE, IPV4_RATE = 1.212, 0.005
REGION, TYPE, RATE = "us-east-2", "g5.2xlarge", round(INSTANCE_RATE + IPV4_RATE, 3)
EBS_PER_GB_MONTH, DATA_GB = 0.08, 200          # user, 2026-11-21 (to be confirmed in console)
SG_NAME = "dbp-p5"                              # the runbook's SSH security group
# gp3's included performance (AWS pricing page; NOT-VERIFIED by a command): anything above it
# is billed on top of the GB-month, so a provisioned extra is refused
GP3_INCLUDED_IOPS, GP3_INCLUDED_MIBS = 3000, 125
GVT_QUOTA_NAME = "On-Demand G and VT"          # matched inside the quota's name
BUDGET = {"setup_pexels": 5, "reconstruction": 45, "training": 15, "features_heads": 3,
          "ebs": 5, "reserve": 27}
HARD_STOP = 75


#: The read-only ``aws ec2`` calls that are not ``describe-*``.
READ_ONLY_GETS = frozenset({"get-ebs-encryption-by-default"})


def aws(*args: str) -> Any:
    """One read-only ``aws ec2 describe-*`` (or :data:`READ_ONLY_GETS`) call, JSON out."""
    if not args or not (args[0].startswith("describe-") or args[0] in READ_ONLY_GETS):
        raise SystemExit("p5_aws_check only describes; it never changes anything")
    done = subprocess.run(["aws", "ec2", *args, "--region", REGION, "--output", "json"],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"aws ec2 {args[0]} failed: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout or "{}")


def gvt_quota_vcpus() -> float | None:
    """The applied 'Running On-Demand G and VT instances' vCPU quota, or None if not listed."""
    done = subprocess.run(["aws", "service-quotas", "list-service-quotas", "--service-code",
                           "ec2", "--region", REGION, "--output", "json"],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"aws service-quotas list-service-quotas failed: "
                         f"{done.stderr.strip()[:300]}")
    for quota in json.loads(done.stdout or "{}").get("Quotas", []):
        if GVT_QUOTA_NAME in quota.get("QuotaName", ""):
            return float(quota["Value"])
    return None


def p5_instances() -> list[dict[str, Any]]:
    out = aws("describe-instances", "--filters", "Name=tag:Name,Values=dbp-p5",
              "Name=instance-state-name,Values=pending,running,stopping,stopped")
    return [i for r in out.get("Reservations", []) for i in r["Instances"]]


def ebs_usd(days: float, gb: float = DATA_GB) -> float:
    """EBS gp3 cost of ``gb`` GB for ``days`` days."""
    return gb * EBS_PER_GB_MONTH / 30 * days


def gp3_extra_problem(what: str, v: dict[str, Any]) -> list[str]:
    """A refusal if the volume provisions IOPS or throughput above gp3's included values."""
    if v.get("Iops", 0) > GP3_INCLUDED_IOPS or v.get("Throughput", 0) > GP3_INCLUDED_MIBS:
        return [f"{what} has {v.get('Iops')} IOPS / {v.get('Throughput')} MiB/s, above gp3's "
                f"included {GP3_INCLUDED_IOPS} / {GP3_INCLUDED_MIBS}"]
    return []


def security_group_problems(name: str) -> tuple[list[str], str, str | None]:
    """Check the P5 SSH group: ingress tcp 22 from exactly one IPv4 /32, nothing else.

    Returns (problems, a summary without the address, the group id or None).
    """
    groups = aws("describe-security-groups", "--filters",
                 f"Name=group-name,Values={name}").get("SecurityGroups", [])
    if len(groups) != 1:
        return [f"security group {name}: {len(groups)} found, expected 1"], "missing", None
    problems: list[str] = []
    addresses = 0
    for perm in groups[0].get("IpPermissions", []):
        proto, lo, hi = perm.get("IpProtocol"), perm.get("FromPort"), perm.get("ToPort")
        if (proto, lo, hi) != ("tcp", 22, 22):
            problems.append(f"{name} allows {proto} {lo}-{hi}: only tcp 22 is allowed")
        for r in perm.get("IpRanges", []):
            cidr = r.get("CidrIp", "")
            if cidr == "0.0.0.0/0":
                problems.append(f"{name} allows 0.0.0.0/0 (the whole internet)")
            elif not cidr.endswith("/32"):
                problems.append(f"{name} allows a /{cidr.rsplit('/', 1)[-1]} range: only one "
                                f"/32 (your current public IP)")
            else:
                addresses += 1
        for r in perm.get("Ipv6Ranges", []):
            cidr = r.get("CidrIpv6", "")
            problems.append(f"{name} allows IPv6 " + ("::/0 (the whole internet)"
                                                      if cidr == "::/0" else "ranges"))
        if perm.get("UserIdGroupPairs") or perm.get("PrefixListIds"):
            problems.append(f"{name} allows another security group or a prefix list")
    if addresses == 0:
        problems.append(f"{name} has no tcp 22 rule from a /32: SSH could not connect")
    elif addresses > 1:
        problems.append(f"{name} allows {addresses} /32 addresses: revoke all but your "
                        f"current public IP")
    return problems, f"tcp 22 from {addresses} IPv4 /32, no IPv6", groups[0]["GroupId"]


def prelaunch(args: argparse.Namespace) -> int:
    problems: list[str] = []
    others = [i["InstanceId"] for i in p5_instances() if i["InstanceId"] != args.instance_id]
    if others:
        problems.append(f"another P5 instance exists: {others} (one at a time)")
    image = aws("describe-images", "--image-ids", args.ami).get("Images", [])
    root_device, root_gb = "?", None
    if not image:
        problems.append(f"AMI {args.ami} not found in {REGION}")
    else:
        root_device = image[0].get("RootDeviceName", "?")
        root_gb = next((b["Ebs"].get("VolumeSize") for b in image[0].get("BlockDeviceMappings", [])
                        if b.get("DeviceName") == root_device and "Ebs" in b), None)
        if root_gb is None:
            problems.append(f"the AMI's root mapping {root_device} has no snapshot size")
        if any(b.get("Ebs", {}).get("Encrypted") for b in image[0].get("BlockDeviceMappings", [])):
            problems.append("the AMI has an encrypted snapshot: its volumes would need KMS, "
                            "which dbp-launch does not have")
    encryption_default = aws("get-ebs-encryption-by-default").get("EbsEncryptionByDefault")
    if encryption_default is not False:
        problems.append(f"EBS encryption by default is {encryption_default}: dbp-launch has no "
                        f"KMS permission, so every new volume (the root too) is deleted right "
                        f"after it is created (2026-11-23). Turn it off first")
    offered = sorted(o["Location"] for o in aws(
        "describe-instance-type-offerings", "--location-type", "availability-zone",
        "--filters", f"Name=instance-type,Values={TYPE}").get("InstanceTypeOfferings", []))
    vol = aws("describe-volumes", "--volume-ids", args.data_volume).get("Volumes", [])
    if not vol:
        problems.append(f"data volume {args.data_volume} not found")
    else:
        v = vol[0]
        if v["Size"] != DATA_GB or v["VolumeType"] != "gp3":
            problems.append(f"data volume is {v['Size']} GB {v['VolumeType']}, "
                            f"expected {DATA_GB} GB gp3")
        if v.get("AvailabilityZone") not in offered:
            problems.append(f"data volume is in {v.get('AvailabilityZone')}; {TYPE} is offered "
                            f"in {offered}")
        problems += gp3_extra_problem("data volume", v)
        if v.get("Encrypted") is not False:
            problems.append(f"data volume Encrypted={v.get('Encrypted')}, expected false")
    sg_problems, sg_summary, sg_id = security_group_problems(args.security_group)
    problems += sg_problems
    vcpus = aws("describe-instance-types", "--instance-types", TYPE)["InstanceTypes"][0][
        "VCpuInfo"]["DefaultVCpus"]
    quota = gvt_quota_vcpus()
    if quota is None:
        problems.append(f"no '{GVT_QUOTA_NAME}' vCPU quota listed: check Service Quotas")
    elif quota < vcpus:
        problems.append(f"the on-demand G/VT vCPU quota is {quota:g} < the {vcpus} one {TYPE} "
                        f"needs: request an increase first")
    root_state = "checked after the launch (--instance-id)"
    if args.instance_id:
        attr = aws("describe-instance-attribute", "--instance-id", args.instance_id,
                   "--attribute", "instanceInitiatedShutdownBehavior")
        if attr.get("InstanceInitiatedShutdownBehavior", {}).get("Value") != "stop":
            problems.append("shutdown behaviour is not 'stop'")
        mapping = aws("describe-instances", "--instance-ids", args.instance_id)
        inst = mapping["Reservations"][0]["Instances"][0]
        if inst["InstanceType"] != TYPE:
            problems.append(f"instance type {inst['InstanceType']}, expected {TYPE}")
        if inst.get("InstanceLifecycle"):
            problems.append(f"instance lifecycle is {inst['InstanceLifecycle']}, not on-demand")
        groups = sorted(g["GroupId"] for g in inst.get("SecurityGroups", []))
        if sg_id is not None and groups != [sg_id]:
            problems.append(f"the instance's security groups are {groups}, expected only "
                            f"{args.security_group} ({sg_id})")
        blocks = inst.get("BlockDeviceMappings", [])
        data = [b for b in blocks if b["Ebs"]["VolumeId"] == args.data_volume]
        if not data:
            problems.append("the data volume is not attached to this instance")
        elif data[0]["Ebs"]["DeleteOnTermination"]:
            problems.append("the data volume has DeleteOnTermination=true (must be false)")
        roots = [b for b in blocks if b.get("DeviceName") == inst.get("RootDeviceName")]
        if not roots:
            problems.append(f"no root volume at {inst.get('RootDeviceName')}")
            root_state = "missing"
        else:
            rv = aws("describe-volumes", "--volume-ids", roots[0]["Ebs"]["VolumeId"])["Volumes"][0]
            root_state = f"{rv['VolumeType']}, {rv['Size']} GB"
            if rv["VolumeType"] != "gp3":
                problems.append(f"the root volume is {rv['VolumeType']}, not gp3")
            if root_gb is not None and rv["Size"] != root_gb:
                problems.append(f"the root volume is {rv['Size']} GB, not the AMI's {root_gb} GB")
            if not roots[0]["Ebs"]["DeleteOnTermination"]:
                problems.append("the root volume has DeleteOnTermination=false (it would be left "
                                "behind, billed)")
            problems += gp3_extra_problem("root volume", rv)
            if rv.get("Encrypted") is not False:
                problems.append(f"root volume Encrypted={rv.get('Encrypted')}, expected false")
    cost = args.hours * RATE
    share = BUDGET.get(args.stage)
    ebs_day = ebs_usd(1, DATA_GB + (root_gb or 0))
    print(f"PRE-LAUNCH CHECKLIST ({dt.datetime.now():%Y-%m-%d %H:%M})")
    print(f"  region {REGION} | type {TYPE} ({vcpus} vCPU) on-demand | AMI {args.ami}")
    print(f"  root volume: {root_device}, gp3, {root_gb} GB (the AMI's size) -- launch with "
          f"\"DeviceName={root_device},Ebs={{VolumeSize={root_gb},VolumeType=gp3,"
          f"DeleteOnTermination=true}}\" | now: {root_state}")
    print(f"  data volume {args.data_volume} | {TYPE} offered in {offered} | on-demand G/VT "
          f"vCPU quota: {quota}")
    print(f"  security group {args.security_group} ({sg_id}): {sg_summary}")
    print(f"  EBS encryption by default: {encryption_default} | data volume Encrypted="
          f"{vol[0].get('Encrypted') if vol else '?'} (root: checked after the launch)")
    print("  shutdown behaviour = stop | watchdog + hard cap: armed by env/aws_p5_session.sh")
    print(f"  rate ${RATE}/h (instance ${INSTANCE_RATE} + public IPv4 ${IPV4_RATE}) | EBS "
          f"{DATA_GB} + {root_gb} GB = ${ebs_day:.4f}/day (export P5_EBS_PER_DAY={ebs_day:.4f})")
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
    if args.hours is not None and ledger:
        raise SystemExit("give --ledger or --hours, not both")
    hours = float(args.hours if args.hours is not None else ledger.get("hours", 0.0))
    days = (dt.date.today() - dt.date.fromisoformat(args.volume_created)).days + 1
    gb = sum(v["Size"] for v in vols)              # root + data, as they exist now
    spend = hours * RATE + ebs_usd(days, gb)
    print(f"POST-SESSION ({dt.datetime.now():%Y-%m-%d %H:%M})")
    print(f"  running instances = {len(running)} {running}")
    print(f"  volumes = {len(vols)}: " + ", ".join(
        f"{v['VolumeId']} {v['Size']} GB {v['State']}" for v in vols))
    print(f"  Elastic IPs = {len(eips)} | snapshots = {len(snaps)}")
    print(f"  spend so far: {hours:.2f} instance-h x ${RATE} + {days} d EBS of {gb} GB "
          f"(${ebs_usd(days, gb):.2f}) = ${spend:.2f} of hard stop ${HARD_STOP} "
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
    pre.add_argument("--security-group", default=SG_NAME, help="the SSH group's name")
    pre.add_argument("--instance-id", default="", help="for a START of a stopped instance")
    pre.add_argument("--stage", required=True, choices=sorted(BUDGET))
    pre.add_argument("--hours", type=float, required=True)
    post = sub.add_parser("postsession")
    post.add_argument("--ledger", default="", help="ledger.json copied from the instance")
    post.add_argument("--volume-created", required=True)
    post.add_argument("--hours", type=float, default=None,
                      help="instance-hours so far, when no ledger copy is on this PC")
    args = parser.parse_args()
    return prelaunch(args) if args.cmd == "prelaunch" else postsession(args)


if __name__ == "__main__":
    sys.exit(main())

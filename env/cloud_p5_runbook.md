# P5 AWS runbook (user rules of 2026-11-21; USD 120 credits since 2026-11-22, zero extra bills)

**On AWS:** the Pexels training pool, the reconstruction bank and training.
**Never on AWS:** test data (AIGVDBench, AEGIS) and any scoring. Those are Kaggle only.

**Who does what:** anything that **starts** spending (launch, start, resize) happens only
after the user's explicit OK. Anything that **stops** spending (stop, terminate) may be done
at once; the user is told afterwards.

## 0. Budget (USD; stop and ask if a projection exceeds its share)

| stage | share | how it is projected |
|---|---|---|
| setup + Pexels download | 5 | about 1.5 instance-h |
| reconstruction | **≤ 45** | **from the pilot's measured s/clip, before any launch** |
| training | 15 | 9 frame models; P3 S2-D: 1.212 T4 GPU-h each |
| features / temporal heads | 3 | |
| EBS (200 GB data + the root at the AMI's size, gp3 at about $0.08/GB-month) | 5 | about $0.533/day for the data volume, plus the root; billed while stopped |
| reserve | 27 | |

- **The hard stop is $75.** Three things enforce it: `scripts/p5_cloud_guard.py`
  (`--cap-usd 75`), the user's AWS Budget action, and each session's hard cap.
- **After every session:** `python scripts/p5_aws_check.py postsession ...` logs
  instance-h × $1.217 + EBS days.
- **$1.217/h** = $1.212 on-demand g5.2xlarge + $0.005 for the public IPv4 address (user,
  2026-11-22). It is the default of the guard's rate (`P5_RATE`) and of `p5_aws_check.py`.

## 1. Rules wired into every session (`env/aws_p5_session.sh`; tests in `tests/test_p5_cloud.py`)

1. **A shutdown trap on EXIT and on ERR** (and on INT/TERM). It writes the session record,
   runs the backup, then stops the instance.
2. **An idle watchdog:** GPU 0% and no job process for 20 min → shutdown.
3. **A hard cap**, scheduled at session start: `shutdown -h +ceil(estimate × 1.5 × 60)`.
4. **Shutdown behaviour = stop**, set at launch. The volumes and every checkpoint survive.
5. **One instance at a time:** g5.2xlarge, us-east-2, on-demand. `prelaunch` refuses if
   another `dbp-p5` instance exists.
6. **No paid extras:** no Elastic IP, no snapshot, no S3, no NAT, no load balancer, no other
   paid service.
7. **RunPod only with the user's approval** (section 7).

## 2. Once: IAM users, keys, the data volume and the launch (user's PC, PowerShell)

**No AWS Agent Toolkit** (user, 2026-11-22): no MCP server, no `aws login`, no rules
file, no uv. The AWS CLI v2 is installed by the user from the official `AWSCLIV2.msi`.

**IAM, once, in the console:**
- **`dbp-agent`** (exists, user 2026-11-22): the AWS managed policy `ReadOnlyAccess`, plus
  the user's customer-managed `dbp-agent-stop`: Allow `ec2:StopInstances` and
  `ec2:TerminateInstances` on `arn:aws:ec2:us-east-2:*:instance/*`. Console access disabled.
  Claude Code runs with these keys only. `env/aws_iam/dbp-agent-inline.json` (the same allow,
  plus explicit denies) is kept for reference and is not needed: ReadOnlyAccess grants no
  write.
- **`dbp-launch`**: the customer-managed policy `env/aws_iam/dbp-launch.json`, attached to
  the user. It is too large for a user's inline policy. Its scope:
  - us-east-2 only, and g5.2xlarge only;
  - gp3 volumes of at most 200 GB;
  - only resources tagged `dbp-p5*` can be started, stopped, attached or torn down;
  - an explicit deny of S3, EIPs, snapshots, images, NAT, load balancers and resizes, and of
    every action not listed.

  Its keys are typed by the user, only for a launch, a start or the teardown.
- **Validation:** the console's policy editor must show 0 errors and 0 warnings for both
  policies; it validates the action names and condition keys. Then run the `--dry-run`
  tests below.
- **Keys:** one access key per user, kept in a password manager. Never in a file, never in
  `~/.aws`.

**Keys in a shell** (`env/aws_p5_env.ps1`: no echo, no history, environment only, no file;
it refuses keys that are not `dbp-<role>`'s):
```powershell
Set-ExecutionPolicy -Scope Process Bypass     # this window only, if scripts are blocked
.\env\aws_p5_env.ps1 -Role agent
claude                                        # or close EVERY VS Code window, then: code F:\thesis-2\Implementation
```
A VS Code that is already running does not pick up a new environment, so it must be started
from this shell.

**For a launch or a start, open a second window.** Run `.\env\aws_p5_env.ps1 -Role launch`,
then the commands below, then `.\env\aws_p5_env.ps1 -Clear`, then close the window.

**Permission tests: `--dry-run` is free and creates nothing.**
- In the `launch` window:
  - expect `DryRunOperation` (allowed):
    `aws ec2 run-instances --dry-run --image-id <AMI> --instance-type g5.2xlarge --block-device-mappings "DeviceName=<ROOT_DEVICE>,Ebs={VolumeSize=<ROOT_GB>,VolumeType=gp3,DeleteOnTermination=true}" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=dbp-p5}]"`
    (the gp3 root mapping matters: the policy refuses any other volume type, and an AMI may
    default to gp2)
  - expect `UnauthorizedOperation`: the same with `--instance-type g5.4xlarge`;
  - expect `UnauthorizedOperation`: `aws ec2 allocate-address --dry-run`.
- In the `agent` window (Claude Code runs it): the g5.2xlarge `run-instances --dry-run`
  above must give `UnauthorizedOperation`.

**Done by the agent, read-only, with its results recorded in the log:**
- the AMI, the current Deep Learning Base (NVIDIA driver, Ubuntu 22.04):
  `aws ssm get-parameters-by-path --path /aws/service/deeplearning/ami/x86_64 --recursive --query "Parameters[?contains(Name,'base-oss-nvidia-driver-gpu-ubuntu-22.04')].[Name,Value]" --output text`;
- the availability zones that offer g5.2xlarge:
  `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g5.2xlarge`.

**Once, in the `launch` window** (`<AZ>` is one of those zones):
```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\dbp-p5          # the private key stays on this PC
aws ec2 import-key-pair --key-name dbp-p5 --public-key-material fileb://$HOME/.ssh/dbp-p5.pub
aws ec2 create-security-group --group-name dbp-p5 --description "dbp p5 ssh"
aws ec2 authorize-security-group-ingress --group-name dbp-p5 --protocol tcp --port 22 --cidr "$((Invoke-RestMethod https://checkip.amazonaws.com).Trim())/32"
aws ec2 create-volume --availability-zone <AZ> --size 200 --volume-type gp3 --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=dbp-p5-data}]"
```

**SSH (user, 2026-11-22):** port 22 is opened **only to your current public IP as a /32**,
never 0.0.0.0/0 and never IPv6. `prelaunch` refuses if the `dbp-p5` group allows 0.0.0.0/0
or ::/0, any range wider than /32, more than one address, any port but tcp 22, or another
group or prefix list. It prints only the prefix length, never the address. If your IP has
changed, the old rule must be revoked before the new one is added.

**Launch, only after the user's OK, and only after `prelaunch` prints OK.** The agent runs
the check:
```powershell
python scripts/p5_aws_check.py prelaunch --ami <AMI> --data-volume <VOL> --stage setup_pexels --hours 2.0
```

It checks:
- region and type;
- the on-demand G/VT vCPU quota against g5.2xlarge's vCPUs;
- that g5.2xlarge is offered in the volume's zone;
- the AMI's root device and snapshot size. It prints the exact root mapping to launch with:
  gp3, at the AMI's size (its minimum), DeleteOnTermination=true;
- the volume: 200 GB gp3, with no extra IOPS or throughput;
- the `dbp-p5` SSH group (above);
- that no other P5 instance exists;
- the projection at $1.217/h against the stage's share, and the EBS cost per day of the data
  + root volumes (`P5_EBS_PER_DAY`, for section 3).

Then the user, in the `launch` window:
```powershell
aws ec2 run-instances --image-id <AMI> --instance-type g5.2xlarge --key-name dbp-p5 `
  --security-groups dbp-p5 --placement AvailabilityZone=<AZ> `
  --instance-initiated-shutdown-behavior stop `
  --block-device-mappings "DeviceName=<ROOT_DEVICE>,Ebs={VolumeSize=<ROOT_GB>,VolumeType=gp3,DeleteOnTermination=true}" `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=dbp-p5}]" `
  --user-data "#!/bin/bash`nshutdown -h +180"
aws ec2 attach-volume --volume-id <VOL> --instance-id <ID> --device /dev/sdf
```

- **The boot cap:** the `--user-data` line schedules a stop at 180 min (2.0 h × 1.5) from
  the first boot, before anyone logs in. The session wrapper's own cap replaces it when the
  session starts.
- **The root volume (user, 2026-11-22):** gp3, set explicitly, so the AMI's default type
  (possibly gp2) is never used; the `dbp-launch` policy refuses any other type anyway.
  `<ROOT_DEVICE>` and `<ROOT_GB>` are the AMI's root device name and snapshot size, exactly as
  `prelaunch` prints them; the size is the AMI's minimum.
- **The data volume:** an attached volume defaults to DeleteOnTermination=false.

After the launch, the agent runs `prelaunch` again with `--instance-id <ID>`. It now also
checks:
- shutdown = stop, and on-demand;
- the data volume attached with DoT=false;
- the root volume: gp3, at the AMI's size, DoT=true, with no extra IOPS or throughput;
- the instance's only security group is `dbp-p5`.

**Each later START** (`aws ec2 start-instances --instance-ids <ID>`, in a `launch` window)
needs the user's OK, after the agent's `prelaunch --instance-id <ID> --stage <stage> --hours <h>`.
On a start the user-data boot cap does **not** run again, so start the session wrapper at
once. The agent can stop the instance at any time (`dbp-agent` may stop and terminate).

## 3. On the instance (inside `tmux`)

```bash
lsblk                                   # the 200 GB device, e.g. /dev/nvme1n1 (FILL)
sudo mkfs.ext4 -L dbpdata /dev/nvme1n1  # FIRST session only
sudo mkdir -p /data && sudo mount LABEL=dbpdata /data && sudo chown ubuntu /data
mkdir -p /data/repo && cd /data/repo && unzip -q ~/dbp-code-<hash>.zip && cd thesis_dbp
export HISTCONTROL=ignorespace
 read -rs HF_TOKEN && export HF_TOKEN                   # never written, never echoed
bash env/cloud_p5_setup.sh /data                         # torch 2.11.0 (Kaggle's) cu130
# if it stops at "CUDA not available": TORCH_INDEX=cu128 bash env/cloud_p5_setup.sh /data
#   -- record that training runs on cu128 (scoring on Kaggle is cu130; hashes are checked)
source /data/venv/bin/activate
export HF_HOME=/data/cache/hf TORCH_HOME=/data/cache/torch PIP_CACHE_DIR=/data/cache/pip
export P5_VOLUME_CREATED=<YYYY-MM-DD> P5_STATE=/data/p5/session
export P5_EBS_PER_DAY=<as prelaunch prints it>     # data + root volume
export P5_BACKUP_CMD="rclone copy /data/p5/session gdrive:dbp-p5/session"
```

**rclone (approved by the user, 2026-11-22). On the AWS instance only**, never on the PC or
on Kaggle. Small backups only (session ledger, records, logs, at the end the models), into
one dedicated Drive folder, `dbp-p5`.
- **Hosts:**
  - `archive.ubuntu.com` and `security.ubuntu.com`, for the package;
  - `accounts.google.com`, for the sign-in, in the PC's browser;
  - `oauth2.googleapis.com` and `www.googleapis.com`, from the instance.
- **Scope `drive.file`:** rclone sees only the files it created itself.
- **Install, once**, from the Ubuntu archive only. The AMI's own apt sources (the EC2
  regional mirror, NVIDIA's) are switched off for this command:
  ```bash
  CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  printf 'deb http://archive.ubuntu.com/ubuntu %s main universe\ndeb http://security.ubuntu.com/ubuntu %s-security main universe\n' \
    "$CODENAME" "$CODENAME" | sudo tee /etc/apt/dbp-ubuntu-archive.list
  ONLY_ARCHIVE="-o Dir::Etc::SourceList=/etc/apt/dbp-ubuntu-archive.list -o Dir::Etc::SourceParts=-"
  sudo apt-get $ONLY_ARCHIVE update && sudo apt-get $ONLY_ARCHIVE install -y rclone
  rclone version | head -1                            # recorded in the log
  ```
  apt checks the archive's signatures as usual.
- **Sign-in, once:**
  1. On the PC, open an SSH session with a tunnel for the sign-in's local redirect:
     `ssh -i $HOME\.ssh\dbp-p5 -L 53682:127.0.0.1:53682 ubuntu@<public DNS>`.
  2. In it, run:
     ```bash
     rclone config create gdrive drive scope drive.file
     chmod 600 ~/.config/rclone/rclone.conf
     rclone mkdir gdrive:dbp-p5 && rclone lsd gdrive:     # only dbp-p5 is visible
     ```
  3. Open the `http://127.0.0.1:53682/auth?...` link rclone prints in the PC's browser, and
     sign in.
  - The token stays in `~/.config/rclone/rclone.conf`, on the root volume (deleted at
    termination). It is never printed, copied or committed.
  - **NOT-VERIFIED:** that the Ubuntu archive's rclone version completes Google's sign-in this
    way. If it does not, stop and tell the user; the ledger then comes by `scp` (section 5).

## 4. Jobs: each one through the session wrapper; the instance stops when it ends

```bash
# FIRST SESSION (approved 2026-11-22, about 2 h; it is also the AWS test launch):
# (a) the Pexels training pool -- downloaded ONCE (shards 4:35; no video id of the pilot pool)
# (b) the A10G smoke (all six members; deterministic, fp32 without TF32): it measures the
#     A10G s/clip that replaces the assumed T4 -> A10G factor. It writes to the same
#     --reports as (c): the reconstruction runs only members whose smoke there is OK.
# Both run in ONE wrapper call; the instance stops after (b). The reconstruction projection
# is redone from (b) and sent to the user BEFORE (c) is started.
bash env/aws_p5_session.sh --label setup_pexels --estimate-hours 2.0 -- bash -c '
  python scripts/p5_fetch_reals.py --source hub --pool train --shards 4:35 --n 2200 \
    --out-root /data/p5/reals --cache /data/cache/shards --reports reports/p5_train_reals \
    --exclude-ids reports/kaggle_20261121/p5_reals/p5_reals_pilot_manifest.csv --workers 8 &&
  python scripts/p5_pilot.py --phase smoke --members wan21,hunyuan,cogvx,ltx090,mochi,sd15_pf \
    --device cuda --expect env/p5_pilot_expect.json --reals-root /data/p5/reals \
    --manifest reports/p5_train_reals/p5_reals_train_manifest.csv --reports reports/p5_train_recon'
# (c) the reconstruction -- resumable per clip; a restart never redoes a finished clip
#     (each clip is recorded in recon_<member>.jsonl)
#   --splits train,validation: the 2,000 train + validation clips only; the 200
#     threshold-held-out reals are never reconstructed (user, 2026-11-22)
#   --fp16-evidence: fp16 only where the pilot's fp16 check passed AND this run's smoke (b)
#     passed it again, deterministically; mochi and cogvx always fp32 (user, 2026-11-22)
bash env/aws_p5_session.sh --label recon --estimate-hours <projected> -- \
  python scripts/p5_pilot.py --phase reconstruct --members <approved members> \
    --device cuda --expect env/p5_pilot_expect.json --reals-root /data/p5/reals \
    --manifest reports/p5_train_reals/p5_reals_train_manifest.csv \
    --splits train,validation --fp16-evidence reports/kaggle_20261121/p5_pilot \
    --recon-root /data/p5/recon --reports reports/p5_train_recon
```

- **The reconstruction is launched only after the user approves its projection** (P5-C).
  The projection comes from the A10G s/clip measured by (b).
- **HunyuanVideo** (user, 2026-11-22) stays in the bank for (b). After (b) the agent shows
  the user its measured cost against the reconstruction's $45 share; **the user decides**
  (re-allocating from the reserve is possible). The agent neither drops nor keeps it alone.
- **`<approved members>`** is the member list the user approves with the projection.
- **Training (P5-D)** checkpoints every 500 steps on `/data` and resumes from the last one.
- **Pexels is downloaded once.** If (a) stops midway it restarts after
  `rm -r /data/p5/reals/train`. The shard plan makes the result identical.

## 5. After every session (user's PC)

rclone does not run on the PC. The user downloads `dbp-p5/session/ledger.json` from Drive
in the browser into `F:\thesis-2\kaggle_raw\p5_session\` (outside the repository, like every raw archive, §7b). Then:
```powershell
python scripts/p5_aws_check.py postsession --ledger F:\thesis-2\kaggle_raw\p5_session\ledger.json --volume-created <YYYY-MM-DD>
```

**If there is no ledger copy** (a failed backup), `postsession` takes the instance-hours with
`--hours`. The agent sums them from each session's start and stop times in
`describe-instances` (`LaunchTime`, `StateTransitionReason`). The ledger on `/data` is
copied at the next start, with `scp`.

It prints:
- running instances = 0;
- volumes = root + data only;
- Elastic IPs = 0; snapshots = 0;
- spend so far against the budget.

The spend goes into the log.

## 6. End (29 Nov, or when training is done)

1. **Back up to Drive** (`rclone copy`): the models (LoRA, heads, temporal heads + their
   sha256 record), manifests, records, logs, the env lock, and the final archive. **Never
   the clips.**
2. **Upload the models to the private Kaggle dataset**, with
   `KAGGLE_USERNAME`/`KAGGLE_KEY` entered by `read -rs`. Check their hashes on Kaggle.
3. **Delete the secrets on the instance:**
   - `rclone config delete gdrive && shred -u ~/.config/rclone/rclone.conf`;
   - `unset HF_TOKEN KAGGLE_KEY KAGGLE_USERNAME`.
   - The user then removes rclone's access in the Google account's third-party access page
     (in the browser).
4. `aws ec2 terminate-instances --instance-ids <ID>`, then
   `aws ec2 delete-volume --volume-id <VOL>` (the data volume does not go with the
   instance).
5. **Delete the security group:** `aws ec2 delete-security-group --group-name dbp-p5`.
6. **Confirm everything is gone:**
   - `postsession` prints running = 0 and volumes = 0;
   - `describe-addresses` and `describe-snapshots` are empty;
   - the billing console shows no running cost.

   Record it in the log.

## 7. RunPod: only with the user's approval

The same wrapper, with three differences:
- **The pod stops itself when its job ends**, through a small stop script given as
  `P5_SHUTDOWN` (`RUNPOD_API_KEY` by `read -rs`).
  - The wrapper calls `$P5_SHUTDOWN now` and `$P5_SHUTDOWN +<min>`. The script turns `now`
    into `runpodctl stop pod $RUNPOD_POD_ID`, and `+<min>` into a timed stop.
  - It is built, tested and dry-run when RunPod is approved. **Not before.**
- **The same egress rule:** only small files leave.

At the end, delete the pod and its volume. The data is deleted, not kept.

# P5 AWS runbook (user rules of 2026-11-21; USD 100 credits, zero extra bills)

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
| EBS (200 GB gp3 at about $0.08/GB-month) | 5 | about $0.533/day; billed while stopped |
| reserve | 27 | |

- **The hard stop is $75.** Three things enforce it: `scripts/p5_cloud_guard.py`
  (`--cap-usd 75`), the user's AWS Budget action, and each session's hard cap.
- **After every session:** `python scripts/p5_aws_check.py postsession ...` logs
  instance-h × $1.21 + EBS days.

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

## 2. Once: the data volume and the launch (user's PC; AWS keys in env vars only)

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-2   # typed, not saved
# the AMI: the current Deep Learning Base (NVIDIA driver, Ubuntu 22.04) -- FILL, record it
aws ssm get-parameters-by-path --path /aws/service/deeplearning/ami/x86_64 --recursive \
  --query "Parameters[?contains(Name,'base-oss-nvidia-driver-gpu-ubuntu-22.04')].[Name,Value]" --output text
# the data volume, ONCE (200 GB gp3, fixed size); AZ FILL (one where g5.2xlarge launches)
aws ec2 create-volume --availability-zone us-east-2a --size 200 --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=dbp-p5-data}]'
# security group: SSH from your IP only
aws ec2 create-security-group --group-name dbp-p5 --description "dbp p5 ssh"
aws ec2 authorize-security-group-ingress --group-name dbp-p5 --protocol tcp --port 22 \
  --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

**Launch, only after the user's OK, and only after `prelaunch` prints OK.**

```bash
python scripts/p5_aws_check.py prelaunch --ami <AMI> --data-volume <VOL> --stage setup_pexels --hours 1.5
aws ec2 run-instances --image-id <AMI> --instance-type g5.2xlarge --key-name <KEY> \
  --security-groups dbp-p5 --placement AvailabilityZone=us-east-2a \
  --instance-initiated-shutdown-behavior stop \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dbp-p5}]'
aws ec2 attach-volume --volume-id <VOL> --instance-id <ID> --device /dev/sdf
# an attached volume defaults to DeleteOnTermination=false, so it survives a termination
python scripts/p5_aws_check.py prelaunch --ami <AMI> --data-volume <VOL> --instance-id <ID> \
  --stage setup_pexels --hours 1.5     # now also checks: shutdown=stop, attached, DoT=false
```

**Each later START of the stopped instance** (`aws ec2 start-instances --instance-ids <ID>`)
also needs the user's OK, after `prelaunch --instance-id <ID> --stage <stage> --hours <h>`.

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
# backups (small files only): rclone from the Ubuntu archive (apt); its token stays in
# ~/.config/rclone/rclone.conf (chmod 600) on the instance, deleted at teardown
export P5_BACKUP_CMD="rclone copy /data/p5/session gdrive:dbp-p5/session"
```

## 4. Jobs: each one through the session wrapper; the instance stops when it ends

```bash
# (a) the Pexels training pool -- downloaded ONCE (shards 4:35; no video id of the pilot pool)
bash env/aws_p5_session.sh --label pexels --estimate-hours 1.5 -- \
  python scripts/p5_fetch_reals.py --source hub --pool train --shards 4:35 --n 2200 \
    --out-root /data/p5/reals --cache /data/cache/shards --reports reports/p5_train_reals \
    --exclude-ids reports/p5_reals/p5_reals_pilot_manifest.csv --workers 8
# (b) the A10G smoke, then (c) the reconstruction -- resumable per clip; a restart never
#     redoes a finished clip (each clip is recorded in recon_<member>.jsonl)
bash env/aws_p5_session.sh --label recon --estimate-hours <projected> -- \
  python scripts/p5_pilot.py --phase reconstruct --members wan21,hunyuan,cogvx,ltx090,mochi,sd15_pf \
    --expect env/p5_pilot_expect.json --reals-root /data/p5/reals \
    --manifest reports/p5_train_reals/p5_reals_train_manifest.csv \
    --recon-root /data/p5/recon --reports reports/p5_train_recon
```

- **The reconstruction is launched only after the user approves its projection** (P5-C).
  The projection comes from the pilot's T4 s/clip × an A10G factor stated as an
  assumption.
- **Training (P5-D)** checkpoints every 500 steps on `/data` and resumes from the last one.
- **Pexels is downloaded once.** If (a) stops midway it restarts after
  `rm -r /data/p5/reals/train`. The shard plan makes the result identical.

## 5. After every session (user's PC)

```bash
rclone copy gdrive:dbp-p5/session ./p5_session     # the ledger, written before the stop
python scripts/p5_aws_check.py postsession --ledger ./p5_session/ledger.json --volume-created <YYYY-MM-DD>
```

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
3. **Delete the secrets on the instance:** `rm ~/.config/rclone/rclone.conf`, then
   `unset HF_TOKEN KAGGLE_KEY KAGGLE_USERNAME`.
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

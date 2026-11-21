# P5 cloud runbook — AWS g5.2xlarge (primary) and RunPod (overflow)

**What runs here:** the Pexels training pool, the 6-member reconstruction bank and all P5
training. **What never runs here:** anything with test data (AIGVDBench, AEGIS), and any
scoring — those stay on Kaggle (user, 2026-11-21).
**Egress-free:** the clips never leave the instance. Only the model files (LoRA, heads,
the temporal heads: MBs) and small reports leave, uploaded to a **private** Kaggle dataset.
**Secrets** live only in environment variables of your shells: AWS keys on your PC;
`HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` on the instance, entered with `read -rs`,
never written to disk, never echoed.

Values marked **FILL** come from your console or a command you run; record each in the
log. Nothing here is a value I could verify by command.

---

## 0. Budget (stop and tell the user before $80 total)

- Rate: g5.2xlarge **$1.21/h** (your figure). RunPod A5000 **~$0.27/h** (your figure).
- The guard (`scripts/p5_cloud_guard.py`) stops the instance at **$75** (`--cap-usd 75`), so
  the $80 limit is never crossed by compute; the volume's cost goes in `--fixed-usd`
  (**FILL** from the console's EBS gp3 price × size × days; not verified here).
- The estimate per stage is in the hand-over report. Reconstruction is the one unknown
  until the pilot measures seconds per clip.

## 1. On your PC (AWS CLI v2, credentials in env vars only)

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-2   # typed, not saved
# the current Deep Learning Base (NVIDIA driver, Ubuntu 22.04) AMI id -- FILL, record it:
aws ssm get-parameters-by-path --path /aws/service/deeplearning/ami/x86_64 --recursive \
  --query "Parameters[?contains(Name,'base-oss-nvidia-driver-gpu-ubuntu-22.04')].[Name,Value]" --output text
MYIP=$(curl -s https://checkip.amazonaws.com)/32
aws ec2 create-security-group --group-name dbp-p5 --description "dbp p5 ssh"
aws ec2 authorize-security-group-ingress --group-name dbp-p5 --protocol tcp --port 22 --cidr "$MYIP"
aws ec2 run-instances --image-id <AMI-ID> --instance-type g5.2xlarge --key-name <YOUR-KEY> \
  --security-groups dbp-p5 --instance-initiated-shutdown-behavior stop \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}},
                            {"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dbp-p5}]'
```

- **Data volume 200 GB** (sizing in the report): the clips are ≤ 129.0 GB raw. That's
  2,400 reals × (1 + 6 members) × 25 frames × 320² × 3 B; PNG is smaller, and the pilot
  measures how much. Add the VAE weights (9.0 GB), a shard working set (about 2–3 GB) and
  the feature caches (5.6 GB).
- `DeleteOnTermination: true` on both volumes, so terminating cannot leave one billing.
  **Shutdown = stop**, so the volume and every checkpoint survive an auto-stop.
- The code goes over **as the code zip built from git's bytes**
  (`python scripts/build_code_zip.py`): no GitHub credentials ever reach the instance.

```bash
scp -i <YOUR-KEY>.pem F:/thesis-2/kaggle_raw/dbp-code-<hash>.zip ubuntu@<PUBLIC-IP>:~
ssh -i <YOUR-KEY>.pem ubuntu@<PUBLIC-IP>
```

## 2. On the instance (inside `tmux`, so a dropped SSH does not stop a job)

```bash
lsblk                                   # FILL: the 200 GB device, e.g. /dev/nvme1n1
sudo mkfs.ext4 -L dbpdata /dev/nvme1n1  # FIRST launch only
sudo mkdir -p /data && sudo mount LABEL=dbpdata /data && sudo chown ubuntu /data
mkdir -p /data/repo && cd /data/repo && unzip -q ~/dbp-code-<hash>.zip && cd thesis_dbp
cat COMMIT.txt                          # record it
export HISTCONTROL=ignorespace
 read -rs HF_TOKEN && export HF_TOKEN   # leading space: kept out of the history
bash env/cloud_p5_setup.sh /data        # venv, torch 2.11.0 (Kaggle's), cu130; record + lock
# if it stops at "CUDA not available" (driver older than R580): TORCH_INDEX=cu128 bash env/cloud_p5_setup.sh /data
source /data/venv/bin/activate
export HF_HOME=/data/cache/hf TORCH_HOME=/data/cache/torch PIP_CACHE_DIR=/data/cache/pip
```

- The g5's local NVMe instance store (if it is mounted) is **scratch only**: assume it is
  lost on stop (FILL: check in the console). Nothing that must survive goes there.
- After a stop and start: re-mount `/data`, `source` the venv, re-enter the secrets.

## 3. Jobs (each one resumable; the guard watches it)

```bash
# (a) the training pool: shards 4:35 of the committed plan, no video id from the pilot pool
python scripts/p5_fetch_reals.py --source hub --pool train --shards 4:35 --n 2200 \
  --out-root /data/p5/reals --cache /data/cache/shards --reports reports/p5_train_reals \
  --exclude-ids reports/p5_reals/p5_reals_pilot_manifest.csv --workers 8 > /data/logs/fetch.log 2>&1 &
JOB=$!; nohup python scripts/p5_cloud_guard.py --state /data/guard --rate 1.21 \
  --fixed-usd <FILL> --cap-usd 75 --watch-pid $JOB --keep-on-finish > /data/logs/guard.log 2>&1 &
```

- **Restart rule for (a):** a fetch that stopped midway is re-run from the start, after
  `rm -r /data/p5/reals/train` (the pool is about 1 h; the shard plan makes it identical).
- **(b) The smoke on the A10G**, before anything long:
  `scripts/p5_pilot.py --phase smoke --members wan21,hunyuan,cogvx,ltx090,mochi,sd15_pf,wan22_5b`
  with `--reports reports/p5_train_recon`. It records this GPU's fp16 verdicts and peak
  memory.
- **(c) The reconstruction:**
  `scripts/p5_pilot.py --phase reconstruct --members wan21,hunyuan,cogvx,ltx090,mochi,sd15_pf`
  with `--manifest reports/p5_train_reals/p5_reals_train_manifest.csv`,
  `--recon-root /data/p5/recon` and `--reports reports/p5_train_recon`.
  - It resumes per clip: a re-run skips every clip in its `recon_<member>.jsonl`.
- **(d) Training** (`P5-C`, not built yet): checkpoints every 500 steps on `/data`, resumed
  from the last one.

Start the guard after each job with `--watch-pid` for it. Without `--keep-on-finish` it
stops the instance when the job ends.

## 4. What leaves AWS (and the only time `KAGGLE_KEY` is used)

```bash
 read -rs KAGGLE_USERNAME && export KAGGLE_USERNAME
 read -rs KAGGLE_KEY && export KAGGLE_KEY
# a directory holding ONLY: LoRA adapters, heads, temporal heads, their sha256 record (MBs)
kaggle datasets version -p /data/p5/upload -m "P5 models <commit> <sha256 of the record>" -r zip
```

- The dataset is **private** (`muhammadadeelshah`).
- Scoring checks every file's hash against the committed model record before it scores, as
  P3b did.
- Copy back to your PC (for committing): `reports/p5_*` and `env/lock/cloud_p5_*.txt`.
  They are small text files: `scp -r ...`.

## 5. Terminate and delete checklist (end of P5, or when told)

1. Models uploaded and their hashes checked on Kaggle. Reports and the lock copied to your
   PC.
2. `aws ec2 terminate-instances --instance-ids <ID>`.
3. `aws ec2 describe-volumes --filters Name=status,Values=available --query "Volumes[].VolumeId"`
   must be empty; otherwise `aws ec2 delete-volume --volume-id <ID>` for each.
4. `aws ec2 describe-snapshots --owner-ids self` must show no P5 snapshot.
5. `aws ec2 delete-security-group --group-name dbp-p5`.
6. The billing console shows no running resource. Record the final spend in the log.

## 6. RunPod (overflow for LOLO folds; you pay)

- **Pod:** RTX A5000 24 GB (or A40), Ubuntu with CUDA 12.x.
- **Volume:** a network or volume disk sized as section 1 says.
- **Same sections 2–4.** Two differences:
  - the data mount is the pod's volume path (FILL);
  - the guard's stop command is `--stop-command "runpodctl stop pod $RUNPOD_POD_ID"`, with
    `RUNPOD_API_KEY` exported by `read -rs`.
- **The same egress rule:** only model files leave, to the private Kaggle dataset.
- **At the end: delete the pod and its volume.** The data is deleted, not kept. Record
  that in the log.

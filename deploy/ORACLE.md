# QuantPulse 24/7 for $0: Oracle Cloud Always Free (ARM)

**Set it up once, turn your PC off.** The Brain keeps supervising the Alpaca **paper** account on a free Oracle
Cloud server. GitHub CI decides what the server may run: it deploys a commit only after CI passed on that commit,
the ARM64 build included. A watchdog restarts a stalled Brain supervisor, and backups leave the server every night.
You watch it (or stop it) from your iPhone.

```
 iPhone (Safari + Tailscale app)           Windows PC (setup only: ssh)            GitHub (CI on every push)
        │ private, encrypted (Tailscale)            │                                      ▲ the server pulls
        ▼                                           ▼                                      │ (HTTPS, read-only)
 ┌──────────── Oracle Cloud Always Free: VM.Standard.A1.Flex, 2 OCPU · 6 GB · Ubuntu 24.04 ARM ────────────┐
 │  dashboard (password) ──► api: QuantPulse API · Brain supervisor · jobs · health · alerts               │
 │                             │  (starts only if the paper-only preflight passes)                        │
 │                             ├──► PostgreSQL 16 (all Brain history, on the boot volume)                │
 │  systemd timers:  watchdog (1 min) · auto-update (10 min, CI-gated) · backup (nightly) · restore test │
 └──────────────┬──────────────────────────────────────────────┬───────────────────────────────────────────┘
                ▼                                              ▼ nightly, write-only link
   Alpaca PAPER API (paper-api.alpaca.markets)       Oracle Object Storage (Always Free, 20 GB)
```

Nothing in QuantPulse's safety model changes on Oracle: the same image, the same paper-only preflight, the same
risk engine, kill switches, reconciliation and single-supervisor lease as on any other server. What is new is
around it: the server tooling (`deploy/qpops.py`, the `./qp` commands and `deploy/systemd/`), which never talks
to Alpaca and never asks QuantPulse for anything but read-only status.

---

## Cost: $0 a month

| Item | What QuantPulse uses | Always Free allowance | Monthly |
|---|---|---|---|
| Compute: `VM.Standard.A1.Flex` (Ampere ARM) | 2 OCPU, 6 GB RAM | 2 OCPU and 12 GB in total (since June 2026) | $0 |
| Block storage (the boot volume) | 100 GB | 200 GB in total | $0 |
| Object Storage (off-site backups) | ≈ 1–3 GB (about 30 generations of a compressed dump) | 20 GB; 50,000 requests a month | $0 (≈ 31 PUTs a month) |
| Outbound data | well under 10 GB (market data comes *in*; inbound is free) | 10 TB | $0 |
| Public IPv4 address (ephemeral, on the VM) | 1 | included | $0 |
| Tailscale (phone ↔ server) | Personal plan | free for personal use | $0 |
| ntfy.sh (phone alerts) | a few messages a day | 250 messages a day | $0 |
| healthchecks.io (server-down and backup alerts) | 2 checks | 20 checks | $0 |
| GitHub Actions (CI, incl. the ARM64 build) | a few runs a day | free and unlimited for a **public** repository; 2,000 minutes a month on a private one | $0 |
| Alpaca paper trading and IEX data | paper account | free | $0 |

**Total: $0 a month**, as long as the account stays on the **Free Tier** (do not upgrade to Pay As You Go) and the
VM stays at 2 OCPU / 6 GB. Oracle asks for a card to verify your identity at sign-up (a small authorisation that is
released); a Free Tier account is never charged. As a safety net, set a **budget alert** of $1 (Billing → Budgets):
it is free and emails you if anything ever starts costing money.

Prices and allowances as published by Oracle, Tailscale, ntfy, healthchecks.io and GitHub in September 2026; check
them when you sign up. **Nothing is bought by QuantPulse or by these scripts.**

### What "free" costs you in reliability

* **No SLA, and capacity is not guaranteed.** "Out of host capacity" when creating the VM is common: retry in
  another availability domain or a few hours later. Once created, a VM keeps its capacity.
* **Idle reclamation is a real risk.** Oracle may stop an Always Free VM when, over 7 days, CPU (95th percentile),
  network *and* memory use all stay under 20%. QuantPulse is light:
  - in a rehearsal (an empty database, the supervisor off) the whole stack used about **450 MB = 7.5% of 6 GB**;
  - with the Brain working, the S&P 500 data loaded and the daily model runs, it uses more: PostgreSQL's 1 GB
    cache fills as the history grows, and the stock model peaks near 1.5 GB;
  - while the market is closed the Brain does real research (grading, analyses, backtests, walk-forward tests:
    see the README's *24/7 operating model*). That uses CPU and memory outside the session, within limits that
    keep execution first: no new job above 70% memory, running jobs stop at 85%, and one job at a time;
  - whether that stays above 20% (1.2 GB) *all week* has not been measured on a real server.

  What protects you:
  - the watchdog records CPU and memory every minute;
  - `./qp status` shows the 7-day picture (`idle check`);
  - the server alerts you ("Oracle may stop this VM as idle") once a day of samples looks idle;
  - the `QP_HEARTBEAT_URL` check alerts you if the VM does get stopped.

  A reclaimed VM is *stopped*, not deleted. Start it again in the console, and QuantPulse recovers by itself (see
  [Recovery](#recovery)). The only way to remove the rule is upgrading to Pay As You Go, which also exposes you to
  charges; this guide does not do it. Running a process only to burn CPU or memory is not something this setup
  does.
* **The home region is permanent.** Always Free resources live only in the region chosen at sign-up.

---

## What was configured for you (in this repository)

| File | What it does |
|---|---|
| `deploy/bootstrap-oracle.sh` | One-time server preparation: Docker (arm64) at boot with graceful stops, swap and memory settings, Oracle's firewall kept (SSH only), SSH keys only, automatic security updates with a 07:40 UTC reboot when needed, UTC clock (chrony), journald cap, Tailscale |
| `deploy/compose.yaml` | The stack (unchanged services; PostgreSQL sized for 6 GB; the API gets 180 s to stop cleanly; images tagged per commit; the `backup` container is the database toolbox) |
| `deploy/qp` | The helper: `setup`, `start`, `status`, `update`, `rollback`, `backup`, `restore`, `restore-test`, `ci-gate`, `install-timers`, … |
| `deploy/qpops.py` | The server's operations (Python standard library only): the CI gate, the atomic deploy with automatic rollback, the watchdog, backups and uploads, the restore test, the status page |
| `deploy/systemd/` | Four timers: watchdog (every minute), auto-update (every 10 minutes), backup (nightly 07:15 UTC), restore test (Sundays 09:40 UTC) |
| `deploy/ops.env.example` | The server's own settings and secrets (copied to `deploy/ops.env`, never given to a container) |
| `.github/workflows/ci.yml` | New job **`docker (arm64)`**: builds the production image for `linux/arm64` and smoke-tests it; the PostgreSQL job also runs the real backup/restore test |
| `GET /api/v1/system/watchdog` | Read-only: is the Brain supervisor alive — not just the process? |

---

## Safety, in one list

* **Paper only, unchanged.** The API refuses to start unless the preflight passes (`QP_ALPACA_PAPER=true`, the
  paper endpoint, a `PK…` paper key, no live URL anywhere, protected risk limits at least as strict as shipped).
  The server tooling never reads the Alpaca keys and never calls Alpaca.
* **The watchdog cannot cause an order.** Its only request to QuantPulse is `GET /api/v1/system/watchdog`; its
  only action is `docker compose restart api` — the same graceful stop a deploy does (no new order, the running
  tick drains, a final reconciliation, the Brain lease handed over). The new process reconciles with Alpaca and
  runs the execution audit before anything else. Both properties are tested
  (`tests/integration/test_watchdog.py`, `tests/unit/test_qpops.py`).
* **Only tested commits run.** A commit is deployed only when the CI workflow's push run on *exactly that commit*
  completed with success and each required job passed: `test (3.11)`, `test (3.12)`, `postgres`, `docker`,
  `docker (arm64)`. Anything else — no run yet, still running, a failure, a skipped or missing job, GitHub
  unreachable, an unexpected answer — means no deploy.
* **GitHub never reaches the server.** The server *pulls* over HTTPS (a public repository needs no credential; a
  private one uses a read-only token in `deploy/ops.env`). No SSH key, no webhook, no inbound port.
* **No credential in Git.** `deploy/.env` (Alpaca keys, database password, API token) and `deploy/ops.env` (the
  backup link, an optional GitHub token) are ignored by git and by the Docker build; `./qp` never prints them.
* **A second preflight, on the server.**
  - `./qp start` and every deploy first run the server's own preflight (`./qp host-preflight`), which refuses
    to start or deploy anything if:
    - any line of `deploy/.env`, `deploy/ops.env` or the server's environment names Alpaca's live or broker API;
    - `QP_ALPACA_PAPER` is not exactly `true`, or `QP_DEPLOYMENT` is anything but `cloud`;
    - the compose file (the running one, or a new version's) no longer forces both.
  - The API's own preflight then runs on the image itself. `deploy/ops.env` never reaches the container, so the
    server's preflight is what covers it.
* **Nothing public: the dashboard is Tailscale-only.**
  - Oracle's default firewall lets in SSH only; the API and the dashboard are bound to 127.0.0.1.
  - The dashboard is published with `tailscale serve` (your tailnet only).
  - `QP_DASHBOARD_ACCESS=tailscale` (the default) makes `./qp start --public` refuse.
  - `./qp status` flags the dashboard as PUBLIC if Tailscale Funnel or the caddy profile ever exposes it.
* **Read-only access is off until you turn it on, and then it can only read** (see
  [Read-only access](#read-only-access-qp-reader)).
  - `./qp reader on` makes a second key and publishes a reader gateway on your tailnet only. The gateway
    forwards GET requests for the monitoring pages and refuses everything else. It holds no Alpaca key, no
    database password and not the API token.
  - The API checks the key again: with it, only GETs to the monitoring pages pass, never an order, a control,
    a switch, a setting or a promotion. `./qp reader off` revokes it at once.
  - `./qp status` flags the reader as PUBLIC if Tailscale Funnel ever exposes it.

---

## One-time setup (about 45 minutes)

Commands are typed into **PowerShell** on the PC, or after `ssh` into the server.

### 1. An SSH key on the PC (once)

```powershell
ssh-keygen -t ed25519
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Press Enter at the questions. The second command prints your **public** key (safe to paste into Oracle).

### 2. The Oracle account

1. Sign up at <https://signup.cloud.oracle.com> for **Oracle Cloud Free Tier**. Choose the **home region**
   carefully (permanent), e.g. one near you with Ampere capacity (US East (Ashburn), US West (Phoenix), Germany
   Central (Frankfurt), UK South (London), …).
2. Card verification: a temporary authorisation; the account stays *Free Tier*. Do **not** click "Upgrade to Pay
   As You Go".
3. Billing & Cost Management → **Budgets** → Create budget: $1, alert at 100% to your email (free safety net).

### 3. The server (Compute → Instances → Create instance)

| Field | Value |
|---|---|
| Name | `quantpulse` |
| Image | **Canonical Ubuntu 24.04** (the aarch64 build is picked with the Ampere shape) |
| Shape | Change shape → **Ampere** → **VM.Standard.A1.Flex**, **2 OCPU**, **6 GB** memory |
| Networking | Create new virtual cloud network + public subnet; **Assign a public IPv4 address** |
| SSH keys | Paste public key → the line printed in step 1 |
| Boot volume | Specify a custom size: **100 GB** (within the 200 GB free); keep in-transit encryption on |

"Out of capacity for shape VM.Standard.A1.Flex": pick another availability domain in the same page, or retry
later; nothing is charged meanwhile. Note the **public IP address** once it runs.

### 4. The network (firewall)

The new subnet's default **security list** allows TCP 22 (SSH) and ICMP only — keep it that way. QuantPulse needs
**no inbound port**: Tailscale connects outward. Tighten SSH to your own address (Networking → Virtual cloud
networks → your VCN → Security Lists → Default → edit the 22 rule: source `<your IP>/32`), or, once Tailscale
works (step 6), remove the rule and use `ssh ubuntu@quantpulse` over Tailscale only.

On the VM itself Oracle's Ubuntu image has its own iptables rules (SSH only); `bootstrap-oracle.sh` keeps them
and refuses to continue if a rule would expose port 8000 or 8501.

### 5. Prepare the server

```powershell
ssh ubuntu@203.0.113.10          # your IP; answer "yes" once
```

```bash
sudo git clone https://github.com/kasmdkasdsad/Finance.git /opt/quantpulse
sudo chown -R ubuntu: /opt/quantpulse
git -C /opt/quantpulse checkout claude/keen-tesla-y7rion      # the branch the server should follow
bash /opt/quantpulse/deploy/bootstrap-oracle.sh
exit                                                          # log in again so Docker works without sudo
```

### 6. Tailscale (the private network to your phone)

```bash
ssh ubuntu@203.0.113.10
sudo tailscale up            # open the printed link on the PC, log in (Google/Microsoft/GitHub account)
```

On the iPhone: install **Tailscale** from the App Store and log in with the same account.

### 7. QuantPulse

```bash
cd /opt/quantpulse/deploy
./qp setup          # generates the database password and API token, asks for your Alpaca PAPER keys (not
                    # shown while typed) and a dashboard password; creates deploy/ops.env
./qp preflight      # must end in "PASS": paper only, secured, PostgreSQL, risk limits intact
./qp import-sqlite /path/to/quantpulse.db     # optional: the PC's history (copy it up with scp first)
./qp start          # builds the image on ARM (≈ 5–10 minutes the first time) and starts everything
./qp tailscale      # prints https://quantpulse.<your-tailnet>.ts.net — the dashboard, on the tailnet only
./qp install-timers # the watchdog, the CI-gated auto-update, the nightly backup, the weekly restore test
./qp status
```

### 8. Alerts (free, recommended)

In `deploy/.env` (then `./qp restart api`):

| Variable | Get it from | Purpose |
|---|---|---|
| `QP_ALERT_NTFY_URL` | the **ntfy** iPhone app: subscribe to a long random topic, e.g. `https://ntfy.sh/qp-7f3k9x2m4q` | every alert on your phone: the Brain (kill switch, blocked execution, anomalies, a supervisor takeover) and the server (a deploy, a rollback, a watchdog restart, a backup failure) |
| `QP_HEARTBEAT_URL` | <https://healthchecks.io>: a check with period 5 min, grace 10 min | alerts you when the server stops reporting healthy — or stops reporting at all (VM down, reclaimed, network lost) |

In `deploy/ops.env`: `QP_BACKUP_HEARTBEAT_URL` — a second healthchecks.io check (period 1 day, grace 2 hours) for
the nightly backup. Server alerts go to `QP_ALERT_NTFY_URL` unless `QP_OPS_NTFY_URL` names another topic.

### 9. Off-site backups (Object Storage)

1. Storage → Object Storage → **Create bucket**: name `quantpulse-backups`, Standard tier, **private** (the
   default), encryption: Oracle-managed keys.
2. The bucket → **Lifecycle policy rules** → create three rules (Action *Delete*):
   `daily/` older than 14 days, `weekly/` older than 90 days, `monthly/` older than 400 days. The server uploads
   each night's dump into one of the three (the 1st of a month → `monthly/`, a Sunday → `weekly/`, else
   `daily/`): about 14 + 13 + 13 generations, a few GB, within the 20 GB free.
3. The bucket → **Pre-authenticated requests** → Create: target **Bucket**, access type **Permit object writes**
   (write only — do *not* enable listing or reads), expiration e.g. one year ahead (set a calendar reminder to
   renew it). Copy the URL shown **once** (`https://objectstorage.<region>.oraclecloud.com/p/…/n/…/b/quantpulse-backups/o/`).
4. On the server: `nano /opt/quantpulse/deploy/ops.env` → `QP_BACKUP_PAR_URL=<that URL>`; save.
5. Test: `./qp backup nightly` (uploads), then `./qp restore-test`. The object appears in the bucket.

The server can only *add* objects: it cannot read, list, overwrite-and-verify, or delete them — a compromised
server cannot destroy the off-site history. Object Storage encrypts everything at rest; the dump contains the
Brain's history but **no credential** (PostgreSQL roles and passwords are never dumped — tested).

### 10. Automatic deploys

`deploy/ops.env` (created by `./qp setup`):

```
QP_DEPLOY_BRANCH=claude/keen-tesla-y7rion   # the branch this server follows
QP_AUTO_UPDATE=true
QP_AUTO_UPDATE_WINDOW=closed                # only outside 08:00–17:30 New York on weekdays; "any" for always
```

Every 10 minutes the server fetches the branch. When there is a new commit, it deploys only if all of these hold:

1. The new commit extends the deployed one (history not rewritten), and it has not failed a deploy before.
2. GitHub CI passed on **exactly that commit**: the workflow's push run completed with success, and each required
   job passed, `docker (arm64)` included. Pending, failed, skipped, missing or unreadable means **no deploy**.
3. It is outside market hours (default), no rollback pin is set, nobody ran `./qp down`, and the running version
   is healthy.

The deploy itself runs in two phases. **Before the switch**, the old version keeps running, and a failure here
changes nothing:
1. a backup;
2. `docker build` of exactly that commit from git (tagged `quantpulse:<commit>`);
3. the paper-only preflight on the new image.

**The switch** is one `docker compose up`:
1. The old API stops gracefully (it hands the Brain lease over).
2. The new one migrates the schema and starts.
3. Its first tick is the startup recovery (it reconciles with Alpaca and runs the execution audit before any order).

**Verification**: within 10 minutes the API must answer and the supervisor must be ticking. If not, the server
**rolls back by itself**:
1. The API stops.
2. The schema is downgraded by the new image (if that fails, the pre-deploy backup is restored).
3. The previous image and commit are restored.
4. The old version starts.

You get an alert either way. A commit that failed is never retried automatically.

`./qp update` does the same at once (inside market hours too; CI must still have passed). `./qp ci-gate` shows
what the gate would decide. `./qp rollback [COMMIT]` goes back by hand and pauses automatic deploys until the
next `./qp update`.

Optional, in GitHub (Settings → Branches → add a rule for `main`): require the five status checks above before
merging, so that nothing that failed CI (ARM64 included) can be merged either.

---

## Everyday use

| From the iPhone | Open the Tailscale app (connected), then Safari → `https://quantpulse.<tailnet>.ts.net` → sign in (it stays signed in for 30 days). **Home** shows the status, the paper account and the Brain; **STOP BRAIN TRADING** holds new Brain orders and cancels working ones. Add it to the home screen (Share → Add to Home Screen). |
|---|---|
| On the server | `./qp status` · `./qp logs` · `./qp stop-trading` · `./qp start-trading` · `journalctl -u 'quantpulse-*' -e` (timers) |

`./qp status` shows, in one screen:
- **the host**: CPU, memory, swap, disk, and the idle-reclamation check (memory and the 7-day CPU p95 against Oracle's 20%);
- **Docker and PostgreSQL**: container states, database size and schema;
- **the API and the Brain supervisor**: the watchdog verdict, the last tick, the last Brain cycle, the last delivered heartbeat;
- **backups**: the last backup (size, off-site object) and the last restore test;
- **the deployment**: the deployed commit and its CI result, the auto-update state, the watchdog's recent restarts;
- **the dashboard and the reader**: their tailnet addresses (or that the reader is off), and PUBLIC if either is
  ever exposed to the internet;
- the health of every part, and anything holding new Brain orders.

### Verify that it is paper only

```bash
./qp preflight                                  # "paper_setting", "paper_endpoint", "paper_key" … all PASS
grep -E '^QP_ALPACA_(PAPER|API_KEY_ID)=' .env | sed 's/=\(PK...\).*/=\1…/'   # QP_ALPACA_PAPER=true, key PK…
./qp status | grep -i paper                     # "(Alpaca PAPER only)"
```

The dashboard banner reads *ALPACA PAPER TRADING — SIMULATED MONEY ONLY*. In Alpaca's own website, orders
appear under the **Paper** account. Nothing in QuantPulse can point at a live account: there is no setting for
it, and the preflight refuses any variable that names Alpaca's live or broker API.

---

## Read-only access (`./qp reader`)

A way to watch QuantPulse from another device, script or assistant without giving it the dashboard password or
the API token. It is **off** until you turn it on.

```bash
./qp reader on     # makes the read-only key, restarts the API gracefully, publishes the reader on the tailnet
./qp reader key    # shows the key again
./qp reader off    # revokes the key at once and unpublishes the reader
```

`./qp reader on` and `off` restart the API the way a deploy does: the running tick finishes, and the Brain
reconciles before any order. Outside market hours is best.

**What it can do.** It can send GET requests to the monitoring pages, at most 60 a minute:
* the system: status, health, alerts, watchdog;
* the paper account: account, positions, orders, cycles, events, performance, risk;
* the Brain: status, positions, cycles, decisions and their audit trail, learning, research, reviews;
* the options book, market changes, the model registry, the prediction record and background jobs.

**What it cannot do.** It cannot send any other request: no order, cancel, control, kill switch, setting or
promotion, whatever the page. The forecasts, stock reports, quotes, option chains and quote streams are left
out, so it cannot slow the server or use up the market-data allowance. Two layers enforce this:
* The reader gateway (`quantpulse/reader.py`) is the only thing on the tailnet. It checks the key, the method,
  the page and the rate before it forwards anything. It holds only the read-only key.
* The API refuses the read-only key for anything but a GET to a monitoring page, and for WebSockets.

The key cannot change anything, but it shows your positions and history: keep it out of chats, emails and
GitHub.

**From a device on your tailnet:**

```bash
curl -H "X-API-Key: <the read-only key>" https://quantpulse.<tailnet>.ts.net:8443/api/v1/brain/status
```

**From a cloud machine** (for example an assistant's container). It has to join your tailnet, and should reach
the reader and nothing else:
1. In the Tailscale admin console → **Access controls**, keep tagged machines away from everything but the
   reader. Change the default rule's source from `"*"` to `"autogroup:member"` (your own devices), then add:
   ```json
   "tagOwners": { "tag:reader": ["autogroup:admin"] },
   "hosts":     { "quantpulse": "100.x.y.z" },
   "grants":    [ { "src": ["tag:reader"], "dst": ["quantpulse"], "ip": ["tcp:8443"] } ]
   ```
   `100.x.y.z` is the server's Tailscale address (`tailscale ip -4` on the server). In an older policy file
   that uses `"acls"`, the rule is `{"action": "accept", "src": ["tag:reader"], "dst": ["quantpulse:8443"]}`.
2. **Settings → Keys → Generate auth key**: ephemeral (the machine disappears when it stops), tagged
   `tag:reader`, with an expiry. Store it, and the read-only key, as that machine's secrets, never in a
   repository.
3. On the machine, Tailscale runs without changing its network (userspace mode), and requests go through its
   local proxy:
   ```bash
   tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --state=mem: &
   tailscale up --auth-key="$TS_AUTHKEY" --hostname=quantpulse-reader
   curl --socks5-hostname localhost:1055 -H "X-API-Key: $QP_READ_KEY" \
     https://quantpulse.<tailnet>.ts.net:8443/api/v1/brain/status
   ```

To take the access away: `./qp reader off` (the key stops working at once), revoke the auth key, and remove
the machine in the admin console if it is still listed.

---

## The watchdog

Every minute, `quantpulse-watchdog.timer` asks `GET /api/v1/system/watchdog`. The verdict is decided inside the
API (`src/quantpulse/services/watchdog.py`), from the supervisor's own timestamps:

| Verdict | Meaning | The watchdog |
|---|---|---|
| `ok`, `starting` | ticking, or the first tick is due | nothing |
| `stalled` | a tick has run longer than 20 min (market open) / 90 min (closed) — hung; or no tick was asked for in 10 min (the scheduler stopped); or the scheduler is not running | after 2 checks in a row: `docker compose restart api` (graceful), an alert |
| no answer | the API does not answer (connection refused, timeout, HTTP 5xx) for 3 minutes | a restart, an alert |
| `blocked` | fail-closed: startup recovery refuses to resume (Alpaca unreachable, the audit failed) | **nothing** — a restart would not fix it; the health alerts say why |
| `standby` | another process holds the supervisor lease | nothing (restarting this one cannot help) |
| `paused`, `not_applicable` | paused by a person, disabled, shutting down | nothing |

Each run also records CPU and memory for the idle check, and warns (at most every 6 hours) when a full day of
samples looks idle by Oracle's rule.

At most **3 restarts in 6 hours**. After that it alerts "Watchdog gave up: a person is needed" and does nothing
more. It leaves the API alone for 5 minutes after any start, while a deploy, backup or restore runs, and after
`./qp down`. A process so stuck that it ignores the stop is killed after 180 s; its lease then lapses within
3 minutes, and the new process recovers first, as after a crash.

---

## Shutdown and start-up behaviour

| Event | What happens |
|---|---|
| **Reboot** (a security update at 07:40 UTC, or `sudo reboot`) | Docker stops the containers gracefully (up to 180 s for the API: no new order, drain, final reconciliation, the lease handed over). At boot Docker starts them again (`restart: unless-stopped`); the API waits for PostgreSQL, the Brain recovers (reconcile, audit) before any order. The timers resume (the watchdog after 3 minutes). |
| **Oracle stops the VM** (maintenance, idle reclamation) or you stop it in the console | The same graceful stop when Oracle sends an ACPI shutdown; data stays on the boot volume. Start it in the console (Instances → quantpulse → Start): everything comes back as after a reboot. The public IP may change (Tailscale's name does not). |
| **A crash of the API** | Docker restarts it; recovery first. |
| **`./qp down`** | Everything stops and stays stopped (the watchdog and auto-update leave it alone) until `./qp start`. |
| **`./qp stop-trading`** | The Brain kill switch: no new Brain orders (persists across restarts); the Brain keeps analysing. |
| **Research jobs** (while the market is closed) | A stop or reboot stops the running research job and queues it again. After the start, the Brain picks it up while the market is still closed. Being interrupted costs a job nothing, and a job is never run in the session. |

---

## Recovery

**After a VM reboot or stop.** Nothing to do. Check:

```bash
./qp status        # everything "running", supervisor OK
```

If the supervisor says `blocked`, read the reason. Alpaca unreachable or a failed audit is fail-closed by design.
It resumes by itself once the cause clears.

**After a failed deploy.** The server has already rolled back and alerted you with the reason. The failed commit
is recorded and not retried. Fix it on GitHub; the next commit that passes CI deploys normally. To retry the
same commit after, e.g., a network failure during the build, run `./qp update <commit>`. `./qp status` shows
`phase rollback_unhealthy` if even the old version did not come back; then:

```bash
./qp logs api                      # why
./qp rollback <a commit that ran>  # or: ./qp restore deploy/backups/quantpulse-predeploy-….dump
```

**The watchdog gave up.** Run `./qp status` and `./qp logs api`. `./qp restart` once you know why. The count
resets after 6 hours.

**Restore PostgreSQL from a local backup** (`deploy/backups`: nightly and pre-deploy dumps, 14 days):

```bash
ls -1t deploy/backups | head
./qp restore deploy/backups/quantpulse-nightly-20261004-071503.dump     # asks: type RESTORE
```

It backs up the current database first, stops the API, restores, starts. The Brain reconciles with Alpaca before
any order: positions and working orders are re-read from Alpaca, never trusted from the backup.

**Restore from Object Storage** (the server's disk is lost, or the local dumps are):

1. Oracle console → the bucket → `daily/` (or `weekly/`, `monthly/`) → the newest object → **Download**.
2. From the PC: `scp quantpulse-nightly-….dump ubuntu@<ip>:/opt/quantpulse/deploy/backups/`
3. On the server: `./qp restore deploy/backups/quantpulse-nightly-….dump`

**The VM is gone** (terminated, account problem): create a new VM (steps 3–7), then restore from Object Storage
as above before `./qp start` sends anything. The Alpaca paper account is the source of truth for positions and
orders; the backup brings back the Brain's history and learning.

---

## Settings and secrets reference

### `deploy/.env` (the containers'; created by `./qp setup`, chmod 600)

| Variable | Value | Secret? |
|---|---|---|
| `POSTGRES_PASSWORD` | generated by `./qp setup` | yes |
| `QP_API_TOKEN` | generated (64 hex characters) | yes |
| `QP_API_READ_TOKEN` | empty (off); `./qp reader on` generates it (64 hex characters), `./qp reader off` empties it | yes (it can only read) |
| `QP_DASHBOARD_PASSWORD_HASH` | made by `./qp setup` / `./qp password` from your password | hash |
| `QP_ALPACA_API_KEY_ID`, `QP_ALPACA_API_SECRET_KEY` | your Alpaca **paper** keys (Alpaca → Paper Trading → API Keys; the id starts with `PK`) — typed into `./qp setup` | **yes** |
| `QP_ALPACA_PAPER` | `true` (compose forces it too) | no |
| `QP_ALPACA_STOCK_FEED` | `iex` (free; history comes from SIP 15 min delayed, free) | no |
| `QP_BRAIN_MODE`, `QP_BRAIN_SUPERVISOR_ENABLED`, `QP_ALPACA_TRADING_ENABLED`, `QP_TRADING_DRY_RUN` | `paper_execution`, `true`, `true`, `false` (the Brain trades the paper account) | no |
| `QP_BRAIN_KILL_SWITCH`, `QP_TRADING_KILL_SWITCH` | `false` (set `true` to stop from the server) | no |
| `QP_ALERT_NTFY_URL`, `QP_ALERT_WEBHOOK_URL`, `QP_HEARTBEAT_URL` | optional alert targets (see step 8) | yes (anyone with the URL can read/post) |
| `QP_PG_SHARED_BUFFERS`, `QP_PG_EFFECTIVE_CACHE` | optional; defaults `1GB`, `3GB` (for 6 GB) | no |

### `deploy/ops.env` (the server's own; created by `./qp setup`, chmod 600, never given to a container)

| Variable | Value | Secret? |
|---|---|---|
| `QP_DEPLOY_BRANCH` | the branch to follow (set by `./qp setup` from the checkout) | no |
| `QP_DEPLOY_REPO` | `owner/name` (empty: from the git remote) | no |
| `QP_AUTO_UPDATE`, `QP_AUTO_UPDATE_WINDOW` | `true`, `closed` | no |
| `QP_WATCHDOG` | `true` | no |
| `QP_DASHBOARD_ACCESS` | `tailscale` (the dashboard is reached through Tailscale only; `public` would allow the caddy profile) | no |
| `QP_BACKUP_PAR_URL` | the write-only pre-authenticated request (step 9) | **yes** |
| `QP_BACKUP_HEARTBEAT_URL` | healthchecks.io check for backups | yes |
| `QP_OPS_NTFY_URL` | optional separate topic for server alerts | yes |
| `GITHUB_TOKEN` | **only for a private repository**: fine-grained token, this repository only, *Contents: Read*, *Actions: Read*, *Metadata: Read* | **yes** |

Secrets you type **on the server** and nowhere else:
- the Alpaca paper key id and secret (into `./qp setup`);
- the dashboard password (into `./qp setup`);
- the Object Storage link (into `ops.env`);
- the ntfy and healthchecks URLs;
- a GitHub token, only if the repository is private.

Never paste them into GitHub, a chat, or a commit.

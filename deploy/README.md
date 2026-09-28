# QuantPulse 24/7 in the cloud

**Set it up once, turn your PC off, and QuantPulse keeps running.** The Brain keeps supervising the Alpaca
**paper** account in the cloud, and you can watch it (or stop it) from your iPhone.

```
 iPhone (Safari + Tailscale app)          Windows PC (only for setup and maintenance: ssh)
        │ private, encrypted (Tailscale)            │
        ▼                                           ▼
 ┌──────────────────────── one small Linux server (Docker) ─────────────────────────┐
 │  dashboard (password login) ──► api: QuantPulse API · Brain supervisor · jobs    │
 │                                   │   (starts only if the preflight passes)       │
 │                                   ├──► PostgreSQL (all Brain history, volume)    │
 │  backup (nightly pg_dump) ────────┘                                               │
 └────────────────────────────────────┬─────────────────────────────────────────────┘
                                       ▼
                     Alpaca PAPER API (https://paper-api.alpaca.markets) — simulated money only
```

| Service | What it does |
|---|---|
| `api` | The QuantPulse API, the Brain supervisor (one tick a minute), background jobs, health checks, alerts |
| `db` | PostgreSQL 16: cycles, agents, opinions, consensus, decisions, predictions, outcomes, reflections, performance, the execution ledger, positions and theses, evaluations, opportunities, events, the strategy shadow |
| `dashboard` | The Streamlit dashboard, behind a password; the **Remote** page is made for the phone |
| `backup` | A compressed database backup every night into `deploy/backups/` (14 days kept) |
| `caddy` | Optional (`--public`): HTTPS on your own domain name instead of Tailscale |

## Cost

| Option | Server | Monthly | Notes |
|---|---|---|---|
| **Recommended** | Hetzner Cloud **CX23** (2 vCPU, 4 GB RAM, 40 GB SSD), Germany or Finland | **≈ €6** (€5.49 + €0.50 IPv4) | Reliable, simple, billed by the hour; plenty for QuantPulse. If the stock model ever runs out of memory, resize to CX33 (8 GB, €8.49) in two clicks. |
| Cheapest | Oracle Cloud *Always Free* ARM instance (2 OCPU, 12 GB since June 2026) | €0 | Free, but capacity is often unavailable, and Oracle may reclaim instances that stay idle for a week — a trading server is idle most of the time. Fine for trying things out; not recommended for the 60-session experiment. |
| US-based alternatives | DigitalOcean / AWS Lightsail, 2 vCPU · 4 GB | ≈ $24 | Closer to the US markets (latency does not matter for 30-minute cycles). |
| Tailscale (phone ↔ server) | Personal plan | €0 | |
| ntfy (phone alerts), healthchecks.io (server-down alert) | Free plans | €0 | Optional |

Prices as published by the providers in September 2026 (Hetzner raised prices on 15 June 2026); check them when you
order. **Nothing is bought by QuantPulse or by these scripts** — you order the server yourself.

## Safety, in one list

* **Paper only.** The server refuses to start unless `./qp preflight` passes: `QP_ALPACA_PAPER=true`, no variable
  anywhere naming Alpaca's live or broker API, the Alpaca client verified on `https://paper-api.alpaca.markets`,
  a paper key (`PK…`), Alpaca's own data URL. There is no setting that points QuantPulse at a live account.
* **Same risk controls, not a second risk engine.** Every Brain order still goes Brain → planner → risk preview →
  execution gates → trading service → risk engine → order manager → Alpaca paper. In the cloud the preflight
  also refuses loss, position, order, spread, quote-age and data-quality limits looser than the shipped ones.
* **One supervisor.** A database lease lets only one server supervise the Brain and send orders; a second copy
  stands by. Orders from *another* QuantPulse installation on the same Alpaca account (the PC left running)
  turn the Brain kill switch on automatically.
* **Fail closed.** After any restart nothing is sent until reconciliation, positions, working orders and the
  startup safety checks pass. While the database or Alpaca is failing, or after a failed reconciliation, new
  Brain orders are held. Three rejected/failed/unknown Brain orders within an hour turn the kill switch on.
* **Locked down.** No port is open to the internet except SSH. The dashboard needs a password; the API needs a
  64-character token (the dashboard holds it; your browser never sees it). Secrets live only in
  `deploy/.env` on the server (git and Docker ignore it), and every log line masks them.
* **Two kill switches.** The dashboard's STOP BRAIN TRADING (persists across restarts) and
  `QP_BRAIN_KILL_SWITCH=true` in `deploy/.env` (released only on the server).

---

## One-time setup (from your Windows PC, about 30 minutes)

Everything below is typed into **PowerShell** on the PC (Start → type *PowerShell*) or, after `ssh`, into the
server. Replace `203.0.113.10` with your server's IP address.

### 1. An SSH key on the PC (once)

```powershell
ssh-keygen -t ed25519
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```
Press Enter three times at the questions. The second command prints your **public** key (safe to share).

### 2. Rent the server

1. Create an account at <https://www.hetzner.com/cloud> → *Add Server*.
2. Location: Nuremberg, Falkenstein or Helsinki · Image: **Ubuntu 24.04** · Type: **CX23** (shared vCPU, x86).
3. *SSH keys* → *Add SSH key* → paste the public key from step 1.
4. Create. Note the IPv4 address.

### 3. Log in and fetch QuantPulse

```powershell
ssh root@203.0.113.10
```
On the server — give it read-only access to your GitHub repository (a *deploy key*):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
```
On GitHub: the repository → *Settings* → *Deploy keys* → *Add deploy key* → paste → leave *Allow write access*
**unticked** → *Add key*. Then, on the server:
```bash
printf 'Host github.com\n  IdentityFile ~/.ssh/github_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
git clone git@github.com:kasmdkasdsad/Finance.git /opt/quantpulse
cd /opt/quantpulse && git checkout claude/keen-tesla-y7rion   # or main, once this is merged
```

### 4. Prepare the server (Docker, firewall, swap, updates, Tailscale)

```bash
sudo bash /opt/quantpulse/deploy/bootstrap.sh
sudo tailscale up
```
`tailscale up` prints a link: open it on the PC and sign in (Google, Microsoft or Apple account). That server is
now on your private *tailnet*.

### 5. Secrets and the preflight

```bash
cd /opt/quantpulse/deploy
./qp setup
./qp preflight
```
`setup` generates the database password and the API token, asks for your **Alpaca paper** key id and secret
(nothing is shown while you type) and a **dashboard password** (at least 12 characters), and builds the image.
`preflight` must end with `PASS`. A `FAIL` line says exactly what to fix (in `deploy/.env`: `nano .env`).

Optional, same file: `QP_ALERT_NTFY_URL` (phone alerts) and `QP_HEARTBEAT_URL` (server-down alerts) — see
*Alerts* below — and the `QP_SEC_USER_AGENT` line (your name and e-mail).

### 6. Bring the history over from the PC (optional but recommended)

**First stop QuantPulse on the PC** (the launcher's *Stop*), and keep its Brain off from now on: two
installations must never trade one Alpaca account (the cloud would notice and stop itself). Then, in PowerShell on the PC:
```powershell
scp "$env:USERPROFILE\Finance\data\quantpulse.db" root@203.0.113.10:/root/quantpulse.db
```
(adjust the path to where the Finance folder is on your PC). On the server:
```bash
cd /opt/quantpulse/deploy
./qp import-sqlite /root/quantpulse.db && rm /root/quantpulse.db
```
Every table is copied and its row count verified. It only works into an empty database, i.e. before the first
`./qp start`.

### 7. Start, and publish the dashboard to your phone

```bash
./qp start
./qp tailscale
```
`start` builds, runs the preflight, starts everything and prints the health of every part. `tailscale` prints the
dashboard's private address, like `https://quantpulse.tail1234.ts.net`. (The first time, Tailscale may print a link
asking to enable HTTPS certificates for your tailnet: open it, click *Enable*, and run `./qp tailscale` again.)
The address works only on devices signed in to your Tailscale account — nothing is open to the internet.

### 8. Turn the PC off

That's it. The server restarts QuantPulse by itself after a crash or a reboot; the Brain recovers (reconciles,
checks positions and working orders, runs its safety checks) before it sends anything.

On the PC, to keep QuantPulse from trading there by accident, set `QP_BRAIN_MODE=research_only` in the PC's
`.env` (or simply don't start it).

---

## On the iPhone

1. Install **Tailscale** from the App Store and sign in with the same account as in step 4.
2. Open Safari → the address from `./qp tailscale` → sign in with the dashboard password.
3. *Share* → **Add to Home Screen**: QuantPulse is now an icon on the phone.
4. The **Remote** page opens first: health of every part, **STOP BRAIN TRADING**, the supervisor's last and next
   cycle, P&L, positions and working orders, the last cycle's agents and consensus, opportunities, data-quality
   blocks, the 20/40/60-session progress and recent alerts. Every other page of the dashboard is in the menu.

**To stop the Brain from the phone:** Remote → **STOP BRAIN TRADING**. One tap: no new Brain orders, and its
working orders are canceled; positions stay. It survives restarts. To allow orders again: tick *I have looked*,
then *Allow Brain orders again*.

### Alerts (optional, free)

* **ntfy** — install *ntfy* from the App Store, *Subscribe to topic* → a long random name such as
  `qp-8f3kd92mxq71zp0a4c` (it acts as a password). Put `QP_ALERT_NTFY_URL=https://ntfy.sh/qp-8f3kd92mxq71zp0a4c`
  in `deploy/.env` and `./qp restart`. You get a push for: QuantPulse started, Brain stopped or waiting,
  reconciliation failed, an unexpected position, a kill switch turned on, repeated data-quality halts, Alpaca
  unreachable, the database failing, an execution anomaly — and when things recover.
* **healthchecks.io** — a free account → *Add check* (period 5 minutes, grace 10) → copy its ping URL into
  `QP_HEARTBEAT_URL`. If the whole server dies, healthchecks.io e-mails (or pushes) you.

---

## Everyday commands

From PowerShell on the PC, one line each (no need to log in first):

| What | Command |
|---|---|
| Status and health | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp status` |
| Follow the logs (Ctrl+C to stop) | `ssh -t root@203.0.113.10 /opt/quantpulse/deploy/qp logs` |
| Dashboard logs | `ssh -t root@203.0.113.10 /opt/quantpulse/deploy/qp logs dashboard` |
| Restart QuantPulse | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp restart` |
| **Stop Brain trading now** | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp stop-trading "reason"` |
| Allow Brain trading again | `ssh -t root@203.0.113.10 /opt/quantpulse/deploy/qp start-trading` |
| Update from GitHub | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp update` |
| Roll back to the previous version | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp rollback` |
| Back up the database now | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp backup` |
| Stop everything | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp down` |
| Start everything | `ssh root@203.0.113.10 /opt/quantpulse/deploy/qp start` |
| Change the dashboard password | `ssh -t root@203.0.113.10 /opt/quantpulse/deploy/qp password` |
| Copy a backup to the PC | `scp root@203.0.113.10:/opt/quantpulse/deploy/backups/<file>.dump .` |

Tip: add a shortcut to `C:\Users\<you>\.ssh\config` so `ssh quantpulse` works:
```
Host quantpulse
  HostName 203.0.113.10
  User root
```

`update` backs up the database, pulls the latest code of the checked-out branch, rebuilds, runs the preflight
and restarts; the Brain recovers before it sends anything. `rollback` backs up, rolls the database schema back
to what the previous version expects, and redeploys that version (`qp update` later returns to the latest).

Changing a setting: `ssh -t root@203.0.113.10 nano /opt/quantpulse/deploy/.env`, then `qp restart`. The preflight
checks the new settings; the API does not start on an unsafe one.

**Adding real-time SIP data later:** subscribe at Alpaca, set `QP_ALPACA_STOCK_FEED=sip` in `deploy/.env`,
`qp restart`. The quote-age and spread limits stay exactly as they are.

## What happens when …

| Event | What QuantPulse does |
|---|---|
| The server reboots | Docker starts at boot; every container restarts; the Brain recovers first, then resumes |
| The API crashes | Docker restarts it; the lease lapses (3 min) or is released; recovery before any order |
| The database restarts | Health: *database failing* → Brain orders held; an alert; everything resumes by itself |
| The network or Alpaca is down | Reconciliation fails → no Brain order until one succeeds; an alert; retried every minute |
| A deployment restarts QuantPulse | A clean stop releases the lease; the new version recovers, then resumes |
| A second copy starts on the same database | It stands by (one supervisor lease); if the first dies, it takes over after recovery |
| The PC's QuantPulse trades the same Alpaca account | The cloud sees QuantPulse orders it did not place → Brain kill switch ON + alert |
| Brain orders keep being rejected | 3 within an hour → Brain kill switch ON + alert |
| Market data is stale | The Brain's quote-age and spread gates hold entries (as on the PC); repeated halts → alert |

## Public HTTPS instead of Tailscale (optional)

If you would rather open the dashboard without the Tailscale app: get a domain name, point an `A` record (e.g.
`quantpulse.example.com`) at the server, put `QP_PUBLIC_HOSTNAME=quantpulse.example.com` in `deploy/.env`, and:
```bash
sudo ufw allow 80,443/tcp
./qp start --public
```
Caddy obtains a free certificate. Only the dashboard (with its password and lock-out) is published; the API never is.
Tailscale remains the safer choice: nothing is reachable from the internet at all.

## Troubleshooting

* **`./qp start` says preflight failed** — read the `FAIL` lines; fix `deploy/.env`; `./qp preflight` again.
* **Health shows `supervisor: waiting`** — startup recovery has not passed yet (usually Alpaca unreachable or a
  failed safety check); the detail says which. It retries every minute and sends nothing meanwhile.
* **Health shows `standby`** — another QuantPulse process on this database holds the supervisor lease.
* **Brain kill switch ON with "automatic: …"** — read the reason (Remote page); fix the cause; then release it.
* **The stock model runs out of memory** — `./qp logs` shows the api restarting; resize the server to CX33.
* **Forgot the dashboard password** — `./qp password`.

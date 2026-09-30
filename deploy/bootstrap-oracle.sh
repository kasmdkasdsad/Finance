#!/usr/bin/env bash
# One-time preparation of an Oracle Cloud Always Free Ampere A1 server (VM.Standard.A1.Flex, Ubuntu 24.04, ARM64)
# for QuantPulse. Run it once as the default "ubuntu" user, from the cloned repository (it is idempotent: running
# it again changes nothing that is already right):
#
#   sudo git clone https://github.com/<owner>/<repo>.git /opt/quantpulse && sudo chown -R ubuntu: /opt/quantpulse
#   git -C /opt/quantpulse checkout <branch>
#   bash /opt/quantpulse/deploy/bootstrap-oracle.sh            # add --no-tailscale to skip Tailscale
#
# What it does (see deploy/ORACLE.md):
#   * packages: git, python3, jq, openssl, chrony (the Brain refuses to trade on a skewed clock)
#   * Docker Engine + Compose for arm64 from Docker's repository, started at boot; graceful container stops at
#     shutdown (the API needs up to 3 minutes to hand the Brain over cleanly)
#   * 2 GB of swap and memory settings for a 6 GB server (low swappiness: memory stays in RAM)
#   * the firewall: Oracle's own iptables rules are KEPT (SSH only) — no ufw (it would fight Docker and Oracle's
#     rules); nothing of QuantPulse listens publicly (the API and dashboard are bound to 127.0.0.1)
#   * SSH: keys only, no root login
#   * automatic security updates, with a reboot at 07:40 UTC when one is needed (outside US market hours)
#   * journald capped at 500 MB; UTC time zone
#   * Tailscale (free for personal use) for the dashboard on your phone
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -ne 0 ]] || die "run as the ubuntu user (the script uses sudo where it must), not as root"
command -v sudo >/dev/null || die "sudo is required"
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "this script is for Ubuntu (24.04); found ${ID:-unknown}"
[[ "$(dpkg --print-architecture)" == "arm64" ]] || echo "note: not an ARM server ($(dpkg --print-architecture)); the script works, but Oracle's Always Free A1 is arm64"
WITH_TAILSCALE=1
[[ "${1:-}" == "--no-tailscale" ]] && WITH_TAILSCALE=0
USER_NAME="$(id -un)"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

say "packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg git jq openssl python3 chrony unattended-upgrades apt-listchanges

say "Docker Engine + Compose (arm64, official repository), enabled at boot"
if ! command -v docker >/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
# logs rotate; containers get their own stop timeout (180 s for the API) when the server shuts down
sudo mkdir -p /etc/docker /etc/systemd/system/docker.service.d
DAEMON='{
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "5"},
  "shutdown-timeout": 180
}'
UNIT='[Service]
# a reboot waits for the API'"'"'s graceful stop (drain, final reconciliation, the Brain lease handed over)
TimeoutStopSec=300'
changed=0
if [[ "$(sudo cat /etc/docker/daemon.json 2>/dev/null)" != "$DAEMON" ]]; then
  printf '%s\n' "$DAEMON" | sudo tee /etc/docker/daemon.json >/dev/null; changed=1
fi
if [[ "$(sudo cat /etc/systemd/system/docker.service.d/quantpulse.conf 2>/dev/null)" != "$UNIT" ]]; then
  printf '%s\n' "$UNIT" | sudo tee /etc/systemd/system/docker.service.d/quantpulse.conf >/dev/null; changed=1
fi
sudo systemctl daemon-reload
sudo systemctl enable --now containerd docker
# only when the settings changed (a restart of Docker stops and restarts QuantPulse, gracefully)
[[ "$changed" == "0" ]] || sudo systemctl restart docker
sudo usermod -aG docker "$USER_NAME"

say "swap (2 GB) and memory settings"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
sudo tee /etc/sysctl.d/60-quantpulse.conf >/dev/null <<'SYSCTL'
# QuantPulse on a 6 GB server: keep the working set in RAM (swap is a safety net, not working space)
vm.swappiness = 10
vm.vfs_cache_pressure = 50
SYSCTL
sudo sysctl --system >/dev/null

say "firewall: Oracle's rules kept (SSH only); nothing of QuantPulse is public"
if command -v ufw >/dev/null && sudo ufw status | grep -q "Status: active"; then
  echo "note: ufw is active; QuantPulse needs no inbound port — leave only SSH open"
fi
if sudo iptables -S INPUT 2>/dev/null | grep -Eq -- '--dport (8000|8501) .*ACCEPT'; then
  die "an iptables rule accepts port 8000 or 8501 from outside: remove it (the API and dashboard stay private)"
fi
sudo iptables -S INPUT 2>/dev/null | sed 's/^/   /' || true

say "SSH: keys only, no root login"
sudo tee /etc/ssh/sshd_config.d/60-quantpulse.conf >/dev/null <<'SSHD'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
SSHD
sudo sshd -t && (sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd 2>/dev/null || true)

say "automatic security updates (reboot at 07:40 UTC when one needs it)"
sudo tee /etc/apt/apt.conf.d/20auto-upgrades >/dev/null <<'APT'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APT
sudo tee /etc/apt/apt.conf.d/52quantpulse-unattended >/dev/null <<'APT'
// after the nightly backup (07:15 UTC) and outside US market hours; Docker restarts QuantPulse after the reboot
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "07:40";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
APT
sudo systemctl enable --now unattended-upgrades

say "clock (UTC, synchronised by chrony) and the journal (500 MB at most)"
sudo timedatectl set-timezone UTC
sudo systemctl enable --now chrony
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=500M\n' | sudo tee /etc/systemd/journald.conf.d/60-quantpulse.conf >/dev/null
sudo systemctl restart systemd-journald

if [[ "$WITH_TAILSCALE" == "1" ]] && ! command -v tailscale >/dev/null; then
  say "Tailscale (the dashboard on your phone, privately; free for personal use)"
  curl -fsSL https://tailscale.com/install.sh | sh
fi

say "QuantPulse folders"
mkdir -p "$HERE/state" "$HERE/backups"
chmod 700 "$HERE/state"

cat <<EOF

Done. Log out and back in once (so "$USER_NAME" can use Docker without sudo), then:

  sudo tailscale up                       # once: open the link it prints, log in (skip with --no-tailscale)
  cd $HERE
  ./qp setup                              # secrets, your Alpaca PAPER keys (asked, never shown), a password
  ./qp preflight                          # paper only, secured, PostgreSQL, risk limits intact
  ./qp start                              # build (ARM), start; comes back by itself after every reboot
  ./qp tailscale                          # the dashboard at https://<this-server>.<tailnet>.ts.net
  ./qp install-timers                     # watchdog, CI-gated auto-update, nightly backup, restore test
  ./qp status

Off-site backups and alerts: edit deploy/ops.env (see deploy/ORACLE.md, "Backups").
EOF

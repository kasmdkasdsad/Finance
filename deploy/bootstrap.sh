#!/usr/bin/env bash
# One-time preparation of a fresh Ubuntu 24.04 server for QuantPulse (run as root, once):
#   Docker Engine + Compose (starts at boot, so QuantPulse comes back after every reboot), a firewall that lets
#   in SSH only, automatic security updates, 2 GB of swap (headroom for the stock model on a 4 GB server),
#   time synchronisation (the Brain refuses to trade on a skewed clock), and — optionally — Tailscale.
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/<repo>/<branch>/deploy/bootstrap.sh | sudo bash
#   (or copy this file to the server and run: sudo bash bootstrap.sh)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root (sudo bash bootstrap.sh)"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "== packages"
apt-get update -y
apt-get install -y ca-certificates curl git openssl python3 ufw unattended-upgrades

echo "== Docker (official repository) — enabled at boot"
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

echo "== firewall: SSH only (the dashboard is reached through Tailscale, or 80/443 with --public)"
ufw allow OpenSSH
ufw --force enable

echo "== automatic security updates"
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "== swap (2 GB)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== clock: UTC, synchronised"
timedatectl set-timezone UTC
timedatectl set-ntp true || true

if [[ "${WITH_TAILSCALE:-1}" == "1" ]] && ! command -v tailscale >/dev/null; then
  echo "== Tailscale (private network to your phone; free for personal use)"
  curl -fsSL https://tailscale.com/install.sh | sh
  echo "   next: sudo tailscale up   (log in once in the browser link it prints)"
fi

echo
echo "Done. Next, as root on this server:"
echo "  git clone <your repository URL> /opt/quantpulse && cd /opt/quantpulse && git checkout <branch>"
echo "  cd deploy && ./qp setup && ./qp preflight && ./qp start && ./qp tailscale"

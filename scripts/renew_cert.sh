#!/bin/sh
# Renew the Let's Encrypt certificate of the Oasis (HTTP-01 via the proxy's
# webroot) and reload nginx when it changed. Run daily from the nomad user's
# crontab:  17 3 * * * /home/nomad/nomad-oasis/scripts/renew_cert.sh
#
# First issuance (proxy must not be running, port 80 free):
#   docker run --rm -p 80:80 -v $PWD/certbot/conf:/etc/letsencrypt \
#     certbot/certbot certonly --standalone -d vm193-209.its.uni-kassel.de \
#     --agree-tos --register-unsafely-without-email
set -eu
DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG="$DIR/certbot/renew.log"
mkdir -p "$DIR/certbot/www"
{
  echo "== $(date -Is)"
  docker run --rm \
    -v "$DIR/certbot/conf:/etc/letsencrypt" \
    -v "$DIR/certbot/www:/var/www/certbot" \
    certbot/certbot renew --webroot -w /var/www/certbot -q \
    --deploy-hook "touch /etc/letsencrypt/.renewed"
  if [ -e "$DIR/certbot/conf/.renewed" ]; then
    docker exec nomad_oasis_proxy nginx -s reload
    rm -f "$DIR/certbot/conf/.renewed"
    echo "renewed, nginx reloaded"
  fi
} >> "$LOG" 2>&1

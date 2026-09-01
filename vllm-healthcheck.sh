#!/usr/bin/env bash
# /usr/local/bin/vllm-healthcheck.sh
set -euo pipefail

# Если сервис не активен — значит его остановили осознанно (вручную,
# через деплой, maintenance-window и т.д.). Ничего не делаем.
if ! systemctl is-active --quiet vllm.service; then
    exit 0
fi

status=$(docker inspect --format='{{.State.Health.Status}}' vllm 2>/dev/null || echo missing)

case "$status" in
  healthy)  exit 0 ;;
  starting) exit 0 ;;
  unhealthy)
    echo "vllm unhealthy, restarting"
    systemctl restart vllm.service
    exit 1
    ;;
  missing)
    # Сервис active, но контейнера нет — гонка (только что стартовал/
    # рестартует) или реальная поломка. Не спамим restart, просто
    # репортим failure — таймер сам перезапустит проверку через 30s,
    # а systemd Restart=always на vllm.service уже разрулит сам контейнер.
    echo "vllm.service is active but container is missing"
    exit 1
    ;;
  *)
    echo "vllm unexpected state=$status"
    exit 1
    ;;
esac

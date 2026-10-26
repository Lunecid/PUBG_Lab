"""공통 설정 및 DB 연결."""
import os
import time
import logging
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

load_dotenv()

# ── 로깅 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pubg")

# ── PUBG API ──
API_KEY = os.getenv("PUBG_API_KEY")
SHARD = os.getenv("PUBG_SHARD", "steam")
BASE_URL = f"https://api.pubg.com/shards/{SHARD}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/vnd.api+json",
}

# ── Rate Limit (공식 기준) ──
# /samples, /players, /seasons 등: 10 req/min → 6초 간격
# /matches, telemetry CDN: 무제한 → 0.3초 간격 (서버 예의)
RATE_LIMIT_INTERVAL = 6.0   # 일반 엔드포인트
MATCH_INTERVAL = 0.3        # /matches (무제한이지만 예의)
TELEMETRY_INTERVAL = 0.2    # CDN (무제한)

# ── PostgreSQL ──
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DATABASE", "pubg_survival"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD"),
}
SCHEMA = os.getenv("PG_SCHEMA", "pubg")


def get_conn():
    """PostgreSQL 연결 반환."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def api_get(url, params=None, is_match=False, is_telemetry=False):
    """PUBG API 호출 + rate limit 준수 + api_rate_log 기록.

    Returns:
        requests.Response 또는 None (실패 시)
    """
    start = time.time()

    try:
        if is_telemetry:
            # 텔레메트리 CDN은 API 키 불필요
            resp = requests.get(url, timeout=60)
        else:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        elapsed_ms = int((time.time() - start) * 1000)

        # rate limit 헤더 파싱
        rate_remaining = resp.headers.get("X-RateLimit-Remaining")
        rate_reset = resp.headers.get("X-RateLimit-Reset")

        # DB에 호출 기록
        _log_api_call(
            endpoint=_classify_endpoint(url),
            status_code=resp.status_code,
            rate_remaining=int(rate_remaining) if rate_remaining else None,
            rate_reset=datetime.fromtimestamp(int(rate_reset), tz=timezone.utc) if rate_reset else None,
            response_ms=elapsed_ms,
        )

        # 429 처리
        if resp.status_code == 429:
            wait = 60  # 기본 1분 대기
            if rate_reset:
                wait = max(int(rate_reset) - int(time.time()), 1)
            log.warning(f"Rate limited! {wait}초 대기...")
            time.sleep(wait)
            return api_get(url, params, is_match, is_telemetry)

        if resp.status_code != 200:
            log.error(f"HTTP {resp.status_code}: {url}")
            return None

        # 호출 간격 준수
        if is_telemetry:
            time.sleep(TELEMETRY_INTERVAL)
        elif is_match:
            time.sleep(MATCH_INTERVAL)
        else:
            time.sleep(RATE_LIMIT_INTERVAL)

        return resp

    except requests.exceptions.RequestException as e:
        log.error(f"요청 실패: {e}")
        return None


def _classify_endpoint(url):
    if "telemetry" in url or "cdn" in url:
        return "telemetry"
    if "/matches/" in url:
        return "matches"
    if "/samples" in url:
        return "samples"
    return "other"


def _log_api_call(endpoint, status_code, rate_remaining, rate_reset, response_ms):
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {SCHEMA}.api_rate_log
                    (endpoint, status_code, rate_remaining, rate_reset, response_ms)
                VALUES (%s, %s, %s, %s, %s)
            """, (endpoint, status_code, rate_remaining, rate_reset, response_ms))
        conn.commit()
        conn.close()
    except Exception:
        pass  # 로깅 실패가 수집을 막으면 안 됨

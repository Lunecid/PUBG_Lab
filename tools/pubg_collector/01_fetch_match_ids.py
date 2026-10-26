"""01 — /samples에서 match_id 수집 → collection_log에 저장.

사용법:
    python 01_fetch_match_ids.py
"""
from datetime import datetime, timedelta, timezone
from config import log, BASE_URL, SCHEMA, api_get, get_conn


def fetch_sample_match_ids():
    """Samples API에서 매치 ID 목록 가져오기."""
    # 24시간 전 시점 기준 샘플 요청
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{BASE_URL}/samples"
    params = {"filter[createdAt-start]": since}

    log.info(f"Samples 요청: {since} 이후")
    resp = api_get(url, params=params)

    if not resp:
        log.error("Samples API 실패")
        return []

    data = resp.json()
    relationships = data.get("data", {}).get("relationships", {}).get("matches", {})
    match_refs = relationships.get("data", [])
    match_ids = [m["id"] for m in match_refs]

    log.info(f"샘플에서 {len(match_ids)}개 매치 발견")
    return match_ids


def save_to_collection_log(match_ids):
    """신규 match_id만 collection_log에 삽입."""
    if not match_ids:
        return 0

    conn = get_conn()
    inserted = 0

    try:
        with conn.cursor() as cur:
            for mid in match_ids:
                cur.execute(f"""
                    INSERT INTO {SCHEMA}.collection_log (match_id)
                    VALUES (%s)
                    ON CONFLICT (match_id) DO NOTHING
                """, (mid,))
                inserted += cur.rowcount
        conn.commit()
        log.info(f"collection_log에 {inserted}개 신규 삽입 (중복 제외)")
    except Exception as e:
        conn.rollback()
        log.error(f"DB 저장 실패: {e}")
    finally:
        conn.close()

    return inserted


def main():
    log.info("=" * 50)
    log.info("01 — Match ID 수집 시작")
    log.info("=" * 50)

    match_ids = fetch_sample_match_ids()
    inserted = save_to_collection_log(match_ids)

    # 현재 상태 출력
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {SCHEMA}.v_collection_status")
        row = cur.fetchone()
        if row:
            log.info(
                f"현황: 총 {row[0]}건 발견, "
                f"매치 {row[1]}건 수집, "
                f"텔레메트리 {row[2]}건 수집, "
                f"에러 {row[3]}건"
            )
    conn.close()

    log.info("01 — 완료")


if __name__ == "__main__":
    main()

"""02 — /matches에서 매치 상세 수집 → matches, rosters, participants 적재.

사용법:
    python 02_fetch_matches.py            # collection_log 미처리 건 전부
    python 02_fetch_matches.py --limit 50 # 최대 50건만
"""
import argparse
from datetime import datetime, timezone

from config import log, BASE_URL, SCHEMA, api_get, get_conn


def get_pending_match_ids(limit=None):
    """collection_log에서 match_fetched=FALSE인 ID 가져오기."""
    conn = get_conn()
    with conn.cursor() as cur:
        sql = f"""
            SELECT match_id FROM {SCHEMA}.collection_log
            WHERE match_fetched = FALSE AND retry_count < 3
            ORDER BY discovered_at
        """
        if limit:
            sql += f" LIMIT {limit}"
        cur.execute(sql)
        ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return ids


def fetch_and_save_match(match_id):
    """단일 매치 데이터 가져와서 DB 적재."""
    url = f"{BASE_URL}/matches/{match_id}"
    resp = api_get(url, is_match=True)

    if not resp:
        _mark_error(match_id, "API 응답 없음")
        return False

    data = resp.json()
    conn = get_conn()

    try:
        with conn.cursor() as cur:
            # ── Match 저장 ──
            attrs = data["data"]["attributes"]
            relationships = data["data"]["relationships"]

            # 텔레메트리 URL 추출
            telemetry_url = None
            included = data.get("included", [])
            for obj in included:
                if obj["type"] == "asset" and obj["attributes"].get("name") == "telemetry":
                    telemetry_url = obj["attributes"]["URL"]
                    break

            cur.execute(f"""
                INSERT INTO {SCHEMA}.matches
                    (match_id, created_at, duration, game_mode, map_name,
                     is_custom, match_type, season_state, shard_id, title_id, telemetry_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING
            """, (
                match_id,
                attrs["createdAt"],
                attrs["duration"],
                attrs.get("gameMode", "unknown"),
                attrs.get("mapName", "unknown"),
                attrs.get("isCustomMatch", False),
                attrs.get("matchType"),
                attrs.get("seasonState"),
                attrs.get("shardId", "steam"),
                attrs.get("titleId"),
                telemetry_url,
            ))

            # ── included 배열에서 roster/participant 분리 ──
            rosters = {}   # roster_id → roster obj
            participants = []

            for obj in included:
                if obj["type"] == "roster":
                    rosters[obj["id"]] = obj
                elif obj["type"] == "participant":
                    participants.append(obj)

            # ── Roster 저장 ──
            # roster → participant 매핑 구성
            roster_participant_map = {}  # roster_id → [participant_id, ...]
            for rid, roster in rosters.items():
                r_attrs = roster.get("attributes", {})
                r_stats = r_attrs.get("stats", {})
                won_str = r_attrs.get("won", "false")
                won = won_str == "true" if isinstance(won_str, str) else bool(won_str)

                cur.execute(f"""
                    INSERT INTO {SCHEMA}.rosters (roster_id, match_id, rank, team_id, won)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (roster_id) DO NOTHING
                """, (
                    rid,
                    match_id,
                    r_stats.get("rank"),
                    r_stats.get("teamId"),
                    won,
                ))

                # roster → participant 관계
                p_refs = roster.get("relationships", {}).get("participants", {}).get("data", [])
                for p_ref in p_refs:
                    roster_participant_map[p_ref["id"]] = rid

            # ── Participant 저장 ──
            for p in participants:
                p_id = p["id"]
                p_attrs = p.get("attributes", {})
                s = p_attrs.get("stats", {})

                cur.execute(f"""
                    INSERT INTO {SCHEMA}.participants
                        (participant_id, match_id, roster_id, player_id, player_name,
                         kills, assists, dbnos, damage_dealt, headshot_kills,
                         longest_kill, most_damage, kill_place, kill_streaks,
                         road_kills, team_kills, vehicle_destroys,
                         death_type, time_survived, win_place,
                         walk_distance, ride_distance, swim_distance,
                         boosts, heals, weapons_acquired, revives)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (participant_id) DO NOTHING
                """, (
                    p_id,
                    match_id,
                    roster_participant_map.get(p_id),
                    s.get("playerId"),
                    s.get("name"),
                    s.get("kills", 0),
                    s.get("assists", 0),
                    s.get("DBNOs", 0),
                    s.get("damageDealt", 0),
                    s.get("headshotKills", 0),
                    s.get("longestKill", 0),
                    s.get("mostDamage", 0),
                    s.get("killPlace"),
                    s.get("killStreaks", 0),
                    s.get("roadKills", 0),
                    s.get("teamKills", 0),
                    s.get("vehicleDestroys", 0),
                    s.get("deathType"),
                    s.get("timeSurvived", 0),
                    s.get("winPlace"),
                    s.get("walkDistance", 0),
                    s.get("rideDistance", 0),
                    s.get("swimDistance", 0),
                    s.get("boosts", 0),
                    s.get("heals", 0),
                    s.get("weaponsAcquired", 0),
                    s.get("revives", 0),
                ))

            # ── collection_log 업데이트 ──
            cur.execute(f"""
                UPDATE {SCHEMA}.collection_log
                SET match_fetched = TRUE,
                    match_fetched_at = NOW(),
                    telemetry_url = %s
                WHERE match_id = %s
            """, (telemetry_url, match_id))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        log.error(f"매치 {match_id} 저장 실패: {e}")
        _mark_error(match_id, str(e))
        return False
    finally:
        conn.close()


def _mark_error(match_id, msg):
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {SCHEMA}.collection_log
                SET error_message = %s, retry_count = retry_count + 1
                WHERE match_id = %s
            """, (msg[:500], match_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="최대 처리 건수")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("02 — 매치 상세 수집 시작")
    log.info("=" * 50)

    pending = get_pending_match_ids(args.limit)
    log.info(f"미처리 매치: {len(pending)}건")

    success = 0
    fail = 0

    for i, mid in enumerate(pending, 1):
        log.info(f"[{i}/{len(pending)}] {mid}")
        if fetch_and_save_match(mid):
            success += 1
        else:
            fail += 1

    log.info(f"완료: 성공 {success}, 실패 {fail}")

    # 현황 출력
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {SCHEMA}.v_collection_status")
        row = cur.fetchone()
        if row:
            log.info(
                f"현황: 총 {row[0]}건, "
                f"매치 {row[1]}건, "
                f"텔레메트리 {row[2]}건, "
                f"에러 {row[3]}건"
            )
    conn.close()


if __name__ == "__main__":
    main()

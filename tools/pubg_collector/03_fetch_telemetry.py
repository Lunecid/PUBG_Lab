"""03 — 텔레메트리 JSON 다운로드 → telem_* 테이블 적재.

사용법:
    python 03_fetch_telemetry.py            # 미처리 건 전부
    python 03_fetch_telemetry.py --limit 10 # 최대 10건만
"""
import argparse
import gzip
import io
import json

from config import log, SCHEMA, api_get, get_conn


# ── 아이템 이벤트 → event_type 매핑 ──
ITEM_PICKUP_TYPES = {
    "LogItemPickup":                    "pickup",
    "LogItemPickupFromCarepackage":     "pickup_carepackage",
    "LogItemPickupFromCustomPackage":   "pickup_custom",
    "LogItemPickupFromLootbox":         "pickup_lootbox",
    "LogItemPickupFromVehicleTrunk":    "pickup_vehicle_trunk",
    "LogItemDrop":                      "drop",
}


def get_pending_telemetry():
    """match_fetched=TRUE, telemetry_fetched=FALSE인 건."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT match_id, telemetry_url
            FROM {SCHEMA}.collection_log
            WHERE match_fetched = TRUE
              AND telemetry_fetched = FALSE
              AND telemetry_url IS NOT NULL
              AND retry_count < 3
            ORDER BY discovered_at
        """)
        rows = cur.fetchall()
    conn.close()
    return rows


def fetch_telemetry_json(url):
    """텔레메트리 CDN에서 JSON 다운로드 (gzip 자동 해제)."""
    resp = api_get(url, is_telemetry=True)
    if not resp:
        return None

    content_type = resp.headers.get("Content-Encoding", "")
    if content_type == "gzip":
        buf = io.BytesIO(resp.content)
        with gzip.GzipFile(fileobj=buf) as f:
            return resp.json()

    return resp.json()


def _extract_character(ev):
    """이벤트에서 character 필드 공통 추출."""
    c = ev.get("character", {}) or {}
    loc = c.get("location", {}) or {}
    return (
        c.get("accountId"),
        c.get("name"),
        c.get("teamId"),
        loc.get("x"), loc.get("y"), loc.get("z"),
    )


def _extract_item(ev, key="item"):
    """이벤트에서 item 필드 공통 추출."""
    item = ev.get(key, {}) or {}
    attached = item.get("attachedItems") or []
    return (
        item.get("itemId"),
        item.get("category"),
        item.get("subCategory"),
        item.get("stackCount"),
        attached if attached else None,
    )


def parse_and_save(match_id, events):
    """텔레메트리 이벤트 배열을 파싱해서 각 테이블에 적재."""
    conn = get_conn()

    # 기존 이벤트 버퍼
    positions = []
    game_states = []
    kills = []
    groggy = []
    damage = []
    phase_changes = []
    match_start = None
    match_end = None
    parachute_landings = []

    # 신규 아이템 이벤트 버퍼
    item_equips = []
    item_pickups = []
    item_uses = []

    for ev in events:
        t = ev.get("_T")
        ts = ev.get("_D")
        is_game = ev.get("common", {}).get("isGame")

        # ── 기존 이벤트 ──

        if t == "LogPlayerPosition":
            c = ev.get("character", {})
            loc = c.get("location", {})
            v = ev.get("vehicle", {})
            positions.append((
                match_id, ts, is_game,
                ev.get("elapsedTime"),
                ev.get("numAlivePlayers"),
                c.get("accountId"),
                c.get("name"),
                c.get("teamId"),
                c.get("health"),
                loc.get("x"), loc.get("y"), loc.get("z"),
                v.get("vehicleType") if v else None,
                v.get("vehicleId") if v else None,
                v.get("velocity") if v else None,
            ))

        elif t == "LogGameStatePeriodic":
            gs = ev.get("gameState", {})
            sz = gs.get("safetyZonePosition", {})
            pz = gs.get("poisonGasWarningPosition", {})
            rz = gs.get("redZonePosition", {})
            bz = gs.get("blackZonePosition", {})
            game_states.append((
                match_id, ts, is_game,
                gs.get("elapsedTime"),
                gs.get("numAliveTeams"),
                gs.get("numAlivePlayers"),
                gs.get("numJoinPlayers"),
                gs.get("numStartPlayers"),
                sz.get("x"), sz.get("y"), sz.get("z"),
                gs.get("safetyZoneRadius"),
                pz.get("x"), pz.get("y"), pz.get("z"),
                gs.get("poisonGasWarningRadius"),
                rz.get("x"), rz.get("y"), rz.get("z"),
                gs.get("redZoneRadius"),
                bz.get("x"), bz.get("y"), bz.get("z"),
                gs.get("blackZoneRadius"),
            ))

        elif t == "LogPlayerKillV2":
            killer = ev.get("killer", {}) or {}
            victim = ev.get("victim", {}) or {}
            dbno = ev.get("dBNOMaker", {}) or {}
            finisher = ev.get("finisher", {}) or {}
            di = ev.get("damageInfo", {}) or {}
            kl = killer.get("location", {}) or {}
            vl = victim.get("location", {}) or {}
            vr = ev.get("victimGameResult", {}) or {}
            assists = ev.get("assists_AccountId", []) or []
            kills.append((
                match_id, ts, is_game,
                killer.get("accountId"), killer.get("name"),
                kl.get("x"), kl.get("y"), kl.get("z"),
                victim.get("accountId"), victim.get("name"),
                vl.get("x"), vl.get("y"), vl.get("z"),
                dbno.get("accountId"), dbno.get("name"),
                finisher.get("accountId"), finisher.get("name"),
                di.get("damageReason"), di.get("damageTypeCategory"),
                di.get("damageCauserName"), di.get("distance"),
                ev.get("isSuicide", False),
                len(assists),
                vr.get("rank"), vr.get("teamId"),
            ))

        elif t == "LogPlayerMakeGroggy":
            attacker = ev.get("attacker", {}) or {}
            victim = ev.get("victim", {}) or {}
            di = ev.get("damageInfo", {}) or {}
            al = attacker.get("location", {}) or {}
            vl = victim.get("location", {}) or {}
            groggy.append((
                match_id, ts, is_game,
                attacker.get("accountId"), attacker.get("name"),
                al.get("x"), al.get("y"), al.get("z"),
                victim.get("accountId"), victim.get("name"),
                vl.get("x"), vl.get("y"), vl.get("z"),
                di.get("damageReason"), di.get("damageTypeCategory"),
                di.get("damageCauserName"), di.get("distance"),
                ev.get("isAttackerInVehicle", False),
            ))

        elif t == "LogPlayerTakeDamage":
            attacker = ev.get("attacker", {}) or {}
            victim = ev.get("victim", {}) or {}
            al = attacker.get("location", {}) or {}
            vl = victim.get("location", {}) or {}
            damage.append((
                match_id, ts, is_game,
                attacker.get("accountId"), attacker.get("name"),
                al.get("x"), al.get("y"), al.get("z"),
                victim.get("accountId"), victim.get("name"),
                vl.get("x"), vl.get("y"), vl.get("z"),
                ev.get("damage"),
                ev.get("damageReason"), ev.get("damageTypeCategory"),
                ev.get("damageCauserName"), ev.get("distance"),
                ev.get("isThroughPenetrableWall", False),
            ))

        elif t == "LogPhaseChange":
            phase_changes.append((
                match_id, ts, is_game,
                ev.get("phase"),
                ev.get("elapsedTime"),
            ))

        elif t == "LogMatchStart":
            match_start = ev

        elif t == "LogMatchEnd":
            match_end = ev

        elif t == "LogParachuteLanding":
            c = ev.get("character", {})
            loc = c.get("location", {})
            parachute_landings.append((
                match_id, ts,
                c.get("accountId"), c.get("name"), c.get("teamId"),
                loc.get("x"), loc.get("y"), loc.get("z"),
                ev.get("distance"),
            ))

        # ── 신규: 아이템 장착/해제 ──

        elif t in ("LogItemEquip", "LogItemUnequip"):
            etype = "equip" if t == "LogItemEquip" else "unequip"
            acc, name, tid, px, py, pz = _extract_character(ev)
            iid, cat, sub, _, attached = _extract_item(ev)
            item_equips.append((
                match_id, ts, is_game, etype,
                acc, name, tid, px, py, pz,
                iid, cat, sub, attached,
            ))

        # ── 신규: 아이템 획득/드롭 ──

        elif t in ITEM_PICKUP_TYPES:
            etype = ITEM_PICKUP_TYPES[t]
            acc, name, tid, px, py, pz = _extract_character(ev)
            iid, cat, sub, stack, attached = _extract_item(ev)

            cp_id = ev.get("carePackageUniqueId")
            lb_team = ev.get("ownerTeamId")
            lb_creator = ev.get("creatorAccountId")

            item_pickups.append((
                match_id, ts, is_game, etype,
                acc, name, tid, px, py, pz,
                iid, cat, sub, stack, attached,
                cp_id, lb_team, lb_creator,
            ))

        # ── 신규: 아이템 사용 ──

        elif t == "LogItemUse":
            acc, name, tid, px, py, pz = _extract_character(ev)
            iid, cat, sub, stack, _ = _extract_item(ev)
            item_uses.append((
                match_id, ts, is_game,
                acc, name, tid, px, py, pz,
                iid, cat, sub, stack,
            ))

    # ── 기존 데이터 삭제 후 벌크 INSERT (멱등성) ──
    try:
        with conn.cursor() as cur:
            # 재수집 시 중복 방지: 해당 match_id의 기존 행 삭제
            for tbl in (
                "telem_positions", "telem_game_states", "telem_kills",
                "telem_groggy", "telem_damage", "telem_phase_changes",
                "telem_match_start", "telem_match_end",
                "telem_parachute_landing",
                "telem_item_equip", "telem_item_pickup", "telem_item_use",
            ):
                cur.execute(f"DELETE FROM {SCHEMA}.{tbl} WHERE match_id = %s", (match_id,))

            # positions
            if positions:
                _bulk_insert(cur, f"{SCHEMA}.telem_positions", """
                    (match_id, event_time, is_game, elapsed_time, num_alive,
                     account_id, player_name, team_id, health,
                     pos_x, pos_y, pos_z,
                     vehicle_type, vehicle_id, vehicle_speed)
                """, positions)
                log.info(f"  positions: {len(positions)}행")

            # game_states
            if game_states:
                _bulk_insert(cur, f"{SCHEMA}.telem_game_states", """
                    (match_id, event_time, is_game, elapsed_time,
                     num_alive_teams, num_alive_players, num_join_players, num_start_players,
                     safe_zone_x, safe_zone_y, safe_zone_z, safe_zone_radius,
                     poison_zone_x, poison_zone_y, poison_zone_z, poison_zone_radius,
                     red_zone_x, red_zone_y, red_zone_z, red_zone_radius,
                     black_zone_x, black_zone_y, black_zone_z, black_zone_radius)
                """, game_states)
                log.info(f"  game_states: {len(game_states)}행")

            # kills
            if kills:
                _bulk_insert(cur, f"{SCHEMA}.telem_kills", """
                    (match_id, event_time, is_game,
                     killer_id, killer_name, killer_x, killer_y, killer_z,
                     victim_id, victim_name, victim_x, victim_y, victim_z,
                     dbno_maker_id, dbno_maker_name,
                     finisher_id, finisher_name,
                     damage_reason, damage_type, damage_causer, distance,
                     is_suicide, assists_cnt, victim_rank, victim_team_id)
                """, kills)
                log.info(f"  kills: {len(kills)}행")

            # groggy
            if groggy:
                _bulk_insert(cur, f"{SCHEMA}.telem_groggy", """
                    (match_id, event_time, is_game,
                     attacker_id, attacker_name,
                     attacker_x, attacker_y, attacker_z,
                     victim_id, victim_name,
                     victim_x, victim_y, victim_z,
                     damage_reason, damage_type, damage_causer, distance,
                     is_attacker_in_vehicle)
                """, groggy)
                log.info(f"  groggy: {len(groggy)}행")

            # damage
            if damage:
                _bulk_insert(cur, f"{SCHEMA}.telem_damage", """
                    (match_id, event_time, is_game,
                     attacker_id, attacker_name,
                     attacker_x, attacker_y, attacker_z,
                     victim_id, victim_name,
                     victim_x, victim_y, victim_z,
                     damage, damage_reason, damage_type, damage_causer, distance,
                     is_through_wall)
                """, damage)
                log.info(f"  damage: {len(damage)}행")

            # phase_changes
            if phase_changes:
                _bulk_insert(cur, f"{SCHEMA}.telem_phase_changes", """
                    (match_id, event_time, is_game, phase, elapsed_time)
                """, phase_changes)
                log.info(f"  phase_changes: {len(phase_changes)}행")

            # match_start
            if match_start:
                bz_opts = match_start.get("blueZoneCustomOptions")
                cur.execute(f"""
                    INSERT INTO {SCHEMA}.telem_match_start
                        (match_id, event_time, map_name, weather_id,
                         camera_view_type, team_size, is_custom, is_event_mode,
                         blue_zone_options)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_id) DO NOTHING
                """, (
                    match_id,
                    match_start.get("_D"),
                    match_start.get("mapName"),
                    match_start.get("weatherId"),
                    match_start.get("cameraViewType"),
                    match_start.get("teamSize"),
                    match_start.get("isCustomGame", False),
                    match_start.get("isEventMode", False),
                    json.dumps(bz_opts) if bz_opts else None,
                ))
                log.info(f"  match_start: 1행")

            # match_end
            if match_end:
                results = match_end.get("gameResultOnFinished", match_end.get("characters"))
                cur.execute(f"""
                    INSERT INTO {SCHEMA}.telem_match_end
                        (match_id, event_time, game_results)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (match_id) DO NOTHING
                """, (
                    match_id,
                    match_end.get("_D"),
                    json.dumps(results) if results else None,
                ))
                log.info(f"  match_end: 1행")

            # parachute_landing
            if parachute_landings:
                _bulk_insert(cur, f"{SCHEMA}.telem_parachute_landing", """
                    (match_id, event_time,
                     account_id, player_name, team_id,
                     pos_x, pos_y, pos_z, distance)
                """, parachute_landings)
                log.info(f"  parachute_landing: {len(parachute_landings)}행")

            # ── 신규: item_equip ──
            if item_equips:
                _bulk_insert(cur, f"{SCHEMA}.telem_item_equip", """
                    (match_id, event_time, is_game, event_type,
                     account_id, player_name, team_id,
                     pos_x, pos_y, pos_z,
                     item_id, category, sub_category, attached_items)
                """, item_equips)
                log.info(f"  item_equip: {len(item_equips)}행")

            # ── 신규: item_pickup ──
            if item_pickups:
                _bulk_insert(cur, f"{SCHEMA}.telem_item_pickup", """
                    (match_id, event_time, is_game, event_type,
                     account_id, player_name, team_id,
                     pos_x, pos_y, pos_z,
                     item_id, category, sub_category, stack_count, attached_items,
                     carepackage_id, lootbox_owner_team_id, lootbox_creator_id)
                """, item_pickups)
                log.info(f"  item_pickup: {len(item_pickups)}행")

            # ── 신규: item_use ──
            if item_uses:
                _bulk_insert(cur, f"{SCHEMA}.telem_item_use", """
                    (match_id, event_time, is_game,
                     account_id, player_name, team_id,
                     pos_x, pos_y, pos_z,
                     item_id, category, sub_category, stack_count)
                """, item_uses)
                log.info(f"  item_use: {len(item_uses)}행")

            # collection_log 업데이트
            cur.execute(f"""
                UPDATE {SCHEMA}.collection_log
                SET telemetry_fetched = TRUE, telemetry_fetched_at = NOW()
                WHERE match_id = %s
            """, (match_id,))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        log.error(f"텔레메트리 저장 실패 ({match_id}): {e}")
        _mark_error(conn, match_id, str(e))
        return False
    finally:
        conn.close()


def _bulk_insert(cur, table, columns, rows):
    """execute_values로 벌크 INSERT."""
    from psycopg2.extras import execute_values
    sql = f"INSERT INTO {table} {columns} VALUES %s"
    execute_values(cur, sql, rows, page_size=1000)


def _mark_error(conn, match_id, msg):
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {SCHEMA}.collection_log
                SET error_message = %s, retry_count = retry_count + 1
                WHERE match_id = %s
            """, (msg[:500], match_id))
        conn.commit()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("03 — 텔레메트리 수집 시작")
    log.info("=" * 50)

    pending = get_pending_telemetry()
    if args.limit:
        pending = pending[:args.limit]

    log.info(f"미처리 텔레메트리: {len(pending)}건")

    success = 0
    fail = 0

    for i, (match_id, telem_url) in enumerate(pending, 1):
        log.info(f"[{i}/{len(pending)}] {match_id}")

        events = fetch_telemetry_json(telem_url)
        if not events:
            log.error(f"  텔레메트리 다운로드 실패")
            fail += 1
            continue

        log.info(f"  이벤트 {len(events)}개 파싱 중...")
        if parse_and_save(match_id, events):
            success += 1
        else:
            fail += 1

    log.info(f"완료: 성공 {success}, 실패 {fail}")

    # 최종 현황
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



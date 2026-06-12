import { useState, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useMatchShard, useIndex } from "../lib/data";
import AnimatedNetwork, {
  type AnimColorBy,
} from "../components/AnimatedNetwork";
import { useTimePlayer } from "../lib/useTimePlayer";
import { fmtDec, fmtNum, fmtDuration, fmtSigned } from "../lib/format";
import StrategyBadge from "../components/StrategyBadge";
import TimeControls from "../components/TimeControls";
import { extractPhaseMarkers } from "../lib/phaseMarkers";
import type { StrategyType } from "../lib/types";

const COLOR_OPTIONS: { value: AnimColorBy; label: string }[] = [
  { value: "archetype", label: "4유형" },
  { value: "position", label: "네트워크 위치" },
  { value: "community", label: "커뮤니티" },
  { value: "movement", label: "이동성" },
  { value: "rank", label: "최종 순위" },
];

export default function MatchDetail() {
  const { matchId } = useParams<{ matchId: string }>();
  const shard = useMatchShard(matchId);
  const idx = useIndex();
  const [colorBy, setColorBy] = useState<AnimColorBy>("archetype");
  const [selectedTeam, setSelectedTeam] = useState<number | null>(null);

  const player = useTimePlayer({
    duration: shard.data?.duration_s ?? 600,
    autoplay: false,
    speed: 24,
  });

  const sortedNodes = useMemo(() => {
    if (!shard.data) return [];
    return [...shard.data.nodes].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
  }, [shard.data]);

  const phaseMarkers = useMemo(
    () => (shard.data ? extractPhaseMarkers(shard.data.frames) : []),
    [shard.data],
  );

  const selectedNode = useMemo(() => {
    if (!shard.data || selectedTeam === null) return null;
    return shard.data.nodes.find((n) => n.team === selectedTeam) ?? null;
  }, [shard.data, selectedTeam]);

  if (shard.isLoading) return <div className="text-muted">매치 로딩 중…</div>;
  if (shard.error || !shard.data)
    return <div className="text-bad">매치를 찾을 수 없습니다.</div>;

  const m = shard.data;

  return (
    <div className="space-y-4">
      {/* header */}
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-rule pb-3">
        <div>
          <Link to="/explorer" className="jet text-[11px] text-muted hover:text-accent">
            ← 매치 탐색으로
          </Link>
          <h1 className="news mt-1 text-2xl font-semibold">
            매치 <span className="jet">{m.match_id.slice(0, 8)}</span>
          </h1>
        </div>
        <div className="flex flex-wrap gap-3 text-xs">
          <Stat label="duration" value={fmtDuration(m.duration_s)} />
          <Stat label="n teams" value={String(m.n_nodes)} />
          <Stat label="n edges" value={String(m.n_edges)} />
          <Stat label="players" value={String(m.n_players)} />
          <Stat label="modularity" value={fmtDec(m.modularity_q, 3)} />
        </div>
      </header>

      {/* main grid: graph + side */}
      <div className="grid gap-4 lg:grid-cols-[2fr,1fr]">
        {/* graph */}
        <div>
          <div className="rounded border border-rule bg-paper p-3">
            <AnimatedNetwork
              nodes={m.nodes}
              edges={m.edges}
              events={m.events ?? []}
              frames={m.frames}
              arenaM={m.arena_m}
              t={player.t}
              colorBy={colorBy}
              size={620}
              mapUrls={idx.data?.map_urls}
            />
          </div>

          {/* time controls */}
          <div className="mt-3">
            <TimeControls
              duration={m.duration_s}
              t={player.t}
              onSeek={player.seek}
              playing={player.playing}
              onTogglePlay={player.togglePlay}
              speed={player.speed}
              onSpeedChange={player.setSpeed}
              phaseMarkers={phaseMarkers}
              enableShortcuts
            />
          </div>

          {/* color by */}
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded border border-rule bg-paper p-3 text-sm">
            <span className="jet text-[10px] uppercase text-muted">노드 색상</span>
            {COLOR_OPTIONS.map((c) => (
              <button
                key={c.value}
                onClick={() => setColorBy(c.value)}
                className={
                  "rounded px-2 py-1 transition-colors " +
                  (colorBy === c.value
                    ? "bg-cream text-ink"
                    : "text-muted hover:bg-cream/50")
                }
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>

        {/* side panel */}
        <aside className="space-y-3">
          <div className="rounded border border-rule bg-paper p-3">
            <div className="jet mb-2 text-[10px] uppercase text-muted">팀 목록 (순위순)</div>
            <div className="max-h-[420px] overflow-y-auto pr-1">
              {sortedNodes.map((n) => {
                const sel = n.team === selectedTeam;
                return (
                  <button
                    key={n.team}
                    onClick={() => setSelectedTeam(sel ? null : n.team)}
                    className={
                      "flex w-full items-center justify-between gap-2 border-b border-rule/60 px-2 py-1.5 text-left text-xs transition-colors hover:bg-cream/40 " +
                      (sel ? "bg-cream" : "")
                    }
                  >
                    <span className="jet w-10 numeric text-muted">
                      #{n.rank ?? "—"}
                    </span>
                    <span className="jet w-12 numeric">T{n.team}</span>
                    <span className="flex-1">
                      {n.strategy_type ? (
                        <StrategyBadge type={n.strategy_type as StrategyType} />
                      ) : (
                        <span className="jet text-[10px] text-muted">—</span>
                      )}
                    </span>
                    <span className="jet numeric text-muted">deg {n.degree}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {selectedNode ? (
            <div className="rounded border border-rule bg-paper p-3">
              <div className="jet mb-2 text-[10px] uppercase text-muted">
                팀 T{selectedNode.team} · rank #{selectedNode.rank ?? "—"}
              </div>
              {selectedNode.strategy_type ? (
                <div className="mb-2">
                  <StrategyBadge type={selectedNode.strategy_type as StrategyType} size="md" />
                </div>
              ) : null}
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <Field label="degree" value={String(selectedNode.degree)} />
                <Field label="weighted" value={fmtNum(selectedNode.dmg ?? 0, 0)} />
                <Field label="events" value={String(selectedNode.events)} />
                <Field label="community" value={String(selectedNode.community)} />
                <Field label="position" value={String(selectedNode.position)} />
                <Field label="archetype" value={String(selectedNode.archetype)} />
                <Field label="movement" value={String(selectedNode.movement_label)} />
                <Field label="survival(s)" value={String(selectedNode.survival ?? "—")} />
                <Field label="kills" value={String(selectedNode.team_kills ?? "—")} />
                <Field label="assists" value={String(selectedNode.team_assists ?? "—")} />
                <Field label="RP proxy" value={fmtSigned(selectedNode.ranked_rp_proxy, 2)} />
                <Field label="FightScore" value={fmtSigned(selectedNode.fight_score, 2)} />
                <Field label="ActiveScore" value={fmtSigned(selectedNode.active_score, 2)} />
                <Field label="bridge" value={fmtSigned(selectedNode.battle_bridge_score, 2)} />
                <Field label="이동 m/min" value={fmtDec(selectedNode.movement_per_min ?? 0, 0)} />
                <Field label="교전 참여" value={fmtDec(selectedNode.combat_st_participation_rate ?? 0, 3)} />
                <Field label="hotzone d(m)" value={fmtDec(selectedNode.distance_to_hotzone_m ?? 0, 0)} />
                <Field label="farm group" value={String(selectedNode.farm_position_group ?? "—")} />
              </dl>
            </div>
          ) : (
            <div className="rounded border border-rule bg-cream/40 p-3 text-xs text-muted">
              팀을 선택하면 상세 스탯이 표시됩니다.
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-rule bg-cream/40 px-2 py-1">
      <span className="jet text-[9px] uppercase text-muted">{label}</span>
      <div className="numeric text-sm">{value}</div>
    </div>
  );
}
function Field({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="jet text-[10px] uppercase text-muted">{label}</dt>
      <dd className="numeric text-right">{value}</dd>
    </>
  );
}

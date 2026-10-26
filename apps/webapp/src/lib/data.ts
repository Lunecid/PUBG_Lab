// 새 webapp 데이터 페처. TanStack Query 와 함께 사용.

import { useQuery } from "@tanstack/react-query";
import type { MatchShard } from "./types";
import type {
  IndexFile,
  NetworkSummary,
  StrategySummary,
  PhaseSummary,
  HypothesesFile,
  MatchIndexEntry,
  GlossaryEntry,
  MethodsFile,
} from "./types";

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return (await r.json()) as T;
}

const KEYS = {
  index: ["index"],
  network: ["network_summary"],
  strategy: ["strategy_summary"],
  phase: ["phase_summary"],
  hypotheses: ["hypotheses"],
  matchIndex: ["match_index"],
  glossary: ["glossary"],
  methods: ["methods"],
  match: (id: string) => ["match", id],
};

export function useIndex() {
  return useQuery({
    queryKey: KEYS.index,
    queryFn: () => fetchJson<IndexFile>("/data/index.json"),
    staleTime: Infinity,
  });
}

export function useNetworkSummary() {
  return useQuery({
    queryKey: KEYS.network,
    queryFn: () => fetchJson<NetworkSummary>("/data/network_summary.json"),
    staleTime: Infinity,
  });
}

export function useStrategySummary() {
  return useQuery({
    queryKey: KEYS.strategy,
    queryFn: () => fetchJson<StrategySummary>("/data/strategy_summary.json"),
    staleTime: Infinity,
  });
}

export function usePhaseSummary() {
  return useQuery({
    queryKey: KEYS.phase,
    queryFn: () => fetchJson<PhaseSummary>("/data/phase_summary.json"),
    staleTime: Infinity,
  });
}

export function useHypotheses() {
  return useQuery({
    queryKey: KEYS.hypotheses,
    queryFn: () => fetchJson<HypothesesFile>("/data/hypotheses.json"),
    staleTime: Infinity,
  });
}

export function useMatchIndex() {
  return useQuery({
    queryKey: KEYS.matchIndex,
    queryFn: () => fetchJson<MatchIndexEntry[]>("/data/match_index.json"),
    staleTime: Infinity,
  });
}

export function useGlossary() {
  return useQuery({
    queryKey: KEYS.glossary,
    queryFn: () => fetchJson<GlossaryEntry[]>("/data/glossary.json"),
    staleTime: Infinity,
  });
}

export function useMethods() {
  return useQuery({
    queryKey: KEYS.methods,
    queryFn: () => fetchJson<MethodsFile>("/data/methods.json"),
    staleTime: Infinity,
  });
}

export function useMatchShard(matchId: string | undefined) {
  return useQuery({
    queryKey: KEYS.match(matchId ?? ""),
    queryFn: () => fetchJson<MatchShard>(`/data/matches/${matchId}.json`),
    enabled: !!matchId,
    staleTime: Infinity,
  });
}

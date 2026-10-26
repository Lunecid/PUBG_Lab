"""
네트워크 분석 설정.
"""

import os

# ── 프로젝트 루트 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_ROOT = os.path.join(PROJECT_ROOT, "data", "graphs")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── DB (legacy/main.py와 동일) ──
DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DBNAME", "pubg_survival"),
    "user": os.environ.get("PG_USER", "postgres"),
}
SCHEMA = "pubg"


def require_pg_password() -> str:
    pw = os.environ.get("PG_PASSWORD")
    if not pw:
        raise RuntimeError(
            "PG_PASSWORD 환경변수가 설정되지 않았습니다. "
            "DB 접속 전 export PG_PASSWORD='<비밀번호>' 를 실행하세요 (.env.example 참조)."
        )
    return pw


# ── 데이터 선택 ──
DEFAULT_MAP = "Baltic_Main"
DEFAULT_MODE = "squad"
DEFAULT_N_MATCHES = 3

# ── 네트워크 구성 ──
# 교전 엣지 윈도우 (최근 N초 이내의 공격자→피해자 관계 집계)
COMBAT_WINDOW_SEC = 30
# 교전 엣지 최소 데미지 (이 값 이하의 누적 데미지는 엣지로 그리지 않음)
COMBAT_MIN_DMG = 5.0
# 실제로 같은 사람(자살)이거나 bluezone 데미지는 교전이 아님 → 제외
EXCLUDE_CAUSERS = ("Bluezonebomb_EffectActor_C",)

# ── 스냅샷 샘플링 ──
SNAPSHOT_INTERVAL_SEC = 10

# ── 시각화 ──
ARENA_SIZE_M = 8160.0  # Baltic_Main 기준

# ── 착륙 (parachute landing) cutoff ──
# PUBG 매치 시작 후 약 N초까지는 비행기 + 낙하 단계.
# 이 구간의 centroid 이동은 모든 팀에 공통적으로 큰 직선 이동을 만들어
# 실제 지상 이동 분석을 흐린다 → trajectory 측정에서 제외.
LANDING_OFFSET_S = 90.0

# ── Movement 측정 고정 윈도우 ──
# 후반 zone 압박은 "살아남은 팀이 강제로 더 많이 움직이게" 만든다.
# 결과적으로 분당 이동거리가 생존 시간과 강하게 결합되어 비교 불공정.
# → 모든 팀을 [LANDING_OFFSET_S, LANDING_OFFSET_S + MOVEMENT_WINDOW_S]
#    동일 초·중반 구간에서만 측정. 일찍 죽은 팀은 살아있던 동안만 누적,
#    오래 산 팀도 윈도우 종료 시점까지만 누적.
MOVEMENT_WINDOW_S = 300.0  # 착륙 후 5분

# 맵 이미지 — 우선순위: 로컬 파일 (오프라인 캐시) > CDN URL (fallback 체인)
# visualize.py 가 쓰는 것과 동일한 GitHub raw 주소 사용
MAP_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "Baltic_Main.jpg"
)
MAP_IMAGE_URLS = [
    "https://raw.githubusercontent.com/pubg/api-assets/master/Assets/Maps/Erangel_Main_Low_Res.png",
    "https://raw.githubusercontent.com/pubg/api-assets/master/Assets/Maps/Erangel_Main_High_Res.png",
]

# PUBG 맵 코드명 → CDN 이미지 URL fallback chain.
# 우선순위: Low_Res (가벼움) → High_Res. 첫 매치에 한해 클라이언트가 onload 시도.
_PUBG_CDN = "https://raw.githubusercontent.com/pubg/api-assets/master/Assets/Maps"
MAP_CDN_BY_NAME = {
    # Erangel 계열
    "Baltic_Main": [
        f"{_PUBG_CDN}/Erangel_Main_Low_Res.png",
        f"{_PUBG_CDN}/Erangel_Main_High_Res.png",
    ],
    "Erangel_Main": [
        f"{_PUBG_CDN}/Erangel_Main_Low_Res.png",
        f"{_PUBG_CDN}/Erangel_Main_High_Res.png",
    ],
    # 그 외 메인 맵
    "Desert_Main": [
        f"{_PUBG_CDN}/Miramar_Main_Low_Res.png",
        f"{_PUBG_CDN}/Miramar_Main_High_Res.png",
    ],
    "Savage_Main": [
        f"{_PUBG_CDN}/Sanhok_Main_Low_Res.png",
        f"{_PUBG_CDN}/Sanhok_Main_High_Res.png",
    ],
    "DihorOtok_Main": [
        f"{_PUBG_CDN}/Vikendi_Main_Low_Res.png",
        f"{_PUBG_CDN}/Vikendi_Main_High_Res.png",
    ],
    "Tiger_Main": [f"{_PUBG_CDN}/Taego_Main_Low_Res.png", f"{_PUBG_CDN}/Taego_Main_High_Res.png"],
    "Kiki_Main": [f"{_PUBG_CDN}/Deston_Main_Low_Res.png", f"{_PUBG_CDN}/Deston_Main_High_Res.png"],
    "Summerland_Main": [
        f"{_PUBG_CDN}/Karakin_Main_Low_Res.png",
        f"{_PUBG_CDN}/Karakin_Main_High_Res.png",
    ],
    "Chimera_Main": [
        f"{_PUBG_CDN}/Paramo_Main_Low_Res.png",
        f"{_PUBG_CDN}/Paramo_Main_High_Res.png",
    ],
    "Heaven_Main": [f"{_PUBG_CDN}/Haven_Main_Low_Res.png", f"{_PUBG_CDN}/Haven_Main_High_Res.png"],
    "Range_Main": [f"{_PUBG_CDN}/Range_Main_Low_Res.png"],
}
# 맵 회전 보정 (north-up 아닐 때만 사용, 기본 0)
MAP_ROTATE_DEG = 0
# 맵 기본 opacity — visualize.py 와 동일하게 0.7
MAP_OPACITY_DEFAULT = 0.70

# Top-K for ranking panel
TOP_K = 10

# 선택 가능한 centrality 지표
CENTRALITY_MEASURES = [
    "degree",
    "closeness",
    "betweenness",
    "eigenvector",
    "pagerank",
]

# ── Community Detection (CH5) ──
COMMUNITY_ENABLED = True
COMMUNITY_SEED = 0

# ── Narrative (프레임 해설) ──
NARRATIVE_ENABLED = True
NARRATIVE_TREND_WINDOW = 5  # 최근 N 프레임 대비 추세 판단

# ── 시각화 기본 뷰 ──
VIEW_MODE_DEFAULT = "team"  # team | louvain | spectral | centrality

# 시계열 차트에 그릴 지표 (visualize_match.py 에서 사용)
TIMESERIES_CHARTS = [
    {
        "id": "ts-density",
        "title": "density / clustering",
        "series": [("density", "#60a5fa"), ("avg_clustering", "#f59e0b")],
    },
    {
        "id": "ts-path",
        "title": "diameter / avg path",
        "series": [("diameter", "#a78bfa"), ("avg_path_length", "#34d399")],
    },
    {
        "id": "ts-edges",
        "title": "n_edges / alive",
        "series": [("n_edges", "#f472b6"), ("alive", "#94a3b8")],
    },
    {
        "id": "ts-mod",
        "title": "modularity Q (Louvain / Spectral)",
        "series": [("modularity_q_louvain", "#10b981"), ("modularity_q_spectral", "#06b6d4")],
    },
    {
        "id": "ts-ncomm",
        "title": "# communities",
        "series": [("n_communities_louvain", "#10b981"), ("n_communities_spectral", "#06b6d4")],
    },
    {
        "id": "ts-combat",
        "title": "combat edges / total dmg",
        "series": [("combat_edges", "#ef4444"), ("combat_dmg_total", "#fb923c")],
    },
]

# 지표 툴팁 (교재 정의)
METRIC_TOOLTIPS = {
    "density": "density = 실제 엣지 / 가능한 엣지. 0 = 고립, 1 = 완전 연결.",
    "avg_clustering": "C̄ = 평균 클러스터링 계수. 이웃끼리도 서로 연결될 확률 (삼각형 밀도).",
    "diameter": "diameter = 그래프 내 최단경로 중 가장 긴 것.",
    "avg_path_length": "평균 최단 경로 길이. 작을수록 small-world.",
    "betweenness": "Betweenness: 다른 쌍들의 최단 경로가 이 노드를 지나는 비율 (중개).",
    "closeness": "Closeness: 다른 모든 노드까지 평균 거리의 역수 (가까움).",
    "eigenvector": "Eigenvector: 중요한 이웃에 연결될수록 본인도 중요. λc = Ac.",
    "pagerank": "PageRank: eigenvector 의 random-teleport 확장. 구글 검색 랭킹 원형.",
    "modularity_q": "Q = (그룹 내 엣지) − (랜덤 기대값). Q > 0.3 유의 (Newman 2004).",
}

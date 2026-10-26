"""
Run Metadata Utility
=====================
실행 결과에 날짜, 버전, 조건을 자동으로 포함시키는 유틸리티.
모든 출력 파일명과 메타데이터에 일관된 run ID를 부여한다.
"""

import os
import glob
import subprocess
from datetime import datetime


def get_git_version():
    """현재 git commit short hash 반환. git 없으면 'nogit'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "nogit"


def get_git_branch():
    """현재 git branch 이름."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_git_dirty():
    """uncommitted 변경 여부."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def make_run_id():
    """
    실행 ID 생성: YYYYMMDD_HHMMSS_<git_short>

    예: 20260407_143052_2b66a8a
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_ver = get_git_version()
    return f"{ts}_{git_ver}"


def make_run_meta(**conditions):
    """
    실행 메타데이터 dict 생성.

    Parameters:
        **conditions: 실행 조건 (map, mode, epochs, lr, etc.)

    Returns:
        dict with run_id, timestamp, git info, conditions
    """
    now = datetime.now()
    meta = {
        "run_id": make_run_id(),
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "git_commit": get_git_version(),
        "git_branch": get_git_branch(),
        "git_dirty": get_git_dirty(),
        "conditions": conditions,
    }
    return meta


def make_output_dir(base_dir, run_id=None):
    """
    날짜/run_id 기반 출력 디렉토리 생성.

    base_dir/YYYYMMDD_HHMMSS_<git>/

    Parameters:
        base_dir: 기본 출력 경로 (e.g., checkpoints/Baltic_Main/squad-fpp)
        run_id: 지정 시 사용, 없으면 자동 생성

    Returns:
        (output_dir, run_id)
    """
    if run_id is None:
        run_id = make_run_id()
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, run_id


def stamp_filename(base_name, run_id=None, ext=None):
    """
    파일명에 run_id 스탬프 추가.

    예: simulation_result.json → simulation_result_20260407_143052_2b66a8a.json

    Parameters:
        base_name: 원본 파일명
        run_id: 지정 시 사용, 없으면 자동 생성
        ext: 확장자 오버라이드

    Returns:
        stamped filename
    """
    if run_id is None:
        run_id = make_run_id()

    name, orig_ext = os.path.splitext(base_name)
    if ext is None:
        ext = orig_ext

    return f"{name}_{run_id}{ext}"


# ============================================================
# 탐색 유틸리티
# ============================================================

def discover_checkpoints(base_dir="checkpoints"):
    """
    사용 가능한 체크포인트 목록 탐색.

    경로 구조:
      checkpoints/{MAP}/{MODE}/{TIMESTAMP}/best_model.pt   (신규)
      checkpoints/{MAP}/{MODE}/best_model.pt               (레거시)

    Returns:
        list[dict]: [{"path", "map", "mode", "run_id"}, ...] — run_id 역순
    """
    candidates = glob.glob(
        os.path.join(base_dir, "**", "best_model.pt"), recursive=True
    )
    if not candidates:
        return []

    results = []
    for path in candidates:
        # checkpoints/ 이후의 경로를 파싱
        rel = os.path.relpath(path, base_dir)
        parts = rel.replace("\\", "/").split("/")
        # parts 예: ["Baltic_Main", "squad-fpp", "20260407_..", "best_model.pt"]
        #       또는 ["Baltic_Main", "squad-fpp", "best_model.pt"]

        if len(parts) >= 4:
            # 타임스탬프 서브디렉토리 있음
            map_name, mode, run_id = parts[0], parts[1], parts[2]
        elif len(parts) == 3:
            # 레거시: 타임스탬프 없음
            map_name, mode = parts[0], parts[1]
            run_id = "legacy"
        else:
            continue

        results.append({
            "path": path,
            "map": map_name,
            "mode": mode,
            "run_id": run_id,
        })

    # run_id 역순 (최신이 먼저)
    results.sort(key=lambda x: x["run_id"], reverse=True)
    return results


def discover_matches(base_dir="data/graphs", map_filter=None, mode_filter=None):
    """
    사용 가능한 매치 .pt 파일 목록 탐색.

    경로 구조: data/graphs/{MAP}/{MODE}/match_XXX.pt

    Parameters:
        base_dir: 그래프 데이터 루트
        map_filter: 특정 맵만 필터 (e.g., "Baltic_Main")
        mode_filter: 특정 모드만 필터 (e.g., "squad-fpp")

    Returns:
        list[dict]: [{"path", "map", "mode", "filename"}, ...] — 파일명순
    """
    candidates = glob.glob(
        os.path.join(base_dir, "**", "*.pt"), recursive=True
    )
    if not candidates:
        return []

    results = []
    for path in candidates:
        rel = os.path.relpath(path, base_dir)
        parts = rel.replace("\\", "/").split("/")
        # parts 예: ["Baltic_Main", "squad-fpp", "match_001.pt"]

        if len(parts) < 3:
            continue

        map_name, mode, filename = parts[0], parts[1], parts[-1]

        if map_filter and map_name != map_filter:
            continue
        if mode_filter and mode != mode_filter:
            continue

        results.append({
            "path": path,
            "map": map_name,
            "mode": mode,
            "filename": filename,
        })

    results.sort(key=lambda x: x["filename"])
    return results

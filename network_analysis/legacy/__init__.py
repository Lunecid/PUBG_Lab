"""
Legacy network analysis modules (pre-v2 redesign)
==================================================
재설계 이전 H1/H2/H3 가설 검증 코드. v2 신규 가설 (archetype / battle-center /
movement) 의 cross-validation 및 reference 용으로 보존.

신규 코드는 network_analysis 루트의:
  - archetype.py / movement.py  (공유 feature)
  - h1_archetype.py / h2_battle_center.py / h3_movement.py  (가설 모듈)
  - export_json.py / run_v2.py  (오케스트레이션)
를 사용한다.

신규 모듈에서 재사용하는 헬퍼:
  from network_analysis.legacy.h1_analysis import (
      _pearson_safe, partial_correlation,
  )
"""

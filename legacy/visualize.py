"""
PUBG Match Graph Visualizer — 3D (OrbitControls) v2
=====================================================
수정 사항:
1. 밝은 배경 + 높은 대비 엣지 색상
2. safe_zone(흰원) + poison_zone(파란원) 이중 자기장
3. 클릭 판정 수정 (mousedown/mouseup 거리 기반)
4. Erangel 맵 텍스처 지면
5. z축 스케일 교정 (min-max 정규화)
6. 노드 피처 14개 (safe/poison 구분) 반영

사용법:
  python visualize.py
  python visualize.py data/graphs/match_0046ac50.pt
"""

import sys, os, json, torch, numpy as np


PUBG_MAP_SIZES = {
    "Baltic_Main": 8160,      # Erangel
    "Erangel_Main": 8160,
    "Desert_Main": 8160,      # Miramar
    "Savage_Main": 4096,      # Sanhok
    "DihorOtok_Main": 6144,   # Vikendi
    "Summerland_Main": 2048,  # Karakin
    "Chimera_Main": 3072,     # Paramo
    "Heaven_Main": 1024,      # Haven
    "Tiger_Main": 8160,       # Taego
    "Kiki_Main": 8160,        # Deston
    "Neon_Main": 8160,        # Rondo
    "Range_Main": 8160,
}


def infer_map_size(meta):
    """
    .pt 메타에서 맵 크기 추출.
    우선순위: heatmap 역산 > map_name 룩업 > 8160 폴백.
    """
    hm = meta.get("heatmaps")
    if hm and "cell_size_m" in hm and "grid_size" in hm:
        size = float(hm["cell_size_m"]) * int(hm["grid_size"])
        if size > 0:
            return size

    map_name = meta.get("map_name", "")
    if map_name in PUBG_MAP_SIZES:
        return float(PUBG_MAP_SIZES[map_name])

    print(f"  ⚠ map size 추정 실패 (map_name={map_name!r}) — 8160m 폴백")
    return 8160.0


def load_graph_data(filepath):
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    return data["graphs"], data["snapshot_times"], data["meta"]


def graphs_to_json(graphs, snapshot_times, meta):
    """
    그래프 스냅샷 → JSON dict 변환.

    새 39차원 스키마 기준:
      - x[:, 0] = hp (0~1 정규화)
      - x[:, 5] = pos_x / map_size (0~1 정규화)
      - x[:, 6] = pos_y / map_size (0~1 정규화)
    좌표는 map_size를 곱해 raw meter로 복원.
    3D 고도(pos_z) 인덱스 미확정 → 0 고정 (평면 처리).
    info panel 부가 필드(sd, sb, dd, dt 등)는 매핑 미확정 → 0.
    """
    total_p = meta["total_players"]
    map_size = infer_map_size(meta)

    # 옛 스키마 경고
    sample_dim = 0
    for g in graphs:
        if g["player"].x.shape[0] > 0:
            sample_dim = g["player"].x.shape[1]
            break
    if 0 < sample_dim < 7:
        print(f"  ⚠ player feature 차원이 {sample_dim} — 옛 스키마 .pt일 수 있음. 재생성 권장.")

    # z축: 평면 처리 (고도 인덱스 미확정)
    z_min, z_max = 0.0, 1.0

    frames = []
    for g, t in zip(graphs, snapshot_times):
        x = g["player"].x
        n = x.shape[0]
        team_idx = g["player"].team_idx.numpy().tolist()
        global_pid = g["player"].global_pid.numpy().tolist()
        account_ids = g["player"].account_ids
        team_ids = g["player"].team_ids

        pid_to_local = {global_pid[i]: i for i in range(n)}

        nodes = []
        for pid in range(total_p):
            if pid in pid_to_local:
                li = pid_to_local[pid]
                px = x[li, 5].item() * map_size if x.shape[1] > 6 else 0.0
                py = x[li, 6].item() * map_size if x.shape[1] > 6 else 0.0
                hp = x[li, 0].item()
                nodes.append({
                    "pid": pid, "alive": 1,
                    "x": round(px, 1), "y": round(py, 1),
                    "z": 0.0, "hp": round(hp, 3),
                    # 부가 필드 — 매핑 미확정, 0 처리
                    "sd": 0, "sb": 0, "is": 1,
                    "pd": 0, "pb": 0, "ip": 1,
                    "vs": 0, "iv": 0,
                    "dd": round(x[li, 12].item(), 1) if x.shape[1] > 13 else 0,
                    "dt": round(x[li, 13].item(), 1) if x.shape[1] > 13 else 0,
                    "t": team_idx[li],
                    "aid": account_ids[li][-8:] if account_ids[li] else "",
                    "tid": str(team_ids[li])[-6:] if team_ids[li] else "",
                })
            else:
                nodes.append({"pid": pid, "alive": 0})

        # 엣지
        ally_ei = g["player", "ally", "player"].edge_index
        ally_edges = []
        for e in range(ally_ei.shape[1]):
            ls, ld = ally_ei[0,e].item(), ally_ei[1,e].item()
            ps, pd = global_pid[ls], global_pid[ld]
            if ps < pd:
                ally_edges.append([ps, pd])

        enc_ei = g["player", "encounter", "player"].edge_index
        enc_attr = g["player", "encounter", "player"].edge_attr
        enc_edges = []; seen = set()
        for e in range(enc_ei.shape[1]):
            ls, ld = enc_ei[0,e].item(), enc_ei[1,e].item()
            ps, pd = global_pid[ls], global_pid[ld]
            key = (min(ps,pd), max(ps,pd))
            if key not in seen:
                seen.add(key)
                dist = round(enc_attr[e,0].item(), 0) if enc_attr.shape[0]>0 else 0
                enc_edges.append([ps, pd, dist])

        # zone context: [safe_cx, safe_cy, safe_r, poison_cx, poison_cy, poison_r, area, density, alive, time]
        zv = g["zone"].x[0]
        zone = {
            "scx": round(zv[0].item(),1), "scy": round(zv[1].item(),1),
            "sr": round(zv[2].item(),1),
            "pcx": round(zv[3].item(),1), "pcy": round(zv[4].item(),1),
            "pr": round(zv[5].item(),1),
            "alive": int(zv[8].item()),
        }
        frames.append({"t": round(t,1), "nodes": nodes,
                       "ally": ally_edges, "enc": enc_edges, "zone": zone})

    return {
        "match_id": meta["match_id"],
        "total_players": total_p,
        "total_teams": len(meta["team_rank"]),
        "map_name": meta.get("map_name", ""),
        "map_size": map_size,
        "z_min": round(z_min, 1),
        "z_max": round(z_max, 1),
        "frames": frames,
        "heatmaps": meta.get("heatmaps", None),
    }


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>PUBG 3D Graph v2</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
:root{
  --bg:#e8ecf1;--panel:rgba(255,255,255,.92);--border:#c8cdd5;
  --text:#2a2e35;--muted:#6b7280;--accent:#2563eb;--red:#dc2626;
  --green:#16a34a;--warn:#d97706;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'SF Mono',Menlo,'Fira Code',monospace;overflow:hidden;font-size:12px}
canvas{display:block}

.panel{position:absolute;background:var(--panel);border:1px solid var(--border);
       border-radius:8px;padding:12px 16px;backdrop-filter:blur(8px);line-height:1.7;
       box-shadow:0 2px 8px rgba(0,0,0,.08)}

#hud{top:12px;left:12px;pointer-events:none;min-width:180px}
#hud .row{display:flex;justify-content:space-between;gap:12px}
#hud .lbl{color:var(--muted)}
#hud .val{font-weight:600}

#info{top:12px;right:12px;min-width:250px;display:none}
#info h3{color:var(--accent);font-size:13px;margin-bottom:6px;border-bottom:1px solid var(--border);padding-bottom:5px}
.ir{display:flex;justify-content:space-between;padding:1px 0}
.il{color:var(--muted)}.iv{color:var(--text);font-weight:500}

#legend{bottom:54px;right:12px;font-size:11px}
.lg{display:flex;align-items:center;gap:8px;margin:3px 0}
.ll{width:22px;height:3px;border-radius:2px}
.ld{width:9px;height:9px;border-radius:50%;border:1px solid rgba(0,0,0,.12)}
.hm-btn{background:var(--panel);border:1px solid var(--border);color:var(--muted);border-radius:4px;
        padding:3px 8px;cursor:pointer;font-size:11px;margin:2px 0;font-family:inherit;transition:all .15s}
.hm-btn:hover,.hm-btn.active{color:var(--accent);border-color:var(--accent);font-weight:600}

#controls{position:absolute;bottom:0;left:0;right:0;display:flex;align-items:center;
          gap:10px;padding:10px 16px;background:rgba(255,255,255,.95);border-top:1px solid var(--border)}
#controls button{background:#f3f4f6;color:var(--muted);border:1px solid var(--border);border-radius:5px;
                 padding:6px 14px;cursor:pointer;font-size:12px;font-family:inherit;transition:all .12s}
#controls button:hover{background:#e5e7eb;color:var(--text)}
#controls button.on{background:var(--accent);border-color:var(--accent);color:#fff}
#timeline{flex:1;height:5px;-webkit-appearance:none;appearance:none;
          background:#d1d5db;border-radius:3px;outline:none;cursor:pointer}
#timeline::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
                                background:var(--accent);border-radius:50%;cursor:pointer;
                                box-shadow:0 0 4px rgba(37,99,235,.3)}
#sp{width:46px;text-align:center;font-size:11px;color:var(--muted)}
#tip{position:absolute;bottom:54px;left:12px;font-size:11px;color:var(--muted);
     background:var(--panel);padding:5px 10px;border-radius:6px;border:1px solid var(--border)}
</style>
</head><body>

<div class="panel" id="hud">
  <div class="row"><span class="lbl">Time</span><span class="val" id="ht">0s</span></div>
  <div class="row"><span class="lbl">Alive</span><span class="val" id="ha">0 / __TOTAL__</span></div>
  <div class="row"><span class="lbl">Poison R</span><span class="val" id="hz">0m</span></div>
  <div class="row"><span class="lbl">Safe R</span><span class="val" id="hsr">0m</span></div>
  <div class="row"><span class="lbl">Teams</span><span class="val" id="htm">0</span></div>
  <div class="row"><span class="lbl">Combat</span><span class="val" id="hc">0</span></div>
</div>

<div class="panel" id="info">
  <h3 id="ni-title">Player</h3>
  <div class="ir"><span class="il">Account</span><span class="iv" id="ni-aid">-</span></div>
  <div class="ir"><span class="il">Team</span><span class="iv" id="ni-team">-</span></div>
  <div class="ir"><span class="il">HP</span><span class="iv" id="ni-hp">-</span></div>
  <div class="ir"><span class="il">Position</span><span class="iv" id="ni-pos">-</span></div>
  <div class="ir"><span class="il">Altitude</span><span class="iv" id="ni-alt">-</span></div>
  <div style="height:1px;background:var(--border);margin:5px 0"></div>
  <div class="ir"><span class="il">Safe Dist</span><span class="iv" id="ni-sd">-</span></div>
  <div class="ir"><span class="il">Inside Safe</span><span class="iv" id="ni-is">-</span></div>
  <div class="ir"><span class="il">Poison Dist</span><span class="iv" id="ni-pd">-</span></div>
  <div class="ir"><span class="il">Inside Poison</span><span class="iv" id="ni-ip">-</span></div>
  <div style="height:1px;background:var(--border);margin:5px 0"></div>
  <div class="ir"><span class="il">Vehicle</span><span class="iv" id="ni-veh">-</span></div>
  <div class="ir"><span class="il">Dmg Dealt</span><span class="iv" id="ni-dd" style="color:var(--red)">-</span></div>
  <div class="ir"><span class="il">Dmg Taken</span><span class="iv" id="ni-dt" style="color:var(--warn)">-</span></div>
  <div class="ir"><span class="il">Status</span><span class="iv" id="ni-st">-</span></div>
</div>

<div class="panel" id="legend">
  <div class="lg"><div class="ll" style="background:rgba(37,99,235,.8)"></div>Ally</div>
  <div class="lg"><div class="ll" style="background:rgba(220,38,38,.6)"></div>Encounter</div>
  <div class="lg"><div class="ld" style="background:var(--accent)"></div>Player</div>
  <div class="lg"><div class="ld" style="background:var(--red);box-shadow:0 0 5px var(--red)"></div>In Combat</div>
  <div style="height:1px;background:var(--border);margin:5px 0"></div>
  <div style="color:var(--muted);margin-bottom:3px">Zone</div>
  <div class="lg"><div class="ll" style="background:rgba(255,255,255,.9);border:1px solid #888"></div>Safe (target)</div>
  <div class="lg"><div class="ll" style="background:rgba(59,130,246,.5)"></div>Poison (dmg)</div>
  <div style="height:1px;background:var(--border);margin:5px 0"></div>
  <div style="color:var(--muted);margin-bottom:3px">Heatmap</div>
  <button class="hm-btn" data-hm="elevation">Elevation</button>
  <button class="hm-btn" data-hm="density">Density</button>
  <button class="hm-btn" data-hm="combat">Combat</button>
  <button class="hm-btn active" data-hm="none">Off</button>
  <div id="hm-range" style="font-size:10px;color:var(--muted);margin-top:3px"></div>
</div>

<div id="tip">Left: rotate | Middle/Right: pan | Scroll: zoom | Click node (paused): details</div>

<div id="controls">
  <button id="bp" class="on">&#x23F8;</button>
  <button class="sb" data-s="0.5">0.5x</button>
  <button class="sb on" data-s="1">1x</button>
  <button class="sb" data-s="2">2x</button>
  <button class="sb" data-s="4">4x</button>
  <input type="range" id="timeline" min="0" max="1000" value="0">
  <span id="sp">1x</span>
</div>

<script>
const D=__JSON_DATA__;
const TC=["#e6194b","#3cb44b","#0077b6","#e85d04","#7209b7","#38b000","#dc2f02",
"#4361ee","#f77f00","#2d6a4f","#9b2226","#023e8a","#ca6702","#588157","#d00000",
"#6a040f","#0096c7","#370617","#55a630","#bc4749","#264653","#e76f51","#2a9d8f",
"#f4a261","#1d3557","#a8dadc","#457b9d","#e63946","#6d6875","#b5838d"];

const F=D.frames, NF=F.length;
const Z_MIN=D.z_min, Z_MAX=D.z_max, Z_RANGE=Math.max(Z_MAX-Z_MIN,1);
// z축 스케일: 지면 고도 범위(p1~p99) 기준, 맵 대비 충분히 보이도록
// 맵이 8160m일 때 지면 고도차가 최소 200 Three.js 단위는 되어야 시각적으로 구분 가능
const MAP_SIZE = D.map_size || 8160;
const Z_SCALE = Math.max(Math.min(MAP_SIZE / Z_RANGE * 0.08, 15), 2.0);
// z값 클램프 범위 (비행기 고도가 스케일을 깨지 않도록)
const Z_CLAMP_MAX = Z_MAX + Z_RANGE * 0.5;
const Z_CLAMP_MIN = Z_MIN - Z_RANGE * 0.2;

let playing=true, spd=1, prog=0, lt=0, selNode=-1;

// === SCENE ===
const W=innerWidth, H=innerHeight-40;
const scene=new THREE.Scene();
scene.background=new THREE.Color(0xdee2e8);
scene.fog=new THREE.FogExp2(0xdee2e8, 0.000025);

const camera=new THREE.PerspectiveCamera(55, W/H, 10, 80000);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W,H);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled=false;
document.body.prepend(renderer.domElement);

const controls=new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping=true;
controls.dampingFactor=0.12;
controls.rotateSpeed=0.8;
controls.panSpeed=0.6;
controls.zoomSpeed=1.2;
controls.mouseButtons={LEFT:THREE.MOUSE.ROTATE, MIDDLE:THREE.MOUSE.PAN, RIGHT:THREE.MOUSE.PAN};
controls.minDistance=100;
controls.maxDistance=25000;

if(NF>0){
  // Map-centred aerial view — robust to matches where the first frame's zone
  // is uninitialised (scx=scy=sr=0).  Snap once to the centre of the map.
  const cx = MAP_SIZE / 2, cz = MAP_SIZE / 2;
  controls.target.set(cx, 200, cz);
  camera.position.set(cx + MAP_SIZE * 0.40, MAP_SIZE * 0.55, cz + MAP_SIZE * 0.40);
  controls.update();
}

addEventListener("resize",()=>{
  const w=innerWidth, h=innerHeight-40;
  camera.aspect=w/h; camera.updateProjectionMatrix();
  renderer.setSize(w,h);
});

// === LIGHTS ===
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dl=new THREE.DirectionalLight(0xffffff, 0.55);
dl.position.set(5000,8000,5000); scene.add(dl);

// === MAP GROUND PLANE — 3D TERRAIN MESH ===
const MAP_SIZE_M = MAP_SIZE;

// === TERRAIN MESH ===
// Build a displaced PlaneGeometry whose Y per-vertex comes from D.heatmaps.elevation
// (when available) and falls back to a flat plane otherwise.  An external DEM
// can override the embedded grid by supplying D.terrain = {grid, elev_min_m,
// elev_max_m, flip_y, rot_deg} from the Python side.
const TERR = D.terrain || (D.heatmaps && D.heatmaps.elevation ? {
  grid:      D.heatmaps.elevation,
  elev_min_m: (D.heatmaps.elev_min_m ?? 0),
  elev_max_m: (D.heatmaps.elev_max_m ?? 400),
  flip_y:    false,
  rot_deg:   0
} : null);

let groundMesh;
function sampleZ(x_m, y_m) {
  if (!TERR) return 0;
  const G = TERR.grid;
  const GS = G.length;
  const cell = MAP_SIZE_M / GS;
  let ix = Math.floor(x_m / cell);
  let iy = Math.floor(y_m / cell);
  ix = Math.max(0, Math.min(GS - 1, ix));
  iy = Math.max(0, Math.min(GS - 1, iy));
  const v = G[iy][ix];
  return TERR.elev_min_m + v * (TERR.elev_max_m - TERR.elev_min_m);
}

if (TERR) {
  const GS = TERR.grid.length;
  const tgGeo = new THREE.PlaneGeometry(MAP_SIZE_M, MAP_SIZE_M, GS - 1, GS - 1);
  // PlaneGeometry default orientation: XY plane.  Displace local Z by elevation,
  // then rotate -π/2 about X so local-Z becomes world-Y (height).
  const pos = tgGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    // local x/y of the vertex (centered) → world x/y (uncentered)
    const lx = pos.getX(i) + MAP_SIZE_M / 2;
    const ly = pos.getY(i) + MAP_SIZE_M / 2;
    // honour optional flip / rotation hints supplied with the DEM
    let mx = lx, my = ly;
    if (TERR.rot_deg === 90)  { mx = ly; my = MAP_SIZE_M - lx; }
    if (TERR.rot_deg === 180) { mx = MAP_SIZE_M - lx; my = MAP_SIZE_M - ly; }
    if (TERR.rot_deg === 270) { mx = MAP_SIZE_M - ly; my = lx; }
    if (TERR.flip_y) my = MAP_SIZE_M - my;
    const z = sampleZ(mx, my);
    pos.setZ(i, z);
  }
  pos.needsUpdate = true;
  tgGeo.computeVertexNormals();

  // Vertex colors: elevation ramp (low = green-tan, high = brown-white)
  const colors = new Float32Array(pos.count * 3);
  const minZ = TERR.elev_min_m, maxZ = TERR.elev_max_m, span = Math.max(1, maxZ - minZ);
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    const t = (z - minZ) / span;
    let r, g, b;
    if (t < 0.3)      { r = 0.30 + t*0.30; g = 0.45 + t*0.30; b = 0.20; }
    else if (t < 0.6) { const u = (t-0.3)/0.3; r = 0.40 + u*0.30; g = 0.55 - u*0.15; b = 0.20 + u*0.10; }
    else              { const u = (t-0.6)/0.4; r = 0.65 + u*0.25; g = 0.40 + u*0.40; b = 0.25 + u*0.50; }
    colors[i*3+0] = r; colors[i*3+1] = g; colors[i*3+2] = b;
  }
  tgGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const tgMat = new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide
  });
  groundMesh = new THREE.Mesh(tgGeo, tgMat);
  groundMesh.rotation.x = -Math.PI / 2;
  groundMesh.position.set(MAP_SIZE_M / 2, 0, MAP_SIZE_M / 2);
  scene.add(groundMesh);
} else {
  // Fallback: flat textured plane (legacy behaviour)
  const groundGeo = new THREE.PlaneGeometry(MAP_SIZE_M, MAP_SIZE_M);
  const groundMat = new THREE.MeshBasicMaterial({
    color: 0x8fbc8f, transparent: true, opacity: 0.35, side: THREE.DoubleSide
  });
  groundMesh = new THREE.Mesh(groundGeo, groundMat);
  groundMesh.rotation.x = -Math.PI/2;
  groundMesh.position.set(MAP_SIZE_M/2, -2, MAP_SIZE_M/2);
  scene.add(groundMesh);

  const mapTexLoader = new THREE.TextureLoader();
  const mapUrls = [
    "https://raw.githubusercontent.com/pubg/api-assets/master/Assets/Maps/Erangel_Main_Low_Res.png",
    "https://raw.githubusercontent.com/pubg/api-assets/master/Assets/Maps/Erangel_Main_High_Res.png"
  ];
  function tryLoadMap(urls, idx) {
    if (idx >= urls.length) return;
    mapTexLoader.load(urls[idx],
      function(tex) {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        groundMat.map = tex;
        groundMat.color.set(0xffffff);
        groundMat.opacity = 0.7;
        groundMat.needsUpdate = true;
      },
      undefined,
      function() { tryLoadMap(urls, idx+1); }
    );
  }
  tryLoadMap(mapUrls, 0);
}

// 그리드 (맵 위에 약하게)
const grid=new THREE.GridHelper(MAP_SIZE_M, 16, 0x999999, 0xbbbbbb);
grid.position.set(MAP_SIZE_M/2, -1, MAP_SIZE_M/2);
grid.material.transparent=true;
grid.material.opacity=0.15;
scene.add(grid);

// === GROUPS ===
const nodeGrp=new THREE.Group(); scene.add(nodeGrp);
const allyGrp=new THREE.Group(); scene.add(allyGrp);
const encGrp=new THREE.Group();  scene.add(encGrp);

// === DUAL ZONE CYLINDERS (3D walls) ===
// Safe zone = 흰 cylinder (목표, 다음 안전 지역)
const safeZoneGeo = new THREE.CylinderGeometry(1, 1, 800, 64, 1, true);
const safeZoneMat = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.22, side: THREE.DoubleSide
});
const safeZoneMesh = new THREE.Mesh(safeZoneGeo, safeZoneMat);
scene.add(safeZoneMesh);

// Poison zone = 파란 내벽 cylinder (현재 데미지 경계, 닫혀오는 벽 효과)
const poisonWallGeo = new THREE.CylinderGeometry(1, 1, 1200, 64, 1, true);
const poisonWallMat = new THREE.MeshBasicMaterial({
  color: 0x3b82f6, transparent: true, opacity: 0.18, side: THREE.BackSide,
  depthWrite: false
});
const poisonFillMesh = new THREE.Mesh(poisonWallGeo, poisonWallMat);
scene.add(poisonFillMesh);

// Poison ring marker on the ground (thin ring at ground level for orientation)
const poisonRingGeo = new THREE.RingGeometry(0.985, 1, 128);
const poisonRingMat = new THREE.MeshBasicMaterial({
  color: 0x3b82f6, transparent: true, opacity: 0.6, side: THREE.DoubleSide
});
const poisonRingMesh = new THREE.Mesh(poisonRingGeo, poisonRingMat);
poisonRingMesh.rotation.x = -Math.PI/2;
scene.add(poisonRingMesh);

// === HEATMAPS ===
const HM=D.heatmaps;
const hmMeshes={};
if(HM){
  const GS=HM.grid_size, cellM=HM.cell_size_m, mapSz=GS*cellM;
  const schemes={
    elevation:v=>{if(v<.3)return[.15+v,.35+v*.6,.12];if(v<.6){const t=(v-.3)/.3;return[.24+t*.4,.55-t*.1,.12+t*.05]};const t=(v-.6)/.4;return[.64+t*.25,.45+t*.35,.17+t*.5]},
    density:v=>{if(v<.25){const t=v/.25;return[.1,.1,.2+t*.5]};if(v<.5){const t=(v-.25)/.25;return[.1,t*.6,.7-t*.1]};if(v<.75){const t=(v-.5)/.25;return[t*.8,.6+t*.2,.6-t*.5]};const t=(v-.75)/.25;return[.8+t*.2,.8+t*.2,t*.7]},
    combat:v=>{if(v<.33){const t=v/.33;return[.3+t*.4,.05,.05]};if(v<.66){const t=(v-.33)/.33;return[.7+t*.2,.05+t*.4,.05]};const t=(v-.66)/.34;return[.9+t*.1,.45+t*.45,.05+t*.4]}
  };
  ["elevation","density","combat"].forEach(type=>{
    const g=HM[type], cv=document.createElement("canvas");
    cv.width=GS; cv.height=GS;
    const c2=cv.getContext("2d"), img=c2.createImageData(GS,GS), fn=schemes[type];
    for(let y=0;y<GS;y++)for(let x=0;x<GS;x++){
      const v=g[y][x],[r,gb,b]=fn(v), i=(y*GS+x)*4;
      img.data[i]=r*255|0; img.data[i+1]=gb*255|0; img.data[i+2]=b*255|0;
      img.data[i+3]=v>.001?(50+v*140)|0:0;
    }
    c2.putImageData(img,0,0);
    const tex=new THREE.CanvasTexture(cv);
    tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;
    const geo=new THREE.PlaneGeometry(mapSz, mapSz);
    const mat=new THREE.MeshBasicMaterial({map:tex, transparent:true, opacity:0.55, side:THREE.DoubleSide, depthWrite:false});
    const mesh=new THREE.Mesh(geo, mat);
    mesh.rotation.x=-Math.PI/2; mesh.position.set(mapSz/2, 1, mapSz/2);
    mesh.visible=false; scene.add(mesh); hmMeshes[type]=mesh;
  });
}
document.querySelectorAll(".hm-btn").forEach(b=>b.addEventListener("click",function(){
  document.querySelectorAll(".hm-btn").forEach(x=>x.classList.remove("active"));
  this.classList.add("active");
  const t=this.dataset.hm;
  Object.values(hmMeshes).forEach(m=>m.visible=false);
  if(t!=="none"&&hmMeshes[t]){hmMeshes[t].visible=true;
    const r=document.getElementById("hm-range");
    r.textContent=t==="elevation"?HM.elev_min_m+"m ~ "+HM.elev_max_m+"m":t==="density"?"Player freq":"Damage intensity";
  } else document.getElementById("hm-range").textContent="";
}));

// === NODE CLICK (drag vs click 구분) ===
const ray=new THREE.Raycaster();
ray.params.Points={threshold:40};
const mouse=new THREE.Vector2();
let mouseDownPos={x:0,y:0};

renderer.domElement.addEventListener("mousedown",e=>{
  mouseDownPos.x=e.clientX; mouseDownPos.y=e.clientY;
});
renderer.domElement.addEventListener("mouseup",e=>{
  const dx=e.clientX-mouseDownPos.x, dy=e.clientY-mouseDownPos.y;
  const dist=Math.sqrt(dx*dx+dy*dy);
  // 3px 미만 이동 = 클릭, 그 이상 = 드래그
  if(dist>3) return;

  mouse.x=(e.clientX/renderer.domElement.clientWidth)*2-1;
  mouse.y=-(e.clientY/renderer.domElement.clientHeight)*2+1;
  ray.setFromCamera(mouse, camera);
  const hits=ray.intersectObjects(nodeGrp.children, false);
  if(hits.length){
    selNode=hits[0].object.userData.pid;
    showInfo(selNode);
  } else {
    selNode=-1;
    document.getElementById("info").style.display="none";
  }
});

function showInfo(pid){
  const fr=interp(prog);
  const n=fr.nodes[pid];
  if(!n||!n.alive) return;
  const p=document.getElementById("info");
  p.style.display="block";
  document.getElementById("ni-title").textContent="Player #"+pid;
  document.getElementById("ni-aid").textContent=n.aid;
  document.getElementById("ni-team").textContent="Team "+n.t+" ("+n.tid+")";
  const hpEl=document.getElementById("ni-hp");
  hpEl.textContent=Math.round(n.hp*100)+"%";
  hpEl.style.color=n.hp>.5?"var(--green)":n.hp>.2?"var(--warn)":"var(--red)";
  document.getElementById("ni-pos").textContent=Math.round(n.x)+"m, "+Math.round(n.y)+"m";
  document.getElementById("ni-alt").textContent=Math.round(n.z)+"m";
  // safe zone info
  document.getElementById("ni-sd").textContent=Math.round(n.sd)+"m";
  const isEl=document.getElementById("ni-is");
  isEl.textContent=n.is?"Yes":"No"; isEl.style.color=n.is?"var(--green)":"var(--warn)";
  // poison zone info
  document.getElementById("ni-pd").textContent=Math.round(n.pd)+"m";
  const ipEl=document.getElementById("ni-ip");
  ipEl.textContent=n.ip?"Safe":"DAMAGE"; ipEl.style.color=n.ip?"var(--green)":"var(--red)";
  // vehicle + combat
  document.getElementById("ni-veh").textContent=n.iv?"Vehicle ("+Math.round(n.vs)+"m/s)":"On Foot";
  document.getElementById("ni-dd").textContent=n.dd>0?n.dd.toFixed(1):"0";
  document.getElementById("ni-dt").textContent=n.dt>0?n.dt.toFixed(1):"0";
  const combat=n.dd>0||n.dt>0;
  const st=document.getElementById("ni-st");
  st.textContent=combat?"IN COMBAT":"Idle"; st.style.color=combat?"var(--red)":"var(--muted)";
}

// === CONTROLS ===
const bp=document.getElementById("bp"), tl=document.getElementById("timeline");
bp.addEventListener("click",()=>{playing=!playing; bp.textContent=playing?"\u23F8":"\u25B6"; bp.classList.toggle("on",playing)});
tl.max=(NF-1)*100;
tl.addEventListener("input",()=>{prog=parseFloat(tl.value)/100});
document.querySelectorAll(".sb").forEach(b=>b.addEventListener("click",function(){
  spd=parseFloat(this.dataset.s); document.getElementById("sp").textContent=spd+"x";
  document.querySelectorAll(".sb").forEach(x=>x.classList.remove("on")); this.classList.add("on");
}));

// === INTERP ===
function lerp(a,b,t){return a+(b-a)*t}
function interp(p){
  const i=Math.floor(p), f=p-i;
  const f0=F[Math.min(i,NF-1)], f1=F[Math.min(i+1,NF-1)];
  if(f<.001||i>=NF-1)return f0;
  const NP=D.total_players, nodes=[];
  for(let j=0;j<NP;j++){
    const a=f0.nodes[j], b=f1.nodes[j];
    if(a.alive&&b.alive){
      nodes.push({pid:j,alive:1,
        x:lerp(a.x,b.x,f),y:lerp(a.y,b.y,f),z:lerp(a.z,b.z,f),
        hp:lerp(a.hp,b.hp,f),
        sd:lerp(a.sd,b.sd,f),sb:lerp(a.sb,b.sb,f),is:f<.5?a.is:b.is,
        pd:lerp(a.pd,b.pd,f),pb:lerp(a.pb,b.pb,f),ip:f<.5?a.ip:b.ip,
        vs:lerp(a.vs,b.vs,f),iv:f<.5?a.iv:b.iv,
        dd:lerp(a.dd,b.dd,f),dt:lerp(a.dt,b.dt,f),
        t:a.t,aid:a.aid,tid:a.tid})
    } else if(a.alive&&!b.alive){
      nodes.push({...a,alive:1,hp:a.hp*(1-f)})
    } else if(!a.alive&&b.alive){
      nodes.push({...b,alive:1,hp:b.hp*f})
    } else {
      nodes.push({pid:j,alive:0})
    }
  }
  return{t:lerp(f0.t,f1.t,f),nodes,
    ally:f<.5?f0.ally:f1.ally, enc:f<.5?f0.enc:f1.enc,
    zone:{
      scx:lerp(f0.zone.scx,f1.zone.scx,f), scy:lerp(f0.zone.scy,f1.zone.scy,f),
      sr:lerp(f0.zone.sr,f1.zone.sr,f),
      pcx:lerp(f0.zone.pcx,f1.zone.pcx,f), pcy:lerp(f0.zone.pcy,f1.zone.pcy,f),
      pr:lerp(f0.zone.pr,f1.zone.pr,f),
      alive:Math.round(lerp(f0.zone.alive,f1.zone.alive,f))
    }}
}

// === COLOR UTIL ===
function hex2c(h){return new THREE.Color(parseInt(h.slice(1,3),16)/255, parseInt(h.slice(3,5),16)/255, parseInt(h.slice(5,7),16)/255)}
const sphereG=new THREE.SphereGeometry(1,12,8);
const ringG=new THREE.RingGeometry(.85,1,16);

// z축 → Three.js Y 변환 (클램프 포함)
function zToY(z_m) {
  const clamped = Math.max(Z_CLAMP_MIN, Math.min(Z_CLAMP_MAX, z_m));
  return (clamped - Z_MIN) * Z_SCALE;
}

// === RENDER ===
function updateScene(fr){
  const nd=fr.nodes, NP=D.total_players, time=performance.now()*.003;

  const aliveNodes=[];
  for(let i=0;i<NP;i++){
    if(nd[i].alive&&nd[i].hp>0.001){
      aliveNodes.push(nd[i]);
    }
  }
  const n=aliveNodes.length;

  // nodes
  while(nodeGrp.children.length>n) nodeGrp.remove(nodeGrp.children[nodeGrp.children.length-1]);
  while(nodeGrp.children.length<n){
    const mat=new THREE.MeshPhongMaterial({transparent:true, shininess:40});
    const m=new THREE.Mesh(sphereG, mat);
    const rm=new THREE.MeshBasicMaterial({color:0xff3333, transparent:true, opacity:0, side:THREE.DoubleSide});
    const ring=new THREE.Mesh(ringG, rm); ring.name="cr"; m.add(ring);
    nodeGrp.add(m);
  }
  let cc=0;
  for(let i=0;i<n;i++){
    const p=aliveNodes[i], m=nodeGrp.children[i], col=hex2c(TC[p.t%TC.length]);
    // z축: 교정된 스케일 사용
    m.position.set(p.x, sampleZ(p.x, p.y) + 25, p.y);
    const sz=14+p.hp*18; m.scale.setScalar(sz);
    m.material.color.copy(col);
    m.material.opacity=.5+p.hp*.5;
    m.material.emissive.copy(col).multiplyScalar(.08);
    m.userData.pid=p.pid;
    const ring=m.getObjectByName("cr");
    const combat=p.dd>0||p.dt>0;
    if(combat){
      cc++;
      ring.material.opacity=.5+.3*Math.sin(time*3+i);
      ring.scale.setScalar(2.2+Math.sin(time*2)*.3);
      ring.lookAt(camera.position);
      m.material.emissive.set(.35,.06,.06);
    } else {
      ring.material.opacity=0;
    }
    // poison 밖 = 데미지 받는 중 (빨간 글로우)
    if(!p.ip) m.material.emissive.set(.3,.04,.04);
    // 선택된 노드 하이라이트
    if(p.pid===selNode){m.material.emissive.set(.15,.25,.5); m.scale.multiplyScalar(1.3)}
  }

  // ally 엣지 (팀 색상, 높은 불투명도)
  allyGrp.children.forEach(c=>{c.geometry.dispose(); c.material.dispose()});
  allyGrp.clear();
  const ap=[], ac=[];
  for(const[sp,dp]of fr.ally){
    const sn=nd[sp], dn=nd[dp];
    if(!sn.alive||!dn.alive||sn.hp<.001||dn.hp<.001) continue;
    const col=hex2c(TC[sn.t%TC.length]);
    ap.push(new THREE.Vector3(sn.x,sampleZ(sn.x,sn.y)+25,sn.y), new THREE.Vector3(dn.x,sampleZ(dn.x,dn.y)+25,dn.y));
    ac.push(col.r,col.g,col.b, col.r,col.g,col.b);
  }
  if(ap.length){
    const g=new THREE.BufferGeometry().setFromPoints(ap);
    g.setAttribute("color",new THREE.Float32BufferAttribute(ac,3));
    allyGrp.add(new THREE.LineSegments(g,
      new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:0.75, linewidth:1})));
  }

  // encounter 엣지 (빨간색, 거리 기반 투명도)
  encGrp.children.forEach(c=>{c.geometry.dispose(); c.material.dispose()});
  encGrp.clear();
  const ep=[], ec2=[];
  const maxD=fr.zone.pr>0?fr.zone.pr*.4:fr.zone.sr*.4;
  for(const[sp,dp,dist]of fr.enc){
    const sn=nd[sp], dn=nd[dp];
    if(!sn.alive||!dn.alive||sn.hp<.001||dn.hp<.001) continue;
    const a=Math.max(.08,.5*(1-dist/Math.max(maxD,100)));
    ep.push(new THREE.Vector3(sn.x,sampleZ(sn.x,sn.y)+25,sn.y), new THREE.Vector3(dn.x,sampleZ(dn.x,dn.y)+25,dn.y));
    ec2.push(.85*a,.12*a,.12*a, .85*a,.12*a,.12*a);
  }
  if(ep.length){
    const g=new THREE.BufferGeometry().setFromPoints(ep);
    g.setAttribute("color",new THREE.Float32BufferAttribute(ec2,3));
    encGrp.add(new THREE.LineSegments(g,
      new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:0.7, linewidth:1})));
  }

  // === DUAL ZONE UPDATE (cylinders) ===
  // Safe zone (흰 cylinder, 목표) — y=400 keeps the 800m-tall cylinder centred above terrain
  safeZoneMesh.scale.set(fr.zone.sr, 1, fr.zone.sr);
  safeZoneMesh.position.set(fr.zone.scx, 400, fr.zone.scy);

  // Poison zone (파란 내벽 cylinder + 지면 ring)
  if(fr.zone.pr > 0) {
    poisonFillMesh.visible = true;
    poisonRingMesh.visible = true;
    poisonFillMesh.scale.set(fr.zone.pr, 1, fr.zone.pr);
    poisonFillMesh.position.set(fr.zone.pcx, 600, fr.zone.pcy);
    poisonRingMesh.scale.set(fr.zone.pr, fr.zone.pr, 1);
    poisonRingMesh.position.set(fr.zone.pcx, sampleZ(fr.zone.pcx, fr.zone.pcy) + 4, fr.zone.pcy);
  } else {
    // 자기장 미생성
    poisonFillMesh.visible = false;
    poisonRingMesh.visible = false;
  }

  // HUD
  document.getElementById("ht").textContent=Math.round(fr.t)+"s";
  document.getElementById("ha").textContent=fr.zone.alive+" / "+D.total_players;
  document.getElementById("hz").textContent=Math.round(fr.zone.pr)+"m";
  document.getElementById("hsr").textContent=Math.round(fr.zone.sr)+"m";
  document.getElementById("htm").textContent=new Set(aliveNodes.map(p=>p.t)).size;
  document.getElementById("hc").textContent=cc;
  if(selNode>=0) showInfo(selNode);
}

// === LOOP ===
function tick(ts){
  if(!lt) lt=ts;
  const dt=(ts-lt)/1000; lt=ts;
  if(playing){prog+=dt*spd; if(prog>=NF-1) prog=0; tl.value=Math.round(prog*100)}
  updateScene(interp(prog));
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
</script>
</body></html>"""


if __name__=="__main__":
    if len(sys.argv)>1: fp=sys.argv[1]
    else:
        gd="data/graphs"
        if not os.path.exists(gd): print(f"Error: {gd}/ 없음"); sys.exit(1)
        fs=sorted(f for f in os.listdir(gd) if f.endswith(".pt"))
        if not fs: print(f"Error: .pt 없음"); sys.exit(1)
        fp=os.path.join(gd, fs[0])
    print(f"로드: {fp}")
    graphs, st, meta=load_graph_data(fp)
    print(f"매치: {meta['match_id'][:20]}..., 스냅샷: {len(graphs)}, 플레이어: {meta['total_players']}")
    print(f"z 범위: {meta.get('z_min','?')}~{meta.get('z_max','?')}m")
    print("JSON 변환...")
    jd=graphs_to_json(graphs, st, meta)
    js=json.dumps(jd, separators=(",",":"))
    print(f"JSON: {len(js)//1024} KB")
    html=HTML.replace("__JSON_DATA__",js).replace("__TOTAL__",str(meta["total_players"]))
    os.makedirs("data",exist_ok=True)
    out="data/match_graph_3d.html"
    with open(out,"w",encoding="utf-8") as f: f.write(html)
    print(f"저장: {out}")
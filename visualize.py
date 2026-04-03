"""
PUBG Match Graph Visualizer — 3D (OrbitControls)
==================================================
Three.js r128 + OrbitControls (unpkg CDN) 기반 3D 시각화.
3d-force-graph (vasturiano) 디자인 참고.

카메라: THREE.OrbitControls (검증된 라이브러리)
- 좌클릭 드래그: 회전
- 우클릭 드래그 / 휠 클릭 드래그: 패닝
- 스크롤: 줌

사용법:
  python visualize.py
  python visualize.py data/graphs/match_0046ac50.pt
"""

import sys, os, json, torch, numpy as np


def load_graph_data(filepath):
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    return data["graphs"], data["snapshot_times"], data["meta"]


def graphs_to_json(graphs, snapshot_times, meta):
    # 글로벌 pid 기반: 노드 배열 크기 고정 (총 플레이어 수)
    total_p = meta["total_players"]

    frames = []
    for g, t in zip(graphs, snapshot_times):
        x = g["player"].x
        n = x.shape[0]
        team_idx = g["player"].team_idx.numpy().tolist()
        global_pid = g["player"].global_pid.numpy().tolist()
        account_ids = g["player"].account_ids
        team_ids = g["player"].team_ids

        # pid → local index 매핑
        pid_to_local = {global_pid[i]: i for i in range(n)}

        # 고정 크기 노드 배열 (pid 순서)
        nodes = []
        for pid in range(total_p):
            if pid in pid_to_local:
                li = pid_to_local[pid]
                nodes.append({
                    "pid": pid, "alive": 1,
                    "x": round(x[li, 0].item(), 1), "y": round(x[li, 1].item(), 1),
                    "z": round(x[li, 2].item(), 1), "hp": round(x[li, 3].item(), 3),
                    "zd": round(x[li, 4].item(), 0), "bd": round(x[li, 5].item(), 0),
                    "iz": int(x[li, 6].item() > 0.5), "vs": round(x[li, 7].item(), 1),
                    "iv": int(x[li, 8].item() > 0.5),
                    "dd": round(x[li, 9].item(), 1), "dt": round(x[li, 10].item(), 1),
                    "t": team_idx[li],
                    "aid": account_ids[li][-8:] if account_ids[li] else "",
                    "tid": str(team_ids[li])[-6:] if team_ids[li] else "",
                })
            else:
                nodes.append({"pid": pid, "alive": 0})

        # 엣지: local index → global pid 변환
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

        zv = g["zone"].x[0]
        zone = {"cx": round(zv[0].item(),1), "cy": round(zv[1].item(),1),
                "r": round(zv[2].item(),1), "alive": int(zv[6].item())}
        frames.append({"t": round(t,1), "nodes": nodes,
                       "ally": ally_edges, "enc": enc_edges, "zone": zone})

    return {"match_id": meta["match_id"], "total_players": total_p,
            "total_teams": len(meta["team_rank"]), "frames": frames,
            "heatmaps": meta.get("heatmaps", None)}


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>PUBG 3D Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
:root{--bg:#0b0b1a;--panel:rgba(12,12,30,.88);--border:#1e1e3a;--text:#c8c8e0;--accent:#4488ff;--red:#ff4455}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',Roboto,monospace;overflow:hidden}
canvas{display:block}

.panel{position:absolute;background:var(--panel);border:1px solid var(--border);
       border-radius:10px;padding:14px 18px;backdrop-filter:blur(10px);font-size:12px;line-height:1.8}

#hud{top:14px;left:14px;pointer-events:none}
#hud b{color:var(--accent);margin-right:4px}

#info{top:14px;right:14px;min-width:240px;display:none}
#info h3{color:var(--accent);font-size:14px;margin-bottom:6px;border-bottom:1px solid var(--border);padding-bottom:6px}
.ir{display:flex;justify-content:space-between;padding:1px 0}
.il{color:#888}.iv{color:#e0e0f0}.ic{color:var(--red);font-weight:600}

#legend{bottom:56px;right:14px;font-size:11px}
.lg{display:flex;align-items:center;gap:8px;margin:3px 0}
.ll{width:20px;height:3px;border-radius:2px}
.ld{width:9px;height:9px;border-radius:50%;border:1px solid rgba(255,255,255,.15)}
.hm-btn{background:var(--panel);border:1px solid var(--border);color:#888;border-radius:4px;
        padding:3px 8px;cursor:pointer;font-size:11px;margin:2px 0;font-family:inherit;
        transition:all .2s}
.hm-btn:hover,.hm-btn.active{color:#fff;border-color:var(--accent)}

#controls{position:absolute;bottom:0;left:0;right:0;display:flex;align-items:center;
          gap:10px;padding:10px 16px;background:rgba(10,10,25,.92);border-top:1px solid var(--border)}
#controls button{background:#14142a;color:#aaa;border:1px solid var(--border);border-radius:5px;
                 padding:6px 14px;cursor:pointer;font-size:12px;font-family:inherit;transition:all .15s}
#controls button:hover{background:#1e1e3e;color:#fff}
#controls button.on{background:#1a3366;border-color:var(--accent);color:#fff}
#timeline{flex:1;height:5px;-webkit-appearance:none;appearance:none;
          background:#14142a;border-radius:3px;outline:none;cursor:pointer}
#timeline::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
                                background:var(--accent);border-radius:50%;cursor:pointer;
                                box-shadow:0 0 6px rgba(68,136,255,.4)}
#sp{width:50px;text-align:center;font-size:11px;color:#666}
#tip{position:absolute;bottom:56px;left:14px;font-size:11px;color:#444;
     background:var(--panel);padding:6px 10px;border-radius:6px;border:1px solid var(--border)}
</style>
</head><body>

<div class="panel" id="hud">
  <div><b>Time</b><span id="ht">0s</span> / <span id="htot">0s</span></div>
  <div><b>Alive</b><span id="ha">0</span> / __TOTAL__</div>
  <div><b>Zone R</b><span id="hz">0</span>m</div>
  <div><b>Teams</b><span id="htm">0</span></div>
  <div><b>Combat</b><span id="hc">0</span></div>
</div>

<div class="panel" id="info">
  <h3 id="ni-title">Player</h3>
  <div class="ir"><span class="il">Account</span><span class="iv" id="ni-aid">-</span></div>
  <div class="ir"><span class="il">Team</span><span class="iv" id="ni-team">-</span></div>
  <div class="ir"><span class="il">HP</span><span class="iv" id="ni-hp">-</span></div>
  <div class="ir"><span class="il">Position</span><span class="iv" id="ni-pos">-</span></div>
  <div class="ir"><span class="il">Altitude</span><span class="iv" id="ni-alt">-</span></div>
  <div class="ir"><span class="il">Zone Dist</span><span class="iv" id="ni-zd">-</span></div>
  <div class="ir"><span class="il">Inside Zone</span><span class="iv" id="ni-iz">-</span></div>
  <div class="ir"><span class="il">Vehicle</span><span class="iv" id="ni-veh">-</span></div>
  <div style="height:1px;background:var(--border);margin:6px 0"></div>
  <div class="ir"><span class="il">Dmg Dealt</span><span class="ic" id="ni-dd">-</span></div>
  <div class="ir"><span class="il">Dmg Taken</span><span class="ic" id="ni-dt">-</span></div>
  <div class="ir"><span class="il">Status</span><span class="iv" id="ni-st">-</span></div>
</div>

<div class="panel" id="legend">
  <div class="lg"><div class="ll" style="background:rgba(80,160,255,.7)"></div>Ally</div>
  <div class="lg"><div class="ll" style="background:rgba(255,70,70,.5)"></div>Encounter</div>
  <div class="lg"><div class="ld" style="background:var(--accent)"></div>Player</div>
  <div class="lg"><div class="ld" style="background:var(--red);box-shadow:0 0 6px var(--red)"></div>In Combat</div>
  <div style="height:1px;background:var(--border);margin:6px 0"></div>
  <div style="color:#666;margin-bottom:4px">Heatmap</div>
  <button class="hm-btn" data-hm="elevation">Elevation</button>
  <button class="hm-btn" data-hm="density">Density</button>
  <button class="hm-btn" data-hm="combat">Combat</button>
  <button class="hm-btn active" data-hm="none">Off</button>
  <div id="hm-range" style="font-size:10px;color:#555;margin-top:4px"></div>
</div>

<div id="tip">Left: rotate | Middle/Right: pan | Scroll: zoom | Click node (paused): details</div>

<div id="controls">
  <button id="bp" class="on">⏸</button>
  <button class="sb" data-s="0.5">0.5×</button>
  <button class="sb on" data-s="1">1×</button>
  <button class="sb" data-s="2">2×</button>
  <button class="sb" data-s="4">4×</button>
  <input type="range" id="timeline" min="0" max="1000" value="0">
  <span id="sp">1×</span>
</div>

<script>
const D=__JSON_DATA__;
const TC=["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
"#f032e6","#bfef45","#fabebe","#469990","#e6beff","#9A6324","#fffac8","#800000",
"#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ff6961","#77dd77","#fdfd96",
"#84b6f4","#fdcae1","#c1e1c1","#b39eb5","#fbe7c6","#a0c4ff","#caffbf"];

const F=D.frames, NF=F.length;
let playing=true, spd=1, prog=0, lt=0, selNode=-1;
document.getElementById("htot").textContent=Math.round(F[NF-1].t)+"s";

// === SCENE ===
const W=innerWidth, H=innerHeight-40;
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0b0b1a);
scene.fog=new THREE.FogExp2(0x0b0b1a, 0.00006);

const camera=new THREE.PerspectiveCamera(55, W/H, 10, 80000);
const renderer=new THREE.WebGLRenderer({antialias:true, alpha:false});
renderer.setSize(W,H);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.prepend(renderer.domElement);

// === OrbitControls (라이브러리 내장) ===
const controls=new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping=true;
controls.dampingFactor=0.12;
controls.rotateSpeed=0.8;
controls.panSpeed=0.6;
controls.zoomSpeed=1.2;
controls.mouseButtons={LEFT:THREE.MOUSE.ROTATE, MIDDLE:THREE.MOUSE.PAN, RIGHT:THREE.MOUSE.PAN};
controls.minDistance=100;
controls.maxDistance=25000;

// 초기 카메라
if(NF>0){
  const z0=F[0].zone;
  controls.target.set(z0.cx, 0, z0.cy);
  camera.position.set(z0.cx+z0.r*0.8, z0.r*1.2, z0.cy+z0.r*0.8);
  controls.update();
}

addEventListener("resize",()=>{
  const w=innerWidth, h=innerHeight-40;
  camera.aspect=w/h; camera.updateProjectionMatrix();
  renderer.setSize(w,h);
});

// === LIGHTS ===
scene.add(new THREE.AmbientLight(0x334466, 0.5));
const dl=new THREE.DirectionalLight(0xffffff, 0.7);
dl.position.set(5000,8000,5000); scene.add(dl);
const hl=new THREE.HemisphereLight(0x2244aa, 0x111122, 0.3);
scene.add(hl);

// === GRID ===
const grid=new THREE.GridHelper(8160, 16, 0x161630, 0x0e0e24);
grid.position.set(4080,0,4080); scene.add(grid);

// === GROUPS ===
const nodeGrp=new THREE.Group(); scene.add(nodeGrp);
const allyGrp=new THREE.Group(); scene.add(allyGrp);
const encGrp=new THREE.Group();  scene.add(encGrp);

// zone ring
const zoneGeo=new THREE.RingGeometry(0.98,1,96);
const zoneMat=new THREE.MeshBasicMaterial({color:0x3388ff, transparent:true, opacity:0.3, side:THREE.DoubleSide});
const zoneMesh=new THREE.Mesh(zoneGeo, zoneMat);
zoneMesh.rotation.x=-Math.PI/2; scene.add(zoneMesh);

// === HEATMAPS ===
const HM=D.heatmaps;
const hmMeshes={};
if(HM){
  const GS=HM.grid_size, cellM=HM.cell_size_m, mapSz=GS*cellM;
  const schemes={
    elevation:v=>{if(v<.3)return[.05+v*.3,.15+v*.8,.05+v*.1];if(v<.6){const t=(v-.3)/.3;return[.14+t*.5,.39-t*.15,.08+t*.02]};const t=(v-.6)/.4;return[.64+t*.36,.24+t*.76,.1+t*.9]},
    density:v=>{if(v<.25){const t=v/.25;return[0,0,t*.6]};if(v<.5){const t=(v-.25)/.25;return[0,t*.7,.6-t*.1]};if(v<.75){const t=(v-.5)/.25;return[t*.9,.7+t*.2,.5-t*.5]};const t=(v-.75)/.25;return[.9+t*.1,.9+t*.1,t*.8]},
    combat:v=>{if(v<.33){const t=v/.33;return[t*.7,0,0]};if(v<.66){const t=(v-.33)/.33;return[.7+t*.3,t*.5,0]};const t=(v-.66)/.34;return[1,.5+t*.5,t*.5]}
  };
  ["elevation","density","combat"].forEach(type=>{
    const g=HM[type], cv=document.createElement("canvas");
    cv.width=GS; cv.height=GS;
    const c2=cv.getContext("2d"), img=c2.createImageData(GS,GS), fn=schemes[type];
    for(let y=0;y<GS;y++)for(let x=0;x<GS;x++){
      const v=g[y][x],[r,gb,b]=fn(v), i=(y*GS+x)*4;
      img.data[i]=r*255|0; img.data[i+1]=gb*255|0; img.data[i+2]=b*255|0;
      img.data[i+3]=v>.001?(40+v*160)|0:0;
    }
    c2.putImageData(img,0,0);
    const tex=new THREE.CanvasTexture(cv);
    tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;
    const geo=new THREE.PlaneGeometry(mapSz, mapSz);
    const mat=new THREE.MeshBasicMaterial({map:tex, transparent:true, opacity:0.65, side:THREE.DoubleSide, depthWrite:false});
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

// === NODE CLICK (raycaster) ===
const ray=new THREE.Raycaster();
ray.params.Points={threshold:40};
const mouse=new THREE.Vector2();
renderer.domElement.addEventListener("click",e=>{
  if(playing)return;
  mouse.x=(e.clientX/renderer.domElement.clientWidth)*2-1;
  mouse.y=-(e.clientY/renderer.domElement.clientHeight)*2+1;
  ray.setFromCamera(mouse, camera);
  const hits=ray.intersectObjects(nodeGrp.children, false);
  if(hits.length){selNode=hits[0].object.userData.pid; showInfo(selNode)}
  else{selNode=-1; document.getElementById("info").style.display="none"}
});
function showInfo(pid){
  const fr=interp(prog);
  const n=fr.nodes[pid];
  if(!n||!n.alive) return;
  const p=document.getElementById("info");
  p.style.display="block";
  document.getElementById("ni-aid").textContent=n.aid;
  document.getElementById("ni-team").textContent="Team "+n.t+" ("+n.tid+")";
  const hpEl=document.getElementById("ni-hp");
  hpEl.textContent=Math.round(n.hp*100)+"%";
  hpEl.style.color=n.hp>.5?"#4f4":n.hp>.2?"#fa0":"#f44";
  document.getElementById("ni-pos").textContent=Math.round(n.x)+"m, "+Math.round(n.y)+"m";
  document.getElementById("ni-alt").textContent=Math.round(n.z)+"m";
  document.getElementById("ni-zd").textContent=Math.round(n.zd)+"m";
  const izEl=document.getElementById("ni-iz");
  izEl.textContent=n.iz?"Yes":"NO"; izEl.style.color=n.iz?"#4f4":"#f44";
  document.getElementById("ni-veh").textContent=n.iv?"Vehicle ("+Math.round(n.vs)+"m/s)":"On Foot";
  document.getElementById("ni-dd").textContent=n.dd>0?n.dd.toFixed(1):"0";
  document.getElementById("ni-dt").textContent=n.dt>0?n.dt.toFixed(1):"0";
  const combat=n.dd>0||n.dt>0;
  const st=document.getElementById("ni-st");
  st.textContent=combat?"IN COMBAT":"Idle"; st.style.color=combat?"var(--red)":"#666";
  document.getElementById("ni-title").textContent="Player #"+pid;
}

// === CONTROLS ===
const bp=document.getElementById("bp"), tl=document.getElementById("timeline");
bp.addEventListener("click",()=>{playing=!playing; bp.textContent=playing?"⏸":"▶"; bp.classList.toggle("on",playing)});
tl.max=(NF-1)*100;
tl.addEventListener("input",()=>{prog=parseFloat(tl.value)/100});
document.querySelectorAll(".sb").forEach(b=>b.addEventListener("click",function(){
  spd=parseFloat(this.dataset.s); document.getElementById("sp").textContent=spd+"×";
  document.querySelectorAll(".sb").forEach(x=>x.classList.remove("on")); this.classList.add("on");
}));

// === INTERP (pid 기반: 배열 인덱스 = 플레이어 ID, 고정) ===
function lerp(a,b,t){return a+(b-a)*t}
function interp(p){
  const i=Math.floor(p), f=p-i;
  const f0=F[Math.min(i,NF-1)], f1=F[Math.min(i+1,NF-1)];
  if(f<.001||i>=NF-1)return f0;
  const NP=D.total_players, nodes=[];
  for(let j=0;j<NP;j++){
    const a=f0.nodes[j], b=f1.nodes[j];
    if(a.alive&&b.alive){
      // 양쪽 다 생존: 보간
      nodes.push({pid:j,alive:1,x:lerp(a.x,b.x,f),y:lerp(a.y,b.y,f),z:lerp(a.z,b.z,f),
        hp:lerp(a.hp,b.hp,f),zd:lerp(a.zd,b.zd,f),bd:lerp(a.bd,b.bd,f),
        iz:f<.5?a.iz:b.iz,vs:lerp(a.vs,b.vs,f),iv:f<.5?a.iv:b.iv,
        dd:lerp(a.dd,b.dd,f),dt:lerp(a.dt,b.dt,f),t:a.t,aid:a.aid,tid:a.tid})
    } else if(a.alive&&!b.alive){
      // 이번 구간에서 탈락: 페이드아웃
      nodes.push({...a,alive:1,hp:a.hp*(1-f)})
    } else if(!a.alive&&b.alive){
      // 재등장 (position 누락 후 복귀): 페이드인
      nodes.push({...b,alive:1,hp:b.hp*f})
    } else {
      nodes.push({pid:j,alive:0})
    }
  }
  return{t:lerp(f0.t,f1.t,f),nodes,
    ally:f<.5?f0.ally:f1.ally, enc:f<.5?f0.enc:f1.enc,
    zone:{cx:lerp(f0.zone.cx,f1.zone.cx,f),cy:lerp(f0.zone.cy,f1.zone.cy,f),
          r:lerp(f0.zone.r,f1.zone.r,f),alive:Math.round(lerp(f0.zone.alive,f1.zone.alive,f))}}
}

// === COLOR UTIL ===
function hex2c(h){return new THREE.Color(parseInt(h.slice(1,3),16)/255, parseInt(h.slice(3,5),16)/255, parseInt(h.slice(5,7),16)/255)}
const sphereG=new THREE.SphereGeometry(1,14,10);
const ringG=new THREE.RingGeometry(.85,1,20);

// === RENDER ===
function updateScene(fr){
  const nd=fr.nodes, NP=D.total_players, time=performance.now()*.003;

  // 생존 노드만 필터링 (렌더링용)
  const aliveNodes=[], aliveMap={};  // pid → aliveNodes index
  for(let i=0;i<NP;i++){
    if(nd[i].alive&&nd[i].hp>0.001){
      aliveMap[i]=aliveNodes.length;
      aliveNodes.push(nd[i]);
    }
  }
  const n=aliveNodes.length;

  // nodes
  while(nodeGrp.children.length>n) nodeGrp.remove(nodeGrp.children[nodeGrp.children.length-1]);
  while(nodeGrp.children.length<n){
    const mat=new THREE.MeshPhongMaterial({transparent:true, shininess:60});
    const m=new THREE.Mesh(sphereG, mat);
    const rm=new THREE.MeshBasicMaterial({color:0xff3333, transparent:true, opacity:0, side:THREE.DoubleSide});
    const ring=new THREE.Mesh(ringG, rm); ring.name="cr"; m.add(ring);
    nodeGrp.add(m);
  }
  let cc=0;
  for(let i=0;i<n;i++){
    const p=aliveNodes[i], m=nodeGrp.children[i], col=hex2c(TC[p.t%TC.length]);
    m.position.set(p.x, p.z*0.3, p.y);
    const sz=12+p.hp*22; m.scale.setScalar(sz);
    m.material.color.copy(col);
    m.material.opacity=.3+p.hp*.7;
    m.material.emissive.copy(col).multiplyScalar(.12);
    m.userData.pid=p.pid;
    const ring=m.getObjectByName("cr");
    const combat=p.dd>0||p.dt>0;
    if(combat){cc++; ring.material.opacity=.4+.3*Math.sin(time*3+i);
      ring.scale.setScalar(2.2+Math.sin(time*2)*.4); ring.lookAt(camera.position);
      m.material.emissive.set(.4,.08,.08)}
    else ring.material.opacity=0;
    if(!p.iz) m.material.emissive.set(.25,.04,.04);
    if(p.pid===selNode){m.material.emissive.set(.2,.35,.6); m.scale.multiplyScalar(1.25)}
  }

  // ally (pid 기반 엣지 → 생존 노드만 그리기)
  allyGrp.children.forEach(c=>{c.geometry.dispose(); c.material.dispose()});
  allyGrp.clear();
  const ap=[], ac=[];
  for(const[sp,dp]of fr.ally){
    const sn=nd[sp], dn=nd[dp];
    if(!sn.alive||!dn.alive||sn.hp<.001||dn.hp<.001) continue;
    const col=hex2c(TC[sn.t%TC.length]);
    ap.push(new THREE.Vector3(sn.x,sn.z*.3,sn.y), new THREE.Vector3(dn.x,dn.z*.3,dn.y));
    ac.push(col.r,col.g,col.b, col.r,col.g,col.b);
  }
  if(ap.length){
    const g=new THREE.BufferGeometry().setFromPoints(ap);
    g.setAttribute("color",new THREE.Float32BufferAttribute(ac,3));
    allyGrp.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:.5})));
  }

  // encounter (pid 기반)
  encGrp.children.forEach(c=>{c.geometry.dispose(); c.material.dispose()});
  encGrp.clear();
  const ep=[], ec2=[];
  const maxD=fr.zone.r*.5||1000;
  for(const[sp,dp,dist]of fr.enc){
    const sn=nd[sp], dn=nd[dp];
    if(!sn.alive||!dn.alive||sn.hp<.001||dn.hp<.001) continue;
    const a=Math.max(.03,.4*(1-dist/maxD));
    ep.push(new THREE.Vector3(sn.x,sn.z*.3,sn.y), new THREE.Vector3(dn.x,dn.z*.3,dn.y));
    ec2.push(a,.06,.06, a,.06,.06);
  }
  if(ep.length){
    const g=new THREE.BufferGeometry().setFromPoints(ep);
    g.setAttribute("color",new THREE.Float32BufferAttribute(ec2,3));
    encGrp.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:.6})));
  }

  // zone
  zoneMesh.scale.set(fr.zone.r, fr.zone.r, 1);
  zoneMesh.position.set(fr.zone.cx, 2, fr.zone.cy);

  // hud
  document.getElementById("ht").textContent=Math.round(fr.t)+"s";
  document.getElementById("ha").textContent=fr.zone.alive;
  document.getElementById("hz").textContent=Math.round(fr.zone.r);
  document.getElementById("htm").textContent=new Set(aliveNodes.map(p=>p.t)).size;
  document.getElementById("hc").textContent=cc;
  if(selNode>=0&&!playing) showInfo(selNode);
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
    print("JSON 변환...")
    jd=graphs_to_json(graphs, st, meta)
    js=json.dumps(jd, separators=(",",":"))
    print(f"JSON: {len(js)//1024} KB")
    html=HTML.replace("__JSON_DATA__",js).replace("__TOTAL__",str(meta["total_players"]))
    os.makedirs("data",exist_ok=True)
    out="data/match_graph_3d.html"
    with open(out,"w",encoding="utf-8") as f: f.write(html)
    print(f"저장: {out}")
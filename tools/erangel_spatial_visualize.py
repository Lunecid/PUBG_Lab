"""Render actual raster/telemetry/graph artifacts; no generated map imagery."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from scipy.stats import binned_statistic_2d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/erangel_alignment_2026-10-15'
ASSETS=ROOT/'data/spatial/erangel/source'
EXTENT=8160


def main():
    m=json.loads((OUT/'alignment_metrics.json').read_text())
    graph=json.loads((OUT/'graph.json').read_text())
    tg=json.loads((OUT/'terrain_graph.json').read_text())
    data=np.load(OUT/'telemetry_probe.npz');p=data['points']
    arrays=np.load(OUT/'alignment_arrays.npz');res=arrays['residual']
    rgb=Image.open(ASSETS/'official_notext.png').resize((1600,1600))
    h=np.asarray(Image.open(ASSETS/'height_2019.png').resize((1024,1024),Image.Resampling.NEAREST))
    topo=np.asarray(plt.get_cmap('terrain')(np.clip((h.astype(float)-32768)/20000,0,1)))
    topo[h<32768]=[.025,.13,.22,1]
    topo_img=Image.fromarray(np.uint8(topo[:,:,:3]*255))
    nodes=graph['nodes'];edges=graph['edges'];xy=np.array([[n['x_m'],n['y_m']] for n in nodes])
    seg=np.array([[xy[e['source']],xy[e['target']]] for e in edges])
    risk=np.array([e['known_change_buffer'] for e in edges]);support=np.array([e['match_support'] for e in edges])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'figure.facecolor':'#f8fafc'})
    def base(ax,im=rgb):
        ax.imshow(im,extent=[0,EXTENT,EXTENT,0]);ax.set_xlim(0,EXTENT);ax.set_ylim(EXTENT,0)
        ax.set_xlabel('x (m, east)');ax.set_ylabel('y (m, south)');ax.grid(alpha=.13,color='white')
    def circles(ax,label=False):
        for r in m['known_change_buffers']:
            ax.add_patch(Circle((r['x_m'],r['y_m']),r['radius_m'],fill=False,edgecolor='#fb923c',lw=1,ls='--'))
            if label:ax.text(r['x_m'],r['y_m'],r['name'].replace('_',' '),color='white',fontsize=7,ha='center',bbox={'facecolor':'#111827','alpha':.7,'pad':1,'edgecolor':'none'})
    fig,axs=plt.subplots(1,3,figsize=(19,7),layout='constrained')
    base(axs[0]);axs[0].scatter(p[::5,2],p[::5,3],s=.25,c='#22d3ee',alpha=.28,rasterized=True)
    axs[0].set_title('A  Official image + telemetry\n40 matches; 170,304 candidate positions')
    base(axs[1],topo_img);axs[1].set_title('B  Historical 16-bit heightmap\nFile last changed: 2019-12-19; raw height colors')
    land=(h>=32768).astype(float);axs[0].contour(np.linspace(0,EXTENT,1024),np.linspace(0,EXTENT,1024),land,levels=[.5],colors=['#f8fafc'],linewidths=.35,alpha=.65)
    base(axs[2]);sel=support>=3
    axs[2].add_collection(LineCollection(seg[sel&~risk],colors='#22d3ee',linewidths=.7,alpha=.85))
    axs[2].add_collection(LineCollection(seg[sel&risk],colors='#fb923c',linewidths=.7,alpha=.85));circles(axs[2])
    axs[2].set_title(f'C  Observed 50 m graph\n{len(nodes):,} nodes / {len(edges):,} directed edges')
    fig.suptitle('ERANGEL SPATIAL AUDIT  |  Coordinates align; map versions do not match',fontsize=17,fontweight='bold')
    fig.text(.5,.012,'White outline: height land proxy. Cyan/orange edges: outside/inside approximate change buffers. Panel C shows edges supported by 3+ matches; not a navmesh.',ha='center',fontsize=9)
    fig.savefig(OUT/'overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Zoom makes actual graph topology inspectable instead of hiding it in a full-map hairball.
    bounds=(3600,4900,2600,3900)
    fig,axs=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    txy=np.array([[n['x_m'],n['y_m']] for n in tg['nodes']]);tseg=np.array([[txy[e['source']],txy[e['target']]] for e in tg['edges']])
    grades=np.array([e['grade_proxy'] for e in tg['edges']])
    for ax in axs:base(ax);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[3],bounds[2])
    lc=LineCollection(tseg,cmap='magma',norm=plt.Normalize(0,.2),linewidths=.8,alpha=.9);lc.set_array(grades);axs[0].add_collection(lc)
    fig.colorbar(lc,ax=axs[0],shrink=.62,label='Height-derived grade proxy (not walkability)')
    axs[0].set_title('A  Terrain candidate: 4-neighbor 50 m grid\nLand proxy only; buildings and bridge decks absent')
    axs[1].add_collection(LineCollection(seg[support>=2],colors='#38bdf8',linewidths=1.3,alpha=.85))
    sizes=np.array([n['match_support'] for n in nodes]);axs[1].scatter(xy[:,0],xy[:,1],s=np.minimum(sizes*2,35),c='#fef08a',alpha=.9)
    axs[1].set_title('B  Observed transitions: 32 reference matches\nEdges shown: 2+ matches; dots: observed cells')
    for ax in axs:
        ax.text(4268,3297,'School',color='white',fontsize=12,bbox={'facecolor':'#111827','alpha':.7,'pad':2})
    fig.suptitle('Two graphs encode different evidence — neither certifies a collision-free route',fontsize=14,fontweight='bold')
    fig.savefig(OUT/'graph_detail.png',dpi=160,bbox_inches='tight');plt.close(fig)
    test=p[:,0]>=32;pred=arrays['predicted_player_z'];ii=np.where(test)[0][::3]
    fig,axs=plt.subplots(1,3,figsize=(17,5.7),layout='constrained')
    axs[0].scatter(pred[ii],p[ii,4],s=2,alpha=.17,color='#0284c7',rasterized=True);axs[0].plot([-5,200],[-5,200],c='#f97316',lw=1)
    axs[0].set(xlim=(-5,200),ylim=(-5,200),xlabel='Height-derived player-z proxy (m)',ylabel='Observed player z (m)',title='A  Eight held-out matches\nPlayer z is not surveyed ground height')
    for label,mask,col in [('All held-out positions',test,'#0284c7'),('Held-out spatial blocks',test&(arrays['spatial_block']==0)&~arrays['risk']&~arrays['water'],'#a855f7')]:
        v=np.sort(np.abs(res[mask]));axs[1].plot(v,np.linspace(0,1,len(v)),label=label,color=col)
    axs[1].set(xlim=(0,20),ylim=(0,1),xlabel='Absolute player-z residual (m)',ylabel='Cumulative share',title=f'B  Residual distribution\nMedian {m["fitted_heldout_all"]["median_abs_m"]:.2f} m; P90 {m["fitted_heldout_all"]["p90_abs_m"]:.2f} m')
    axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
    base(axs[2]);bin_edges=np.arange(0,EXTENT+100,100)
    stat,xe,ye,_=binned_statistic_2d(p[test,2],p[test,3],np.abs(res[test]),statistic='median',bins=[bin_edges,bin_edges])
    count,*_=binned_statistic_2d(p[test,2],p[test,3],np.abs(res[test]),statistic='count',bins=[bin_edges,bin_edges]);stat[count<5]=np.nan
    mesh=axs[2].pcolormesh(xe,ye,stat.T,cmap='inferno',vmin=0,vmax=15,alpha=.85);circles(axs[2])
    axs[2].set_title('C  Held-out cell residuals\n100 m cells, at least 5 observations');fig.colorbar(mesh,ax=axs[2],shrink=.7,label='Median absolute player-z residual (m)')
    fig.suptitle('Alignment diagnostic, NOT a sub-meter terrain-accuracy claim',fontsize=15,fontweight='bold')
    fig.savefig(OUT/'alignment_diagnostics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Self-contained local browser artifact: no network calls or user-data upload.
    def uri(im):
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=88);return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
    payload=dict(map_image=uri(rgb),height_image=uri(topo_img),nodes=nodes,edges=edges,
                 terrain_nodes=tg['nodes'],terrain_edges=tg['edges'],regions=m['known_change_buffers'],
                 points=p[::15,2:4].round(2).tolist(),metrics=m)
    template=(ROOT/'tools/erangel_spatial_viewer.html.template').read_text()
    (OUT/'viewer.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False)))
    print('Rendered overview.png, graph_detail.png, alignment_diagnostics.png, viewer.html')


if __name__=='__main__':main()

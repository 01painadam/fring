import sys, numpy as np
sys.path.insert(0,'validation/src')
from fring_ref import encounter as enc
from fring_ref.params import load_saturn
SAT=load_saturn()
R=63.300144434718870
ids=[int(x) for x in open('runs/diag/ids.txt').read().split(',')]
ctrl=ids[-10:]
def load(f):
    d=np.loadtxt(f,delimiter=',',comments='#'); out={}
    for i in np.unique(d[:,0]).astype(int):
        r=d[d[:,0]==i]; out[i]=r
    return out
H=load('runs/diag/th/plain/trace_l2.csv'); H4=load('runs/diag/th4/plain/trace_l2.csv')
def series(r):
    g=enc.geo_elements(SAT,r[:,2:8])
    dist=np.linalg.norm(r[:,2:5]-r[:,8:11],axis=1)
    return r[:,1],g['a'],g['e'],np.unwrap(g['varpi']),np.unwrap(g['lam']),dist
day=86400.0
yr=365.25
def slope(t,y):
    return np.polyfit(t/day/yr,y,1)[0]
print("id  kind  t_sep[d]  dist_at_sep[km]  min_dist_before_sep(h,h4)  |  dvarpi, dlam rate [deg/yr] before sep  |  da,de(final)")
rows=[]
for i in ids:
    t,a,e,w,l,d=series(H[i]); t4,a4,e4,w4,l4,d4=series(H4[i])
    da=a-a4; de=e-e4; dw=np.degrees(w-w4); dl=np.degrees(l-l4)
    # separation: first sample with |da|>1 km
    sep=np.where(np.abs(da)>1.0)[0]
    k=sep[0] if len(sep) else len(t)
    kind='ctrl' if i in ctrl else 'diff'
    sl=slice(0,max(k-1,3))
    # distance min over a window around the separation
    if k<len(t):
        lo=max(k-3,0); dmin=min(d[lo:k+1].min(),d4[lo:k+1].min())
    else: dmin=np.nan
    mb=(d[:max(k,1)].min(),d4[:max(k,1)].min())
    rw=slope(t[sl],dw[sl]); rl=slope(t[sl],dl[sl])
    print(f"{i:3d} {kind} {t[k]/day if k<len(t) else float('nan'):8.1f} {dmin:9.1f}   ({mb[0]:8.1f},{mb[1]:8.1f})  | {rw:9.4f} {rl:9.4f} | {da[-1]:9.2f} {de[-1]:.2e}")
    rows.append((i,kind,rw,rl,k,t,da,dw,dl,de,d,d4))
c=[r for r in rows if r[1]=='ctrl']
print("\ncontrols: full-year slopes of (h - h/4):")
for r in c:
    i,_,_,_,_,t,da,dw,dl,de,_,_=r
    print(i,"dvarpi %.5f dlam %.5f deg/yr; da max %.3g km, de max %.2e, dlam end %.4f deg"%(slope(t,dw),slope(t,dl),abs(da).max(),abs(de).max(),dl[-1]))
np.save('runs/diag/rows.npy',np.array([(r[0],r[2],r[3],r[4]) for r in rows]))

print("\n=== timelines ===")
import sys
for i in [248,262,224,101,218,176]:
    r=[x for x in rows if x[0]==i][0]
    k=r[4]; t=r[5]; da=r[6]; dh=r[10]; dh4=r[11]
    print(f"id {i}: day, da[km], dist_h, dist_h4 around separation")
    lo=max(k-6,0)
    for j in range(lo,min(k+4,len(t))):
        print(f"  {t[j]/day:7.1f} {da[j]:10.3f} {dh[j]:10.1f} {dh4[j]:10.1f}")

print("\n=== element differences (h - h/4) around the separation ===")
for i in [224,248,262,87]:
    r=[x for x in rows if x[0]==i][0]
    k=r[4]; t,a,e,w,l,d=series(H[i]); t4,a4,e4,w4,l4,d4=series(H4[i])
    print(f"id {i}: a={a[0]:.1f} e={e[0]:.5f}  day, da[km], de, dvarpi[deg], dlam[deg], dlam in km, dist_h")
    for j in range(max(k-8,0),min(k+3,len(t))):
        print(f"  {t[j]/day:7.1f} {a[j]-a4[j]:9.3f} {e[j]-e4[j]:11.2e} {np.degrees(w[j]-w4[j]):10.5f} {np.degrees(l[j]-l4[j]):10.5f} {np.radians(np.degrees(l[j]-l4[j]))*a[j]:8.2f} {d[j]:9.1f}")

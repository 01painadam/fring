import sys, numpy as np
sys.path.insert(0,'validation/src')
from fring_ref import encounter as enc
from fring_ref.params import load_saturn
SAT=load_saturn()
L=lambda f: np.loadtxt(f,delimiter=',',comments='#')
ref=L('validation/reference/e1_l2_ref.csv'); st=L('validation/reference/e1_l2_states.csv')
runs={'WHFast h':L('runs/standin_l2_h1.csv'),'Rust h':L('runs/diag/h/plain/l2_wh.csv'),'Rust h/4':L('runs/diag/h4/plain/l2_wh.csv')}
a0=enc.geo_elements(SAT,st[:,1:7])['a']
aref=enc.geo_elements(SAT,ref[:,2:8])['a']
print("|a_final - a_IAS15| > 5 km (diverged from IAS15), of 300; and > 50 km")
for n,r in runs.items():
    a=enc.geo_elements(SAT,r[:,2:8])['a']; d=np.abs(a-aref)
    print(f"{n:9s} {int((d>5).sum()):4d} {int((d>50).sum()):4d}   median |da| {np.median(d):.3f} km")
# ids diverging from IAS15 only for Rust h and not h/4
a={n:enc.geo_elements(SAT,r[:,2:8])['a'] for n,r in runs.items()}
bad={n:np.abs(a[n]-aref)>5 for n in a}
print("Rust h bad & h/4 ok:",int((bad['Rust h']&~bad['Rust h/4']).sum())," h/4 bad & h ok:",int((~bad['Rust h']&bad['Rust h/4']).sum()),"both:",int((bad['Rust h']&bad['Rust h/4']).sum()))
print("WHFast bad:",int(bad['WHFast h'].sum()),"; WHFast bad & Rust h/4 ok:",int((bad['WHFast h']&~bad['Rust h/4']).sum()))
# initial offset: day-1 dlam in km, h - h/4, for all traced ids
def load(f):
    d=np.loadtxt(f,delimiter=',',comments='#'); return {int(i):d[d[:,0]==i] for i in np.unique(d[:,0])}
H=load('runs/diag/th/plain/trace_l2.csv'); H4=load('runs/diag/th4/plain/trace_l2.csv')
ids=[int(x) for x in open('runs/diag/ids.txt').read().split(',')]
off=[]
for i in ids:
    g=enc.geo_elements(SAT,H[i][:3,2:8]); g4=enc.geo_elements(SAT,H4[i][:3,2:8])
    off.append([(g['lam'][1]-g4['lam'][1]+np.pi)%(2*np.pi)-np.pi, g['a'][1]-g4['a'][1], g['e'][1]-g4['e'][1], g['a'][1]])
off=np.array(off)
print("\nday-1 offset h - h/4 over traced ids: dlam [km] median %.2f p5 %.2f p95 %.2f; da [km] median %.4f; de median %.2e"%(np.median(off[:,0]*off[:,3]),np.percentile(off[:,0]*off[:,3],5),np.percentile(off[:,0]*off[:,3],95),np.median(off[:,1]),np.median(off[:,2])))
# does dlam correlate with e or a? relation to epicycle of initial e
st_e=enc.geo_elements(SAT,st[ids,1:7])['e']; 
print("corr(dlam_km, e0) = %.2f, corr(dlam_km, a0-139380) = %.2f"%(np.corrcoef(off[:,0]*off[:,3],st_e)[0,1],np.corrcoef(off[:,0]*off[:,3],off[:,3])[0,1]))

print("\nmedian |a_final - a_IAS15| [km] by closest approach to Prometheus (IAS15 min distance)")
mind=ref[:,1]
for lo,hi in [(0,500),(500,1500),(1500,3000),(3000,10000),(10000,1e9)]:
    m=(mind>=lo)&(mind<hi)
    print(f"{lo:>6.0f}-{hi:<9.0f} n={int(m.sum()):3d} "+"  ".join(f"{n}: {np.median(np.abs(a[n]-aref)[m]):.3f}" for n in a))

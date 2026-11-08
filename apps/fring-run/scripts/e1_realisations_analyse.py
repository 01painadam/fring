import numpy as np, re, sys
R=63.300144434718870
sets=['h','substep','h4']
cnt={s:[] for s in sets}; inside={s:[] for s in sets}
for k in range(1,11):
    for s in sets:
        v='plain' if s!='substep' else 'substep'
        d=np.loadtxt(f'runs/e1-real/s{k}/{s}/{v}/l2_wh.csv',delimiter=',',comments='#')
        m=d[:,1]; y=d[:,8]
        cnt[s].append([(m<R).sum(),(m<3*R).sum(),(m<6.3*R).sum(),(y>=5).mean()])
        inside[s].append(m<R)
c={s:np.array(cnt[s]) for s in sets}
print("seed | h: 1/3/6.3 chaos | substep | h/4")
for k in range(10):
    print(k+1,*[" ".join(f"{x:g}" if j<3 else f"{x:.3f}" for j,x in enumerate(c[s][k])) +" |" for s in sets])
se=lambda a: a.std(ddof=1)/np.sqrt(len(a))
for s in sets: print(s,"mean",c[s].mean(0).round(3),"se",np.array([se(c[s][:,j]) for j in range(4)]).round(3))
for a in ['h','substep']:
    d=c[a]-c['h4']; print(a,"- h/4 mean",d.mean(0).round(3),"se",np.array([se(d[:,j]) for j in range(4)]).round(3))
for s in sets:
    I=np.array(inside[s]); print(s,"inside 1 R_H in all 10:",I.all(0).sum(),"in any:",I.any(0).sum(),"mean per run",I.sum(1).mean())

import numpy as np
R=63.300144434718870
L=lambda f: np.loadtxt(f,delimiter=',',comments='#')
ref=L('validation/reference/e1_l2_ref.csv')[:,1]
wf=L('runs/standin_l2_h1.csv')[:,1]
h=L('runs/diag/h/plain/l2_wh.csv')[:,1]
h4=L('runs/diag/h4/plain/l2_wh.csv')[:,1]
D={'IAS15':ref,'WHFast':wf,'Rust h':h,'Rust h/4':h4}
for k in (1,3,6.3):
    print(k,{n:int((v<k*R).sum()) for n,v in D.items()})
for k in (1,3,6.3):
    ins=np.array([v<k*R for v in D.values()])
    diff=np.where(~(ins.all(0)|~ins.any(0)))[0]
    print(f"\n< {k} R_H: {len(diff)} ids differ; table min dist (km): IAS15 WHFast h h/4")
    if k==1 or k==3:
        for i in diff: print(i,*[f"{v[i]:8.1f}" for v in D.values()])
    # h in, h4 out
    a=np.where((h<k*R)&~(h4<k*R))[0]; b=np.where(~(h<k*R)&(h4<k*R))[0]
    print("in at h only:",len(a),"in at h/4 only:",len(b))
    print("h-only ids",a.tolist()); print("h4-only ids",b.tolist())
    if k==6.3: print("h-only dists h:",h[a].round(0).tolist(),"h4:",h4[a].round(0).tolist())

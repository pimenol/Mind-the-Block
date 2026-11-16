from pathlib import Path
import numpy as np
from starter import GRU,impose,windows

for seed in [940,941,942]:
    d=np.load(Path(__file__).parent/f'data/synthetic_{seed}.npz')
    for split,n in [('train',2800),('dev',1000),('test',1000)]:
        x=d[split+'_observed'];assert x.shape==(n,6)
        assert np.array_equal(np.isnan(x),d[split+'_natural_missing'])
        counts=[]
        for kind in ['scatter','block','value']:
            z,m=impose(x,kind,seed);assert not (m & np.isnan(x)).any();counts.append(m.sum(0))
        assert all(np.array_equal(counts[0],c) for c in counts)

# Finite-difference checks include every GRU gate's recurrent/input/bias parameter.
rng=np.random.default_rng(940);x=rng.normal(size=(3,4,5));y=rng.normal(size=3);model=GRU(5,940,hidden=3)
old={k:v.copy() for k,v in model.p.items()};lr=1e-5;model.step(x,y,lr)
analytic={k:(old[k]-model.p[k])/lr for k in old};model.p={k:v.copy() for k,v in old.items()}
errors=[]
for k in model.p:
    idx=(0,)*model.p[k].ndim;v=float(model.p[k][idx]);eps=1e-5
    model.p[k][idx]=v+eps;hi=np.mean((model.predict(x)-y)**2)
    model.p[k][idx]=v-eps;lo=np.mean((model.predict(x)-y)**2);model.p[k][idx]=v
    errors.append(abs((hi-lo)/(2*eps)-analytic[k][idx]))
assert max(errors)<1e-6, errors
# Changes to later raw observations cannot alter already formed input windows.
raw=rng.normal(size=(100,6));target=rng.normal(size=100);a=windows(raw,target,np.zeros(6),np.ones(6),True)[0]
raw[60:]=1000;b=windows(raw,target,np.zeros(6),np.ones(6),True)[0]
assert np.array_equal(a[:37],b[:37])
print('PASS: data/mask integrity, exact marginal counts, GRU finite-difference gradients, causal preprocessing.')

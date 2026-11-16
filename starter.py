"""Small NumPy GRU: identical model capacity with observed ages or zero ages."""
import argparse,json
from pathlib import Path
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def impose(observed,kind,seed,rate=.25):
    rng=np.random.default_rng(seed); mask=np.zeros(observed.shape,bool)
    for j in range(observed.shape[1]):
        valid=np.flatnonzero(np.isfinite(observed[:,j])); count=int(rate*len(valid))
        if kind=='scatter': chosen=rng.choice(valid,count,replace=False)
        elif kind=='value':
            # Held-out mechanism: removal depends on the current observed value.
            score=observed[valid,j]+rng.normal(0,.3,len(valid));chosen=valid[np.argsort(score)[-count:]]
        else:
            # Select whole eight-step blocks first, then truncate to the exact count.
            blocks=rng.permutation(int(np.ceil(len(observed)/8)))
            order=np.concatenate([np.arange(b*8,min((b+1)*8,len(observed))) for b in blocks])
            chosen=order[np.isfinite(observed[order,j])][:count]
        mask[chosen,j]=True
    out=observed.copy();out[mask]=np.nan
    return out,mask

def windows(raw,target,mean,std,use_age):
    # Causal fill. At the beginning of each chronological partition use training mean.
    valid=np.isfinite(raw);filled=np.empty_like(raw);age=np.empty_like(raw);prev=mean.copy();a=np.zeros(len(mean))
    for t in range(len(raw)):
        prev=np.where(valid[t],raw[t],prev);a=np.where(valid[t],0,a+1);filled[t]=prev;age[t]=a
    features=np.concatenate([(filled-mean)/std,valid.astype(float),np.minimum(age,48)/48 if use_age else np.zeros_like(age)],1)
    ends=np.arange(23,len(raw)-6)
    xx=np.stack([features[e-23:e+1] for e in ends]); yy=(target[ends+6]-mean[0])/std[0]
    persistence=(filled[ends,0]-mean[0])/std[0]
    return xx.astype('float64'),yy,persistence,ends

class GRU:
    def __init__(self,inputs,seed,hidden=8):
        r=np.random.default_rng(seed);self.p={}
        for k in 'zrn':
            self.p['w'+k]=r.normal(0,.1,(inputs,hidden));self.p['u'+k]=r.normal(0,.1,(hidden,hidden));self.p['b'+k]=np.zeros(hidden)
        self.p['o']=r.normal(0,.1,hidden);self.p['c']=np.array(0.)
    def predict(self,x,cache=False):
        h=np.zeros((len(x),len(self.p['o'])));hist=[];p=self.p
        for t in range(x.shape[1]):
            v=x[:,t];z=sigmoid(v@p['wz']+h@p['uz']+p['bz']);r=sigmoid(v@p['wr']+h@p['ur']+p['br'])
            n=np.tanh(v@p['wn']+(r*h)@p['un']+p['bn']);new=z*h+(1-z)*n
            if cache:hist.append((v,h,z,r,n))
            h=new
        out=h@p['o']+p['c'];return (out,h,hist) if cache else out
    def step(self,x,y,lr=.025):
        out,h,hist=self.predict(x,True);p=self.p;g={k:np.zeros_like(v) for k,v in p.items()}
        do=2*(out-y)/len(y);g['o']=h.T@do;g['c']=do.sum();dh=do[:,None]*p['o']
        for v,h,z,r,n in reversed(hist):
            dz=dh*(h-n)*z*(1-z);dn=dh*(1-z)*(1-n*n);dhprev=dh*z
            drh=dn@p['un'].T;dr=drh*h*r*(1-r);dhprev+=drh*r
            for k,d,prev in [('z',dz,h),('r',dr,h),('n',dn,r*h)]:
                g['w'+k]+=v.T@d;g['u'+k]+=prev.T@d;g['b'+k]+=d.sum(0)
            dh=dhprev+dz@p['uz'].T+dr@p['ur'].T
        norm=np.sqrt(sum(np.sum(v*v) for v in g.values()));factor=min(1,5/(norm+1e-12))
        for k in p:p[k]-=lr*factor*g[k]

def run(seed,quick,final):
    d=np.load(Path(__file__).parent/f'data/synthetic_{seed}.npz');mean=np.nanmean(d['train_observed'],0);std=np.nanstd(d['train_observed'],0)
    partition='test' if final else 'dev';conditions=['scatter','block','value'] if final else ['scatter','block']
    train,tm=impose(d['train_observed'],'scatter',seed+10)
    records=[]
    for ages in [False,True]:
        tx,ty,_,te=windows(train,d['train_target'],mean,std,ages)
        keep=~d['train_natural_missing'][te+6,0];tx,ty=tx[keep],ty[keep]
        # Identical examples, architecture, initialization, and minibatch order in both arms.
        model=GRU(tx.shape[2],seed);rng=np.random.default_rng(seed)
        for epoch in range(2 if quick else 12):
            ix=rng.permutation(len(tx))[:320 if quick else 1600]
            for start in range(0,len(ix),32):j=ix[start:start+32];model.step(tx[j],ty[j])
        for condition in conditions:
            test,vm=impose(d[partition+'_observed'],condition,seed+20)
            vx,vy,persist,ve=windows(test,d[partition+'_target'],mean,std,ages)
            keep=~d[partition+'_natural_missing'][ve+6,0];vx,vy,persist=vx[keep],vy[keep],persist[keep]
            records.append(dict(age_input=ages,condition=condition,rmse=float(np.sqrt(np.mean((model.predict(vx)-vy)**2))*std[0]),
                persistence_rmse=float(np.sqrt(np.mean((persist-vy)**2))*std[0]),examples=len(vy)))
    # A matched-count scatter mask on this same partition is provided as a control.
    _,scatter=impose(d[partition+'_observed'],'scatter',seed+20)
    assert np.array_equal(scatter.sum(0),vm.sum(0))
    return dict(seed=seed,dataset='SYNTHETIC six-sensor benchmark',partition=partition,conditions=conditions,
      additional_missing_per_sensor=vm.sum(0).tolist(),matched_scatter_counts=True,results=records,
      warning='Starter results are not evidence that observation age fully explains missingness shift.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=940);p.add_argument('--quick',action='store_true');p.add_argument('--final',action='store_true',help='Use only after freezing the research design')
    a=p.parse_args();print(json.dumps(run(a.seed,a.quick,a.final),indent=2))

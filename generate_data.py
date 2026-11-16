"""Generate the fixed, explicitly synthetic shared forecasting benchmark."""
from pathlib import Path
import numpy as np, json

def generate(seed):
    rng=np.random.default_rng(seed); t=np.arange(4800); a=np.zeros((4800,5))
    for i in range(1,len(t)):
        a[i]=.88*a[i-1]+rng.normal(0,.22,5)
    a[:,0]+=np.sin(2*np.pi*t/96); a[:,1]+=.6*np.cos(2*np.pi*t/37)
    x=a@rng.normal(0,.65,(5,6))+rng.normal(0,.1,(len(t),6))
    # The forecast target is sensor 0; direct clean values are not provided as inputs.
    natural=rng.random(x.shape)<.035; observed=x.copy(); observed[natural]=np.nan
    arrays={}
    for label,lo,hi in [('train',0,2800),('dev',2800,3800),('test',3800,4800)]:
        arrays[label+'_observed']=observed[lo:hi].astype('float32')
        arrays[label+'_natural_missing']=natural[lo:hi]
        arrays[label+'_target']=x[lo:hi,0].astype('float32')
    return arrays

if __name__=='__main__':
    p=Path(__file__).parent/'data';p.mkdir(exist_ok=True)
    for seed in [940,941,942]: np.savez_compressed(p/f'synthetic_{seed}.npz',**generate(seed))
    (p/'metadata.json').write_text(json.dumps({'type':'SYNTHETIC, not Beijing or any real sensor dataset','seeds':[940,941,942],
      'sensors':6,'sampling':'one arbitrary time unit','split':'chronological; 2800 train,1000 dev,1000 final test',
      'natural_missing_probability':.035,'lookback':24,'horizon':6,'source':'generate_data.py','license':'CC0-1.0'},indent=2))

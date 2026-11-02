"""PICA interface demonstration using synthetic single-mode acoustics only.

This is NOT the study's neural-operator/hardware implementation. The frozen
solution family is analytical: u(xi)=exp((-beta+i*He)*xi). No dense target labels
enter fit_case(). Oracle density ratios use generator parameters for a synthetic
calibration reference, not a deployable hardware ratio estimator.
"""
from pathlib import Path
import argparse,csv,json,hashlib,time,sys
import numpy as np

WARNING='SYNTHETIC DEMONSTRATION ONLY. Not experimental results or a reproduction of the manuscript tables.'

def field(par,x):
 par=np.asarray(par);x=np.asarray(x)
 return np.exp((-par[...,1,None]+1j*par[...,0,None])*x)

def magnitude_ssim(reference, prediction):
 """Mean SSIM of pressure magnitudes on valid 11 x 11 windows, range 1."""
 def windows(field_image):
  return np.lib.stride_tricks.sliding_window_view(np.abs(field_image), (11, 11))

 a, b = windows(reference), windows(prediction)
 axes = (-2, -1)
 ma, mb = a.mean(axis=axes), b.mean(axis=axes)
 da, db = a - ma[..., None, None], b - mb[..., None, None]
 va, vb = (da * da).mean(axis=axes), (db * db).mean(axis=axes)
 cov = (da * db).mean(axis=axes)
 luminance = (2 * ma * mb + 0.01**2) / (ma**2 + mb**2 + 0.01**2)
 structure = (2 * cov + 0.03**2) / (va + vb + 0.03**2)
 return float((luminance * structure).mean())

def fit_case(y,sensors):
 """Observation-only deterministic fitting; y complex pressure/p0, x dimensionless.
 Initial unwrapped phase/log-amplitude fit followed by bounded Gauss-Newton.
 No reference field or generator parameters are accessible to this function.
 """
 den=np.sum(sensors*sensors)
 if den<=1e-12:raise ValueError('Sensor layout has no parameter information')
 p=np.array([np.dot(sensors,np.unwrap(np.angle(y)))/den,
             -np.dot(sensors,np.log(np.maximum(np.abs(y),1e-12)))/den])
 p=np.clip(p,[8.,.1],[14.,.8]);steps=0
 for step in range(10):
  pred=field(p,sensors);r=pred-y
  jac=np.column_stack([1j*sensors*pred,-sensors*pred])
  jr=np.vstack([jac.real,jac.imag]);rr=np.r_[r.real,r.imag]
  delta=np.linalg.lstsq(jr,-rr,rcond=None)[0]
  accepted=False
  for shrink in [1,.5,.25,.125,.0625]:
   candidate=np.clip(p+shrink*delta,[8.,.1],[14.,.8])
   if np.linalg.norm(field(candidate,sensors)-y)<=np.linalg.norm(r)+1e-14:
    accepted=True;break
  if not accepted:break
  steps=step+1;change=np.linalg.norm(candidate-p);p=candidate
  if change<1e-9:break
 pred=field(p,sensors);jac=np.column_stack([1j*sensors*pred,-sensors*pred])
 sigma=float(np.linalg.svd(np.vstack([jac.real,jac.imag]),compute_uv=False)[-1])
 return p,steps,sigma,float(np.sqrt(np.mean(np.abs(pred-y)**2)))

def weighted_quantile(scores,weights,test_weight,alpha=.1):
 """Weighted split conformal quantile, including test-point +infinity mass."""
 s=np.asarray(scores,dtype=float);w=np.asarray(weights,dtype=float)
 if s.shape!=w.shape or s.ndim!=1:raise ValueError('Score/weight shape mismatch')
 if not(0<alpha<1) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):raise ValueError('Invalid inputs')
 if np.any(w<0) or not np.isfinite(test_weight) or test_weight<0:raise ValueError('Weights must be finite and nonnegative')
 total=float(w.sum()+test_weight)
 if total<=0:return float('inf')
 order=np.argsort(s,kind='stable');cw=np.cumsum(w[order])/total
 idx=np.searchsorted(cw,1-alpha,side='left')
 return float(s[order[idx]]) if idx<len(s) else float('inf')

def draw_parameters(rng,n,target=False):
 bounds=([10.5,.35],[12.5,.75]) if target else ([8.,.1],[14.,.8])
 return rng.uniform(*bounds,size=(n,2))

def oracle_ratio(p):
 inside=(p[:,0]>=10.5)&(p[:,0]<=12.5)&(p[:,1]>=.35)&(p[:,1]<=.75)
 return inside.astype(float)*(6*.7)/(2*.4)

def run(seed,out):
 out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
 ncal,ntest=800,400;x=np.linspace(0,1,201);sensors=np.array([.1,.25,.4,.55,.7,.85]);noise=.01
 calp=draw_parameters(rng,ncal);testp=draw_parameters(rng,ntest,True)
 def acquisition(p):return field(p,sensors)+noise/np.sqrt(2)*(rng.standard_normal((len(p),6))+1j*rng.standard_normal((len(p),6)))
 caly=acquisition(calp);testy=acquisition(testp)
 calfit=np.array([fit_case(y,sensors)[0] for y in caly]);results=[fit_case(y,sensors) for y in testy];testfit=np.array([v[0] for v in results])
 truth_cal=field(calp,x);truth_test=field(testp,x);pred_cal=field(calfit,x);pred_test=field(testfit,x)
 scores=np.max(np.abs(pred_cal-truth_cal),axis=1);target_scores=np.max(np.abs(pred_test-truth_test),axis=1)
 weights=oracle_ratio(calp);testweights=oracle_ratio(testp)
 quantiles=np.array([weighted_quantile(scores,weights,v) for v in testweights])
 qunweighted=weighted_quantile(scores,np.ones(ncal),1.)
 covered=target_scores<=quantiles;unweighted=target_scores<=qunweighted
 base=np.broadcast_to(field(np.array([10.,.2]),x),truth_test.shape)
 rel=lambda a,b:np.sqrt(np.sum(np.abs(a-b)**2,axis=1)/np.sum(np.abs(b)**2,axis=1))
 error=rel(pred_test,truth_test);base_error=rel(base,truth_test)
 groups=(testp[:,0]>=11.5).astype(int);group_stats=[]
 for g in [0,1]:
  take=groups==g;group_stats.append({'group':g,'n':int(take.sum()),'covered':int(covered[take].sum()),'coverage':float(covered[take].mean())})
 mean_from_groups=sum(v['n']*v['coverage'] for v in group_stats)/ntest
 scale=20.;L=.09;c=1500.;om=testp[0,0]*c/L;loss=testp[0,1]/L
 dimensional=np.exp((-loss+1j*om/c)*(L*x));scaled=np.exp((-loss/scale+1j*(om/scale)/c)*(L*scale*x))
 scale_error=float(np.max(np.abs(dimensional-scaled)))
 z=np.zeros(6);null_difference=float(np.max(np.abs(field(np.array([8.,.1]),z)-field(np.array([14.,.8]),z))))
 # A separable transverse profile gives one 2-D illustration of the image metric.
 transverse=np.sin(np.linspace(0,np.pi,33))[:,None]
 slice_reference=transverse*truth_test[0]
 slice_prediction=transverse*pred_test[0]
 slice_score=magnitude_ssim(slice_reference,slice_prediction)
 tests={
  'magnitude_ssim_identical':bool(abs(magnitude_ssim(slice_reference,slice_reference)-1)<1e-12),
  'magnitude_ssim_global_phase_invariance':bool(abs(magnitude_ssim(slice_reference,1j*slice_prediction)-slice_score)<1e-12),
  'weighted_test_mass_infinity':bool(np.isinf(weighted_quantile([.2],[1.],100.))),
  'zero_total_weight_infinity':bool(np.isinf(weighted_quantile([.2],[0.],0.))),
  'finite_sample_unweighted_order_statistic':bool(weighted_quantile(np.arange(9),np.ones(9),1.,alpha=.2)==7.),
  'coverage_group_identity':bool(abs(mean_from_groups-covered.mean())<1e-12),
  'worst_group_le_marginal':bool(min(v['coverage'] for v in group_stats)<=covered.mean()+1e-12),
  'scale_equivalence':bool(scale_error<1e-12),
  'null_sensor_layout_has_identical_data':bool(null_difference==0.),
  'adaptation_reduces_mean_error':bool(error.mean()<base_error.mean()),
  'point_predictions_unchanged_by_calibration':bool(np.array_equal(pred_test,field(testfit,x))),
  'positive_sensor_jacobian':bool(min(v[2] for v in results)>0),
 }
 try:weighted_quantile([1.],[-1.],1.)
 except ValueError:tests['negative_weight_rejected']=True
 else:tests['negative_weight_rejected']=False
 try:fit_case(np.ones(6,dtype=complex),np.zeros(6))
 except ValueError:tests['null_layout_rejected']=True
 else:tests['null_layout_rejected']=False
 summary={'warning':WARNING,'seed':seed,'model':'analytical single-mode solution family; no neural network trained','n_source_calibration':ncal,'n_target_test':ntest,'sensor_count':6,'complex_noise_rms':noise,'calibration':'oracle-ratio reference; known generator distributions','mean_nominal_rel_l2':float(base_error.mean()),'mean_fitted_rel_l2':float(error.mean()),'weighted_field_coverage':float(covered.mean()),'covered_fields':int(covered.sum()),'unweighted_field_coverage':float(unweighted.mean()),'weighted_radius':float(quantiles[0]),'unweighted_radius':qunweighted,'source_weight_ESS':float(weights.sum()**2/(weights@weights)),'min_sensor_jacobian_singular_value':min(v[2] for v in results),'scale_equivalence_max_abs':scale_error,'groups':group_stats,'checks':tests,'all_checks_pass':all(tests.values())}
 np.savez_compressed(out/'synthetic_fields.npz',x=x,sensors=sensors,source_parameters=calp,target_parameters=testp,source_observations=caly,target_observations=testy,source_reference=truth_cal,target_reference=truth_test,source_prediction=pred_cal,target_prediction=pred_test,source_fitted_parameters=calfit,target_fitted_parameters=testfit,source_scores=scores,source_oracle_weights=weights,target_scores=target_scores,target_quantiles=quantiles)
 with (out/'synthetic_case_results.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['case_id','data_kind','group','he_true','loss_true','he_fitted','loss_fitted','baseline_rel_l2','fitted_rel_l2','field_score','quantile','covered','steps','jacobian_sigma_min','sensor_rmse'])
  for i in range(ntest):writer.writerow([f'synthetic-test-{i:04d}','synthetic',int(groups[i]),*testp[i],*testfit[i],base_error[i],error[i],target_scores[i],quantiles[i],int(covered[i]),*results[i][1:]])
 with (out/'synthetic_source_calibration.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['case_id','data_kind','he','loss','score','oracle_ratio'])
  for i in range(ncal):writer.writerow([f'synthetic-cal-{i:04d}','synthetic',*calp[i],scores[i],weights[i]])
 summary['synthetic_slice_magnitude_ssim']=slice_score
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
 rows=''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in summary.items() if isinstance(v,(int,float,str)))
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>PICA synthetic demo</title><style>body{max-width:900px;margin:40px auto;font:17px system-ui;color:#19334b}td{padding:9px;border-bottom:1px solid #ddd}strong{color:#a64b12}</style><h1>PICA synthetic interface demonstration</h1><p><strong>'+WARNING+'</strong></p><table>'+rows+'</table><h2>Checks</h2><pre>'+json.dumps(tests,indent=2)+'</pre>',encoding='utf-8')
 outputs=['synthetic_fields.npz','synthetic_source_calibration.csv','synthetic_case_results.csv','summary.json','report.html']
 hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in outputs}
 (out/'sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
 if not all(tests.values()):raise AssertionError(tests)
 return summary

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261031);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'synthetic_data');args=parser.parse_args()
 result=run(args.seed,args.output);print(json.dumps(result,indent=2))

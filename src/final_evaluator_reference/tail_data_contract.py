"""Schema3 metadata checks reused from the executed trainer; no policy or statistics fitting."""
import math
from contracts import *
digest_json=canonical
CFG={'dataset_id':DATASET_ID,'split_seed':20260927,'tail_target_policy':TAIL_POLICY}
EXPECTED_DATASET_ID=DATASET_ID
EXPECTED_EXPORT_PROTOCOL_HASH=EXPORT_PROTOCOL_HASH
EXPECTED_EXPORT_BUILD=EXPORT_BUILD
EXPECTED_SOURCE_PROTOCOL_HASH=D200_SOURCE_PROTOCOL_HASH
EXPECTED_RAW_SHA256=RAW_SHA
EXPECTED_PARENT_DATA_FINGERPRINT=PARENT_DATA_HASH
EXPECTED_PARENT_NORMALIZATION_HASH=PARENT_NORM_HASH
EXPECTED_NORMALIZATION_HASH=NORM_HASH
EXPECTED_PAIRED_FINGERPRINT=DATA_HASH
EXPECTED_TRAIN_DEMOS,EXPECTED_VAL_DEMOS=160,40
EXPECTED_TRAIN_SAMPLES,EXPECTED_VAL_SAMPLES=7460,1806
COMMON_KEYS=['control_steps','control_timestamps','proprio_previous','proprio_current','action_targets','action_is_recorded','action_source_indices']
ARRAY_KEYS=set(COMMON_KEYS+[c+'_'+s for c in CAMERAS for s in ['previous','current']])
def verify_paired_identity(index, dataset, protocol, normalization):
    assert index['status']==dataset['status']=='COMPLETED' and index['schema']==dataset['schema']==3
    assert index['dataset_id']==dataset['dataset_id']==EXPECTED_DATASET_ID==CFG['dataset_id']
    assert digest_json(protocol)==dataset['export_protocol_hash']==index['export_protocol_hash']==EXPECTED_EXPORT_PROTOCOL_HASH
    assert protocol['version']==dataset['code_version']=='paired-gpu-tail-export-d200-v1'
    assert protocol['build_sha256']==dataset['export_code_sha256']==EXPECTED_EXPORT_BUILD
    assert protocol['source_protocol_hash']==dataset['source_protocol_sha256']==EXPECTED_SOURCE_PROTOCOL_HASH
    assert protocol['raw_sha256']==EXPECTED_RAW_SHA256
    assert protocol['parent_data_fingerprint']==EXPECTED_PARENT_DATA_FINGERPRINT
    assert protocol['parent_normalization_hash']==EXPECTED_PARENT_NORMALIZATION_HASH
    assert protocol['policy']==dataset['tail_target_policy']==index['tail_target_policy']==CFG['tail_target_policy']
    assert dataset['parent_dataset_id']==protocol['parent_dataset_id']=='d200_paired_gpu_v2_93f4f6bf54a9'
    split=dataset['split'];training=split['train_demos'];validation=split['validation_demos']
    assert len(training)==len(set(training))==EXPECTED_TRAIN_DEMOS
    assert len(validation)==len(set(validation))==EXPECTED_VAL_DEMOS and set(training).isdisjoint(validation)
    assert split['source_split']=='D200' and split['source_split_seed']==CFG['split_seed']==20260927
    assert len(split['common_validation_demos'])==10 and set(split['common_validation_demos'])<=set(validation)
    assert set(split['historical_d50_train_demos'])<=set(training)
    assert split['common_validation_demos']==dataset['common_validation_demos']
    assert normalization['schema']==3 and normalization['train_demos']==training
    assert normalization['tail_target_policy']==CFG['tail_target_policy']
    assert normalization['scope']=='TRAIN_ONLY_OVERLAPPING_WINDOWS_INCLUDING_ENDPOINT_REPEATS'
    assert set(normalization['train_export_manifest_sha256'])==set(training)
    assert digest_json(normalization)==index['normalization_hash']==dataset['normalization_hash']==EXPECTED_NORMALIZATION_HASH
    assert normalization['proprio_sample_count']==EXPECTED_TRAIN_SAMPLES
    assert normalization['action_target_count']==EXPECTED_TRAIN_SAMPLES*16
    assert normalization['endpoint_repeat_action_target_count']==EXPECTED_TRAIN_DEMOS*120
    identity={'export_protocol':digest_json(protocol),'export_schema':3,'parent_data_fingerprint':EXPECTED_PARENT_DATA_FINGERPRINT,
        'demos':[{k:v for k,v in r.items() if k not in ['absolute_directory','relative_directory']} for r in dataset['demos']],
        'normalization':digest_json(normalization),'split':split}
    assert identity==dataset['fingerprint_basis']
    assert digest_json(identity)==index['paired_data_fingerprint']==dataset['paired_data_fingerprint']==EXPECTED_PAIRED_FINGERPRINT
    assert len(dataset['demos'])==len({r['demo_id'] for r in dataset['demos']})==EXPECTED_TRAIN_DEMOS+EXPECTED_VAL_DEMOS
    assert {r['demo_id'] for r in dataset['demos']}==set(training+validation)
    assert dataset['train_samples']==EXPECTED_TRAIN_SAMPLES and dataset['validation_samples']==EXPECTED_VAL_SAMPLES
    assert dataset['state_per_observation']==18 and dataset['state_total']==36
    assert dataset['observation_horizon']==2 and dataset['prediction_horizon']==16 and dataset['action_dim']==7
    return split

def verify_tail_shared(rgb,events,states,actions):
    assert set(rgb)==set(events)==set(COMMON_KEYS)
    for key in COMMON_KEYS:np.testing.assert_array_equal(rgb[key],events[key])
    T=len(actions);steps=rgb['control_steps'];n=T-2
    assert states.shape==(T,32) and actions.shape==(T,7) and T>=18
    assert np.isfinite(states).all() and np.isfinite(actions).all() and (np.abs(actions)<=1+1e-6).all()
    assert steps.dtype==np.int64 and rgb['control_timestamps'].shape==(n,)
    np.testing.assert_array_equal(steps,np.arange(2,T,dtype=np.int64))
    np.testing.assert_allclose(rgb['control_timestamps'],steps/20.,atol=1e-6,rtol=0)
    for suffix,offset in [('previous',-1),('current',0)]:
        value=rgb['proprio_'+suffix];assert value.dtype==np.float32 and value.shape==(n,18)
        expected=np.c_[states[steps+offset,1:10],states[steps+offset,17:26]].astype(np.float32)
        np.testing.assert_allclose(value,expected,atol=1e-6,rtol=1e-6)
    requested=steps[:,None]+np.arange(16,dtype=np.int64)
    assert rgb['action_is_recorded'].dtype==np.bool_ and rgb['action_is_recorded'].shape==(n,16)
    assert rgb['action_source_indices'].dtype==np.int64 and rgb['action_source_indices'].shape==(n,16)
    np.testing.assert_array_equal(rgb['action_is_recorded'],requested<T)
    np.testing.assert_array_equal(rgb['action_source_indices'],np.minimum(requested,T-1))
    assert int((~rgb['action_is_recorded']).sum())==120
    targets=rgb['action_targets'];assert targets.dtype==np.float32 and targets.shape==(n,16,7)
    np.testing.assert_array_equal(targets,np.asarray(actions,np.float32)[np.minimum(requested,T-1)])
    return n

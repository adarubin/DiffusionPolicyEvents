"""D200 role expansion and read-only D50 source references. No model loading."""
import random

D50_DATASET_ID = 'd50_paired_gpu_v1_227b2750b9f0'
D50_PROTOCOL_HASH = '227b2750b9f0b382da57d8d895881fe5356b426956f45d61fb9e19d619ce855c'
D50_DATA_FINGERPRINT = '746f8ea0e53ed698299b08b7e4e044e944ee29cb9d39c6f58d27b37ef29ccba8'
D50_BUILD_SHA = '3fdaa4d464fa5a1669db886e839ed70c714e5b27573c456d14ce841d7a9404bb'
D200_SPLIT_RELATIVE = 'configs/training/d200_rgb_event_train_val.json'
SPLIT_POLICY = 'nested-role-preserving-expansion-v1'


def require_pinned_v2e(settings):
    assert settings['device'] == 'cpu' and settings['emulator_seed'] == 42
    assert settings['resolved_commit'] == 'cac638444557a224ba6ee5c29260916f152c00bd'
    assert settings['pos_thres'] == settings['neg_thres'] == .2
    assert all(settings[k] == 0 for k in ['sigma_thres', 'cutoff_hz', 'leak_rate_hz',
        'shot_noise_rate_hz', 'refractory_period_s'])
    return settings


def expanded_split(d50, selections, raw_ids, selection_hashes):
    """Keep old roles and expand each new selection tranche at 80/20."""
    assert len(raw_ids) == len(set(raw_ids)) == 200
    expected = sorted(raw_ids)
    random.Random(20260927).shuffle(expected)
    for size in [50, 100, 200]:
        item = selections[size]
        assert item['selection_seed'] == 20260927
        assert item['demo_ids'] == expected[:size], 'Selection differs from the locked Notebook01 seed/order.'
    train, val = list(d50['train_demos']), list(d50['validation_demos'])
    assert len(train) == len(set(train)) == 40 and len(val) == len(set(val)) == 10
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(selections[50]['demo_ids'])
    assert all(d in train for d in TRAIN_GATE_DEMOS)
    roles = {d: 'train' for d in train}
    roles.update({d: 'validation' for d in val})
    rng = random.Random(20260927)  # split-role selection only; never policy training
    tranches = []
    for before, after in [(50, 100), (100, 200)]:
        added = selections[after]['demo_ids'][before:after].copy()
        rng.shuffle(added)
        added_val = sorted(added[:len(added)//5])
        added_train = sorted(added[len(added)//5:])
        for d in added_train: roles[d] = 'train'
        for d in added_val: roles[d] = 'validation'
        tranches.append({'selection_from': before, 'selection_to': after,
            'added_train_demos': added_train, 'added_validation_demos': added_val})
    all_ids = selections[200]['demo_ids']
    return {'task': 'Lift', 'source_split': 'D200', 'source_split_seed': 20260927,
        'split_method': SPLIT_POLICY, 'schema': 1,
        'train_demos': sorted(d for d in all_ids if roles[d] == 'train'),
        'validation_demos': sorted(d for d in all_ids if roles[d] == 'validation'),
        'num_train_demos': 160, 'num_validation_demos': 40,
        'historical_d50_split_sha256': SPLIT_SHA,
        'historical_d50_train_demos': train, 'common_validation_demos': val,
        'selection_file_sha256': selection_hashes, 'expansion_tranches': tranches,
        'seed_scope': '20260927 selection/roles; policy42/43/44; final evaluation1000-1049'}


def verify_d200_roles(split):
    assert split['split_method'] == SPLIT_POLICY and split['source_split_seed'] == 20260927
    train, val = split['train_demos'], split['validation_demos']
    assert len(train) == len(set(train)) == 160 and len(val) == len(set(val)) == 40
    assert set(train).isdisjoint(val)
    assert set(split['historical_d50_train_demos']) <= set(train)
    assert len(split['common_validation_demos']) == 10
    assert set(split['common_validation_demos']) <= set(val)
    return split


def prepare_d200_split(project, raw_ids, publish=True):
    project = Path(project)
    d50 = verify_split(project/'configs/training/d50_rgb_event_train_val.json')
    selections, hashes = {}, {}
    for size in [50, 100, 200]:
        path = project/'configs/splits'/('d'+str(size)+'.json')
        selections[size] = json.loads(path.read_text())
        hashes['d'+str(size)] = sha_file(path)
    expected = verify_d200_roles(expanded_split(d50, selections, raw_ids, hashes))
    path = project/D200_SPLIT_RELATIVE
    if path.is_file():
        assert canonical(json.loads(path.read_text())) == canonical(expected), \
            'An existing D200 role file differs. It is protected; no silent overwrite is allowed.'
    else:
        assert publish, 'Run Notebook02 first: the reviewed D200 role file is not present.'
        atomic_json(path, expected)
    return expected, sha_file(path)


def d50_anchor(project):
    """Resolve only the reviewed dataset, never a mutable latest-D50 pointer."""
    project = Path(project)
    root = safe_child(project, 'data/processed/paired_gpu_source/'+D50_DATASET_ID)
    protocol = json.loads((root/'protocol.json').read_text())
    inventory = json.loads((root/'source_inventory.json').read_text())
    assert canonical(protocol) == D50_PROTOCOL_HASH
    assert protocol['raw_sha256'] == RAW_SHA and protocol['split_sha256'] == SPLIT_SHA
    assert protocol['build_sha256'] == D50_BUILD_SHA
    assert protocol['verified_source_sha256'] == SOURCE_SHA and protocol['frozen_helpers_sha256'] == FROZEN_SHA
    assert protocol['validation_bundle_sha256'] == VALIDATION_SHA and protocol['source_contract'] == CONTRACT
    assert inventory['status'] == 'COMPLETED' and inventory['dataset_id'] == D50_DATASET_ID
    assert inventory['source_protocol_hash'] == D50_PROTOCOL_HASH
    records = inventory['demos']
    split = verify_split(project/'configs/training/d50_rgb_event_train_val.json')
    assert protocol['split'] == split and len(records) == 50
    by_demo = {r['demo_id']: r for r in records}
    assert len(by_demo) == 50 and set(by_demo) == set(split['train_demos']+split['validation_demos'])
    for demo, rec in by_demo.items():
        assert rec['dataset_role'] == ('train' if demo in split['train_demos'] else 'validation')
    return root, protocol, by_demo


def source_runtime_compatibility(old, current):
    keys = ['verified_source_sha256', 'frozen_helpers_sha256', 'raw_sha256', 'source_contract',
        'runtime_stack', 'python_major_minor', 'graphics', 'nvidia_library_sha256', 'policy_contract']
    mismatches = [k for k in keys if old.get(k) != current.get(k)]
    return {'status': 'COMPATIBLE' if not mismatches else 'REGENERATE_IN_CURRENT_RUNTIME',
        'mismatched_fields': mismatches, 'source_contract_unchanged': old['source_contract'] == current['source_contract'],
        'scope': 'identical source implementation, dependency stack and graphics; D50 references remain read-only'}


def resolve_demo_source(root, record, protocol=None):
    """An external source is allowed only under the exact recorded D50 origin."""
    root = Path(root)
    if protocol is None: protocol = json.loads((root/'protocol.json').read_text())
    demo = record['demo_id']
    expected_role = 'train' if demo in protocol['split']['train_demos'] else 'validation'
    assert demo in protocol['split']['train_demos' if expected_role == 'train' else 'validation_demos']
    assert record['dataset_role'] == expected_role
    origin = record['source_origin']
    if origin['kind'] == 'd200_generation':
        assert origin['source_dataset_id'] == 'd200_paired_gpu_v2_'+canonical(protocol)[:12]
        assert origin['source_protocol_hash'] == canonical_source_protocol(protocol)
        assert origin['split_sha256'] == protocol['split_sha256']
        folder = safe_child(root, origin['generation_relative'])
    else:
        assert origin['kind'] == 'reviewed_d50_reference'
        assert protocol['d50_reuse']['status'] == 'COMPATIBLE'
        assert origin['source_dataset_id'] == D50_DATASET_ID and origin['source_protocol_hash'] == D50_PROTOCOL_HASH
        assert origin['split_sha256'] == SPLIT_SHA
        oldroot, oldprotocol, oldrecords = d50_anchor(PROJECT_ROOT)
        assert source_runtime_compatibility(oldprotocol, protocol)['status'] == 'COMPATIBLE'
        oldrec = oldrecords[demo]
        assert oldrec['dataset_role'] == expected_role
        assert origin['generation_relative'] == oldrec['generation_relative']
        assert record['manifest_sha256'] == oldrec['manifest_sha256']
        assert record['candidate_fingerprint'] == oldrec['candidate_fingerprint']
        folder = safe_child(oldroot, origin['generation_relative'])
        old_pointer_folder, _ = checked_pointer(oldroot, oldroot/'demos'/demo/'complete.json', D50_DATASET_ID, 'source')
        assert old_pointer_folder == folder
    assert sha_file(folder/'manifest.json') == record['manifest_sha256']
    manifest = verify_source(folder, demo, origin['source_protocol_hash'], split_sha256=origin['split_sha256'])
    check_source_draw_rows(folder, manifest)
    assert manifest['candidate_fingerprint'] == record['candidate_fingerprint']
    assert all(manifest['provenance']['graphics'][k] == v for k,v in protocol['graphics'].items()), 'Per-demo graphics identity differs.'
    return folder, manifest


def canonical_source_protocol(protocol):
    return canonical(protocol)


def install_source_protocol(root, protocol):
    path = Path(root)/'protocol.json'
    if path.is_file(): assert canonical(json.loads(path.read_text())) == canonical(protocol)
    else: atomic_json(path, protocol)


def make_source_record(demo, role, manifest, folder, root, origin, reused):
    return {'demo_id': demo, 'dataset_role': role, 'status': 'COMPLETED', 'reused': reused,
        'source_origin': origin, 'frames_per_camera': manifest['frames_per_camera'],
        'event_windows': manifest['event_windows'], 'samples': manifest['paired_policy_samples'],
        'candidate_fingerprint': manifest['candidate_fingerprint'], 'manifest_sha256': sha_file(folder/'manifest.json'),
        'raw_scene_xml_sha256': manifest['provenance']['raw_scene_xml_sha256'],
        'render_seconds': manifest['render_seconds'], 'local_grayscale_failures': 0}

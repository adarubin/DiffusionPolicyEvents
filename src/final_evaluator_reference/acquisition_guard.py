"""Rollout-only quantization-amplitude guard with persistent Event equivalence checks."""
import time
from pathlib import Path
import numpy as np
from contracts import atomic_json, write_csv, CAMERAS

ROLLOUT_ACQUISITION = {
    'version': 'fixed-three-native-draws-one-level-dual-v2e-v2',
    'native_draws': 3, 'policy_image': 'always native draw3; never retry, average or quantize',
    'rgb_max_abs_uint8': 1, 'rgb_changed_color_values_per_camera': 'audit only; no count-based rejection',
    'gray_max_abs_uint8': 1, 'gray_changed_pixels_per_camera': 'audit only; no count-based rejection',
    'primary_event_source': 'native draw3 RGB2GRAY',
    'shadow_event_source': 'native draw2 RGB2GRAY, separate persistent identical CPU V2E',
    'window_tensor_max_abs': 1e-5,
    'window_rule': 'both previous/current tensors, both cameras, every control endpoint, before policy input',
    'scope': 'NEW amplitude-bounded rollout contract; source parity stays strict; only explicitly verified stricter-v2 episodes can be reused',
}

def install_guard(ctx, folder):
    """Reuse the verified three native reads and additionally retain draw2 for a shadow stream."""
    folder = Path(folder)
    # Only this isolated live worker changes the helper's acceptance predicate. File bytes,
    # separate source-validation processes and packed-input calibration are unchanged.
    original_fidelity = ctx['helper'].input_fidelity
    def rollout_fidelity(*args):
        result = original_fidelity(*args)
        result['historical_rgb_count_bound_passed'] = result['rgb_within_bound']
        result['rgb_within_bound'] = result['rgb_max_abs_uint8'] <= ROLLOUT_ACQUISITION['rgb_max_abs_uint8']
        return result
    ctx['source_input_fidelity'] = original_fidelity
    ctx['helper'].input_fidelity = rollout_fidelity
    ctx['helper'].RENDER_STABILITY_CONTRACT = ROLLOUT_ACQUISITION
    raw = ctx['get_observation']
    captures = []
    def capture(env):
        value = raw(env)
        captures.append({k: np.asarray(v).copy() for k, v in value.items()})
        if len(captures) > 3: del captures[0]
        return value
    ctx['get_observation'] = capture
    ctx['helper'].install_render_stability(ctx, folder)
    stable = ctx['get_observation']
    state = {'reads': [], 'windows': [], 'feeds': 0, 'failed': False}
    ctx['_acquisition_guard'] = state
    ctx['_last_images'] = None

    def fail(message, detail):
        state['failed'] = True
        atomic_json(folder/'acquisition_guard_failure.json', {'error': message, 'detail': detail,
            'contract': ROLLOUT_ACQUISITION})
        raise AssertionError(message)

    def read(env):
        captures.clear()
        try: result = stable(env)
        except BaseException:
            state['failed'] = True
            raise
        assert len(captures) == 3
        row = ctx['_render_stability_audit'][-1]
        for cam in CAMERAS:
            fidelity = ctx['helper'].input_fidelity(ctx, captures[1][cam], captures[2][cam])
            record = {'logical_read': row['logical_read'], 'simulation_time': row['sim_timestamp_s'],
                'camera': cam, **fidelity}
            state['reads'].append(record)
            bounded = fidelity['rgb_within_bound'] and fidelity['grayscale_max_abs_uint8'] <= 1
            if not fidelity['grayscale_exact']:
                marker = folder/('first_gray_ambiguity_'+cam+'.json')
                if not marker.exists():
                    ctx['helper'].save_native_pair_evidence(ctx,folder,marker.stem,
                        captures[1][cam],captures[2][cam],{'fidelity':record,'contract':ROLLOUT_ACQUISITION})
            if not bounded:
                fail('Acquisition exceeds the one-level native amplitude bound; input blocked.', record)
        # Commit only native images after all bounds/state checks. No pixel edits or extra renders.
        ctx['_last_draw2'] = {cam: captures[1][cam].copy() for cam in CAMERAS}
        ctx['_last_images'] = {k: v.copy() for k, v in result.items()}
        return result

    base = ctx['EventStream']
    class GuardedEventStream(base):
        def __init__(self):
            super().__init__()
            try: self.shadow = base()
            except BaseException:
                super().close()
                raise
        def feed(self, images, timestamp):
            for cam in CAMERAS:
                np.testing.assert_array_equal(images[cam], ctx['_last_images'][cam])
            started = time.perf_counter()
            try:
                super().feed(images, timestamp)
                self.shadow.feed(ctx['_last_draw2'], timestamp)
                assert self.frame_count == self.shadow.frame_count
                assert self.last_frame_time == self.shadow.last_frame_time
                state['feeds'] += 1
            except BaseException:
                state['failed'] = True
                raise
            self.last_emulation_seconds = time.perf_counter()-started
        def windows(self, end):
            try:
                result = super().windows(end)
                self.shadow.windows(end)
                for suffix in ['previous','current']:
                    for cam in CAMERAS:
                        a = getattr(self, suffix)[cam]; b = getattr(self.shadow, suffix)[cam]
                        assert a.shape == b.shape == (3,84,84)
                        error = np.abs(a.astype(np.float64)-b.astype(np.float64))
                        maximum = float(error.max())
                        record = {'end_s':float(end),'camera':cam,'window':suffix,
                            'max_abs':maximum,'changed_values':int(np.count_nonzero(error)),
                            'exact':bool(np.array_equal(a,b))}
                        state['windows'].append(record)
                        if not np.isfinite(error).all() or maximum > ROLLOUT_ACQUISITION['window_tensor_max_abs']:
                            fail('Draw2/draw3 V2E tensor equivalence failed; policy input blocked.',record)
                return result
            except BaseException:
                state['failed'] = True
                raise
        def close(self):
            try: super().close()
            finally: self.shadow.close()
    ctx.update(get_observation=read,EventStream=GuardedEventStream)

def save_guard(ctx, folder):
    state = ctx['_acquisition_guard']; folder = Path(folder)
    write_csv(folder/'acquisition_read_audit.csv',state['reads'])
    write_csv(folder/'event_equivalence.csv',state['windows'])
    windows = state['windows']
    result = {'contract':ROLLOUT_ACQUISITION,'status':'FAILED' if state['failed'] or not windows else 'PASS',
        'camera_reads':len(state['reads']), 'feeds_per_stream':state['feeds'],
        'ambiguous_camera_reads':sum(not r['grayscale_exact'] for r in state['reads']),
        'camera_reads_exceeding_historical_count_bound':sum(not r['historical_rgb_count_bound_passed'] for r in state['reads']),
        'max_changed_rgb_color_values':max((r['rgb_changed_color_values'] for r in state['reads']),default=0),
        'max_changed_gray_pixels':max((r['grayscale_changed_pixels'] for r in state['reads']),default=0),
        'tensor_checks':len(windows), 'control_endpoints_checked':len(windows)//4,
        'max_abs_tensor_difference':max((r['max_abs'] for r in windows),default=None),
        'exact_tensor_checks':sum(r['exact'] for r in windows)}
    atomic_json(folder/'acquisition_guard_summary.json',result)
    return result

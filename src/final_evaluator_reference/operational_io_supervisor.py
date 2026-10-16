"""Operational supervision of an unchanged, frozen final-evaluation payload.

No policy/source/sampler/condition/seed or scored-episode code is edited.
The supervisor never treats an unreadable completed record as a new trial.
"""
import builtins
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

VERSION = 'notebook08-operational-io-recovery-v2'


class RecoveryBlocked(RuntimeError):
    """An existing completion record could not be verified; scoring must stop."""


def write_local(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temporary, path)


class DriveMonitor:
    def __init__(self, root, status_path):
        self.root = os.path.abspath(str(root))
        self.status_path = Path(status_path)
        self.stack = []
        self.phase = 'bootstrap'
        self.job = None
        self.checked = 0
        self.cache_hits = 0
        self.hashed_bytes = 0
        self.children = {}
        self.scanned_directories = 0
        self.scanned_entries = 0
        self.protocols_checked = 0
        self.last_activity = time.monotonic()
        self.originals = []
        self.enabled = True
        self.emit()

    def is_drive(self, value):
        try:
            text = os.fsdecode(os.fspath(value))
        except TypeError:
            return False
        return text == self.root or text.startswith(self.root + os.sep)

    def emit(self):
        if not self.enabled:
            return
        active = self.stack[-1] if self.stack else None
        write_local(self.status_path, {
            'version': VERSION, 'pid': os.getpid(), 'phase': self.phase,
            'job': self.job, 'completed_records_verified': self.checked,
            'same_session_hash_cache_hits': self.cache_hits,
            'bytes_hashed': self.hashed_bytes, 'activity_monotonic': self.last_activity,
            'active_children': list(self.children.values()),
            'scanned_directories': self.scanned_directories,
            'scanned_entries': self.scanned_entries,
            'protocols_checked': self.protocols_checked,
            'pending_io': active, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        })

    def activity(self):
        self.last_activity = time.monotonic()
        self.emit()

    @contextlib.contextmanager
    def operation(self, kind, path):
        if not self.is_drive(path):
            yield
            return
        operation = {'kind': kind, 'path': os.fsdecode(os.fspath(path)),
                     'started_monotonic': time.monotonic()}
        self.stack.append(operation)
        self.activity()
        try:
            yield
        finally:
            self.stack.pop()
            self.activity()

    def set_phase(self, phase):
        self.phase = phase
        self.job = None
        self.activity()
        print('[RECOVERY PHASE]', phase, flush=True)

    def install_path_hooks(self):
        def wrap(owner, name, kind):
            original = getattr(owner, name)
            self.originals.append((owner, name, original))
            def monitored(path, *args, **kwargs):
                with self.operation(kind, path):
                    return original(path, *args, **kwargs)
            setattr(owner, name, monitored)
        # Path.read_text/bytes cover the read after open, including small JSON stalls.
        for name in ['read_text', 'read_bytes', 'write_text', 'write_bytes']:
            wrap(Path, name, name)
        for name in ['stat', 'lstat', 'listdir', 'readlink', 'mkdir', 'unlink', 'rmdir']:
            wrap(os, name, name)
        wrap(builtins, 'open', 'open')
        wrap(io, 'open', 'open')
        original_scandir = os.scandir
        monitor = self
        class MonitoredEntry:
            def __init__(self, entry):
                self._entry = entry
                self.name = entry.name
                self.path = entry.path
            def __fspath__(self):
                return self.path
            def __getattr__(self, name):
                return getattr(self._entry, name)
            def is_dir(self, *args, **kwargs):
                with monitor.operation('directory_entry_is_dir', self.path):
                    return self._entry.is_dir(*args, **kwargs)
            def is_file(self, *args, **kwargs):
                with monitor.operation('directory_entry_is_file', self.path):
                    return self._entry.is_file(*args, **kwargs)
            def is_symlink(self):
                with monitor.operation('directory_entry_is_symlink', self.path):
                    return self._entry.is_symlink()
            def stat(self, *args, **kwargs):
                with monitor.operation('directory_entry_stat', self.path):
                    return self._entry.stat(*args, **kwargs)
        class MonitoredScan:
            def __init__(self, path):
                self.path = path
                with monitor.operation('scandir_open', path):
                    self.iterator = original_scandir(path)
                monitor.scanned_directories += 1
                monitor.activity()
            def __iter__(self):
                return self
            def __next__(self):
                with monitor.operation('scandir_next', self.path):
                    entry = next(self.iterator)
                monitor.scanned_entries += 1
                monitor.activity()
                return MonitoredEntry(entry)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
            def close(self):
                with monitor.operation('scandir_close', self.path):
                    self.iterator.close()
        def monitored_scandir(path='.'):
            if not monitor.is_drive(path):
                return original_scandir(path)
            return MonitoredScan(path)
        self.originals.append((os, 'scandir', original_scandir))
        os.scandir = monitored_scandir
        original_popen = subprocess.Popen
        monitor = self
        class TrackedPopen(original_popen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                monitor.children[self.pid] = {'pid': self.pid,
                    'own_process_group': bool(kwargs.get('start_new_session', False))}
                monitor.activity()
            def poll(self):
                result = super().poll()
                if result is not None and self.pid in monitor.children:
                    monitor.children.pop(self.pid, None)
                    monitor.activity()
                return result
            def wait(self, *args, **kwargs):
                result = super().wait(*args, **kwargs)
                monitor.children.pop(self.pid, None)
                monitor.activity()
                return result
        self.originals.append((subprocess, 'Popen', original_popen))
        subprocess.Popen = TrackedPopen

    def restore(self):
        self.enabled = False
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()


def install_recovery(namespace, monitor, known_completed_keys=None):
    """Change read supervision/reuse only; retain all original semantic validators."""
    cache = {}
    original_hash = namespace['sha_file']
    original_validate = namespace['validate_episode_files']
    known_completed = set(known_completed_keys or [])

    def signature(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def monitored_hash(path):
        if not monitor.is_drive(path):
            return original_hash(path)
        key = os.path.abspath(str(path))
        before = signature(path)
        if key in cache and cache[key][0] == before:
            monitor.cache_hits += 1
            monitor.activity()
            return cache[key][1]
        digest = hashlib.sha256()
        with monitor.operation('hash', path):
            with open(path, 'rb') as stream:
                while True:
                    with monitor.operation('hash_read_8MiB', path):
                        block = stream.read(8 * 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    monitor.hashed_bytes += len(block)
                    monitor.activity()
        after = signature(path)
        if before != after:
            raise RecoveryBlocked('File changed during verification: ' + str(path))
        value = digest.hexdigest()
        cache[key] = (after, value)
        return value

    namespace['sha_file'] = monitored_hash

    def verified_episode(root, protocol_hash, job):
        monitor.job = job['key']
        monitor.activity()
        pointer = Path(root) / 'episodes' / (job['key'] + '.json')
        try:
            pointer.stat()
        except FileNotFoundError:
            if job['key'] in known_completed:
                raise RecoveryBlocked('Previously supplied completed pointer is now missing; '
                                      'do not rerun: ' + str(pointer))
            # Absence is eligible only as in the frozen implementation. The full
            # immutable input/source gates still precede all scoring.
            return None
        except OSError as exc:
            raise RecoveryBlocked('Cannot access episode pointer: ' + str(pointer)) from exc
        try:
            row = json.loads(pointer.read_text())
            if row.get('status') != 'COMPLETED':
                raise RecoveryBlocked('Existing episode pointer is not COMPLETED: ' + str(pointer))
            # Require actual stat errors to propagate, rather than is_file hiding
            # a disconnected mount or missing completed artifact behind False.
            for relative in row['artifacts']:
                path = namespace['safe_child'](root, relative)
                path.stat()
            original_validate(root, protocol_hash, job, row)
        except RecoveryBlocked:
            raise
        except (OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
            raise RecoveryBlocked('Existing completed episode is unreadable or fails validation; '
                                  'retained without rerun: ' + job['key'] + ' | ' + str(exc)) from exc
        monitor.checked += 1
        known_completed.add(job['key'])
        monitor.activity()
        print('[RESUME VERIFIED]', job['key'], '| checks this session:', monitor.checked, flush=True)
        return row

    namespace['verified_episode'] = verified_episode
    for name in ['audit_queue_rows', 'prepare_dependencies', 'audit_inputs',
                 'load_selected_models', 'publish_delivery', 'validate_export_archives']:
        original = namespace[name]
        def wrapped(*args, _original=original, _name=name, **kwargs):
            monitor.set_phase(_name)
            return _original(*args, **kwargs)
        namespace[name] = wrapped

    def verify_new_seed_roles(project, own_output, options, session):
        """Same collision predicate; one streaming scandir per foreign directory.

        The original rglob enumerated the current study before excluding every
        protocol beneath it. Prune ONLY that already-exempt subtree. All other
        directories, including historical sessions and attempts, remain in scope.
        Directory symlinks are not followed, matching the frozen Path.rglob.
        """
        monitor.set_phase('verify_new_seed_roles: streaming directory scan')
        requested = set(options['environment_seeds']) | set(options['policy_seeds'])
        results = Path(project) / 'results'
        own = Path(own_output).resolve()
        frontier = [results]
        checked = []
        directories = 0
        same_study = 0

        def visit(value, used):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ['environment_seeds', 'policy_seeds']:
                        assert isinstance(item, list) and all(type(x) is int for x in item)
                        used.update(item)
                    visit(item, used)
            elif isinstance(value, list):
                for item in value:
                    visit(item, used)

        while frontier:
            directory = frontier.pop()
            # All protocol files below this lexical subtree were excluded by
            # the original resolved-path guard; do not enumerate its artifacts.
            if directory == own or directory.is_relative_to(own):
                continue
            try:
                with monitor.operation('protocol_directory_listing', directory):
                    with os.scandir(directory) as iterator:
                        entries = sorted(iterator, key=lambda e: e.name)
                directories += 1
                if directories == 1 or directories % 25 == 0:
                    print('[SEED AUDIT]', directories, 'directories |', len(checked),
                          'foreign protocols | current:', directory, flush=True)
                children = []
                for entry in entries:
                    path = directory / entry.name
                    if path == own or path.is_relative_to(own):
                        continue
                    if entry.name == 'protocol.json':
                        if path.resolve().is_relative_to(own):
                            continue
                        record = namespace['load_json'](path)
                        if record.get('study_identity') == namespace['study_identity']():
                            same_study += 1
                        else:
                            used = set()
                            visit(record, used)
                            assert not requested & used, (
                                'Protected final/policy seed IDs were already used in another result protocol: ' + str(path))
                            checked.append(str(path.relative_to(project)))
                            monitor.protocols_checked = len(checked)
                            monitor.activity()
                            print('[SEED PROTOCOL VERIFIED]', path.relative_to(project), flush=True)
                    with monitor.operation('protocol_directory_entry', path):
                        if entry.is_dir(follow_symlinks=False):
                            children.append(path)
                frontier.extend(reversed(children))
            except OSError as exc:
                raise RecoveryBlocked('Cannot enumerate/read historical protocols; seed protection remains blocked: '
                                      + str(directory) + ' | ' + str(exc)) from exc
        namespace['atomic_json'](Path(session) / 'new_seed_role_audit.json', {
            'status': 'PASS', 'environment_seeds': options['environment_seeds'],
            'policy_seeds': options['policy_seeds'],
            'prior_protocol_files_checked': sorted(checked),
            'scope': 'protected final reset IDs1000-1049 and dedicated policy IDs4000-4049 checked against previous result protocols',
            'operational_version': VERSION, 'directories_scanned': directories,
            'same_study_protocols_exempted': same_study,
            'only_pruned_subtree': str(own),
            'traversal': 'streaming, no directory-symlink following; failures block completion',
        })
        print('[SEED AUDIT PASS]', directories, 'directories;', len(checked),
              'foreign protocol files;', same_study, 'same-study protocol files exempted.', flush=True)
    namespace['verify_new_seed_roles'] = verify_new_seed_roles

    original_run_process = namespace['run_process']
    def run_process(worker, request, *args, **kwargs):
        monitor.set_phase('worker: ' + request['id'])
        return original_run_process(worker, request, *args, **kwargs)
    namespace['run_process'] = run_process

    original_publish = namespace['publish_stress']
    def publish_stress(root, protocol, rows, status, session, *args, **kwargs):
        if kwargs.get('final') and status != 'COMPLETED':
            # The frozen finally block otherwise audits all files again after an
            # access failure. Preserve the previous reports; do not erase them
            # with an empty partial audit or perform another blocking sweep.
            print('[RECOVERY] Failure: keeping existing reports and pointers; '
                  'no automatic episode rerun or second full audit.', flush=True)
            namespace['atomic_json'](Path(session) / 'io_recovery_failure.json', {
                'status': status, 'error': kwargs.get('error'),
                'rows_verified_in_execution_loop': len(rows),
                'scope': 'partial verification count, not the count of durable episodes',
                'existing_reports_preserved': True, 'operational_version': VERSION,
            })
            return {'completed_episodes': len(rows)}
        monitor.set_phase('publish_stress: ' + status)
        return original_publish(root, protocol, rows, status, session, *args, **kwargs)
    namespace['publish_stress'] = publish_stress
    return cache


def execute_payload(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    local = request_path.parent
    monitor = DriveMonitor(request['project_root'], local / 'io_status.json')
    monitor.install_path_hooks()
    namespace = {'__name__': '__notebook08_supervised__'}
    outcome = {'status': 'FAILED', 'operational_version': VERSION,
               'operation_sha256': request['operation_sha256'], 'completion_verified': False}
    try:
        monitor.set_phase('frozen configuration and session setup')
        config = (local / 'frozen_configuration.py').read_text()
        # Authentication/mount belongs to the notebook, not a subprocess.
        config = config.replace("drive.mount('/content/drive')", "# Drive mounted by parent notebook.")
        config_lines = config.splitlines(True)
        for i, line in enumerate(config_lines):
            if line.startswith('PROJECT_DIRECTORY='):
                config_lines[i] = 'PROJECT_DIRECTORY=' + repr(request['project_root']) + '\n'
            elif line.startswith('EXPORT_FULL_BUNDLE='):
                config_lines[i] = 'EXPORT_FULL_BUNDLE=' + repr(request['export_full_bundle']) + '\n'
        config = ''.join(config_lines)
        exec(compile(config, 'frozen_configuration.py', 'exec'), namespace)
        exec(compile((local / 'frozen_payload.py').read_text(), 'frozen_payload.py', 'exec'), namespace)
        assert namespace['BUILD_SHA'] == request['frozen_build_sha256']
        install_recovery(namespace, monitor, request.get('known_completed_keys', []))
        # Preserve operational provenance separately from the frozen scientific
        # identity. This file does not change protocol.json or study_identity.
        session = namespace['SESSION']
        namespace['atomic_json'](session / 'operational_io_recovery.json', {
            'version': VERSION, 'operation_sha256': request['operation_sha256'],
            'frozen_build_sha256': namespace['BUILD_SHA'],
            'io_no_progress_seconds': request['io_timeout_seconds'],
            'preserves_scientific_identity': True,
            'changes': ['isolated parent/controller', 'visible path and phase',
                        'bounded stalled Drive reads', 'same-session stat-bound SHA cache',
                        'streaming fully monitored history scan; prune current-study subtree only',
                        'block rather than rerun unreadable completed episodes'],
        })
        (session / 'operational_io_supervisor.py').write_text(Path(__file__).read_text())
        result = namespace['run_stress_pilot']()
        outcome.update(result)
        if result['status'] == 'COMPLETED':
            monitor.set_phase('final archive and protocol verification before release')
            namespace['assert_final_protocol_lock'](namespace['OUTPUT'])
            namespace['validate_export_archives'](namespace['OUTPUT'])
            outcome['completion_verified'] = True
            for name in ['final_evaluation_summary.md', 'stress_results.png']:
                p = namespace['OUTPUT'] / name
                if p.is_file():
                    (local / name).write_bytes(p.read_bytes())
    except BaseException as exc:
        outcome.update(status='ABORTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                       error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        outcome.update(output=str(namespace.get('OUTPUT', request['expected_output'])),
                       session=str(namespace.get('SESSION', 'NOT_CREATED')),
                       frozen_build_sha256=request['frozen_build_sha256'],
                       records_verified_this_session=monitor.checked,
                       same_session_hash_cache_hits=monitor.cache_hits,
                       bytes_hashed=monitor.hashed_bytes)
        write_local(local / 'result.json', outcome)
        monitor.set_phase('worker exited: ' + outcome['status'])
        monitor.restore()
    return 0 if outcome['status'] == 'COMPLETED' else 1


def process_descendants(pid):
    # Some test sandboxes expose host /proc inside a separate PID namespace.
    # Never treat an unrelated host PID as an owned process in that situation.
    if os.readlink('/proc/self') != str(os.getpid()):
        return {pid}
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(') ', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            pass
    targets = {pid}
    while True:
        new = {p for p, parent in parents.items() if parent in targets}
        if new <= targets:
            return targets
        targets |= new


def stop_process_tree(process, known_children=None):
    targets = process_descendants(process.pid)
    groups = {process.pid}
    for child in known_children or []:
        targets.add(child['pid'])
        if child.get('own_process_group'):
            groups.add(child['pid'])
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        for pid in sorted(targets, reverse=True):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
    survivors = []
    if os.readlink('/proc/self') != str(os.getpid()):
        # Only the Popen handle is authoritative when /proc is in another PID
        # namespace. Separate child-group termination is covered by fixtures.
        return [process.pid] if process.poll() is None else []
    for pid in targets:
        try:
            state = Path('/proc') / str(pid) / 'stat'
            if state.read_text().rsplit(') ', 1)[1].split()[0] not in ['Z', 'X']:
                survivors.append(pid)
        except (OSError, IndexError):
            pass
    return survivors


def local_evidence(local):
    target = Path(local) / 'notebook08_recovery_local_evidence.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(local).iterdir()):
            if p.is_file() and p != target and p.suffix in ['.json', '.py', '.log', '.md']:
                z.write(p, p.name)
    return str(target)


def supervise(request_path, heartbeat_seconds=15):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    local = request_path.parent
    script = Path(__file__).resolve()
    command = [sys.executable, '-u', str(script), '--worker', str(request_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = b''
    last_print = last_output = started = time.monotonic()
    status = {}
    stopped = None
    survivors = []
    with open(local / 'supervisor_stdout.log', 'wb') as log:
        try:
            while True:
                for key, _ in selector.select(timeout=.5):
                    block = os.read(key.fd, 65536)
                    if block:
                        log.write(block)
                        log.flush()
                        buffer += block
                        last_output = time.monotonic()
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            print(line.decode(errors='replace'), flush=True)
                    else:
                        selector.unregister(key.fileobj)
                status_file = local / 'io_status.json'
                if status_file.is_file():
                    try:
                        status = json.loads(status_file.read_text())
                    except (ValueError, OSError):
                        pass
                now = time.monotonic()
                activity = max(last_output, status.get('activity_monotonic', started))
                idle = now - activity
                pending = status.get('pending_io')
                if process.poll() is not None and not selector.get_map():
                    break
                if pending and idle > request['io_timeout_seconds']:
                    stopped = {'status': 'IO_TIMEOUT', 'error': 'Drive operation made no progress within the bound.',
                               'idle_seconds': round(idle, 1), 'last_io': pending}
                    break
                if not pending and idle > request['silent_stage_timeout_seconds']:
                    stopped = {'status': 'STAGE_TIMEOUT', 'error': 'Controller phase made no observable progress.',
                               'idle_seconds': round(idle, 1)}
                    break
                if now - last_print >= heartbeat_seconds:
                    where = (pending or {}).get('path', '-')
                    print('[RECOVERY HEARTBEAT]', status.get('phase', 'starting'),
                          '| job:', status.get('job'), '| verified checks:', status.get('completed_records_verified', 0),
                          '| MiB hashed:', round(status.get('bytes_hashed', 0) / 1048576, 1),
                          '| directories:', status.get('scanned_directories', 0),
                          '| protocols:', status.get('protocols_checked', 0),
                          '| idle seconds:', round(idle, 1), '| file:', where, flush=True)
                    last_print = now
        except KeyboardInterrupt:
            stopped = {'status': 'ABORTED', 'error': 'User interrupted supervisor; current episode remains uncommitted.'}
        finally:
            if stopped is not None or process.poll() is None:
                survivors = stop_process_tree(process, status.get('active_children', []))
            selector.close()
            process.stdout.close()
            if buffer:
                print(buffer.decode(errors='replace'), flush=True)
    result_path = local / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if stopped:
        result = {**result, **stopped, 'completion_verified': False}
    elif process.returncode != 0 or not result:
        result.update(status='FAILED', completion_verified=False,
                      error=result.get('error', 'Worker exited without a verified completion receipt.'))
    result.update(local_directory=str(local), controller_exit_code=process.returncode,
                  final_io_status=status, surviving_processes=survivors,
                  operation_sha256=request['operation_sha256'],
                  frozen_build_sha256=request['frozen_build_sha256'],
                  output=result.get('output', request['expected_output']))
    write_local(local / 'supervisor_result.json', result)
    result['local_evidence_zip'] = local_evidence(local)
    if result.get('status') != 'COMPLETED':
        print('Evaluation paused:', result.get('status'), '|', result.get('error'), flush=True)
        print('DOWNLOAD LOCAL EVIDENCE BEFORE RESET:', result['local_evidence_zip'], flush=True)
        print('Existing Drive episode pointers were not deleted. No automatic retry was started.', flush=True)
    if survivors:
        print('A kernel I/O wait survived termination:', survivors,
              'Download local evidence, then disconnect/delete the Colab runtime. Do not start another session.', flush=True)
    return result


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        raise SystemExit(execute_payload(sys.argv[2]))
    raise SystemExit('Use supervise(request_path) from Notebook 08.')

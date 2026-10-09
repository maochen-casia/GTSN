"""A seed repeat must preserve numerical code, recovery data and all settings."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cam_var_pipeline as pipeline
from tsn.common.config import read_json, write_json


class RepeatPreparationTests(unittest.TestCase):
    def make_source(self, directory, config):
        for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor/Pi3', 'instructions'):
            (directory/name).mkdir(parents=True, exist_ok=True)
        for name in ('Dockerfile', 'pyproject.toml', 'instructions/cam_var.md', 'src/model.py'):
            (directory/name).write_text('unchanged numerical source\n')
        for name in ('run_cam_var_pipeline.py', 'report_cam_var.py', 'smoke_cam_var.py',
                     'compare_cam_var_seeds.py'):
            (directory/'scripts'/name).write_text('current orchestration\n')
        write_json(directory/'configs/cam_var.json', config)

    def test_second_seed_changes_only_seed_and_reuses_exact_recovery_and_source(self):
        base = read_json(pipeline.PROJECT_ROOT/'configs/cam_var.json')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference, current, output = directory/'reference', directory/'current', directory/'repeat'
            output.mkdir()
            (reference/'recovery').mkdir(parents=True)
            (reference/'recovery/episode_000.npz').write_bytes(b'fixed recovery bytes')
            base['train']['recovery_root'] = str(reference/'recovery')
            self.make_source(reference/'source', base)
            self.make_source(current, base)
            (current/'src/model.py').write_text('changed current source must not be used\n')
            for variant in ('full', 'without_c1'):
                config = copy.deepcopy(base)
                config['model']['contributions'] = dict(c1=variant == 'full', c2=True, c3=True)
                write_json(reference/'configs'/f'{variant}.json', config)
            write_json(reference/'protocol.json', dict(seed=20261002))
            for name in ('summary.json', 'benchmark_sha256.json', 'benchmark_audit.json'):
                write_json(reference/name, {})
            write_json(reference/'runtime_image.json', dict(gpu_ids=[0, 1, 2, 3]))
            original = pipeline.files_digest(reference)
            with patch.object(pipeline, 'PROJECT_ROOT', current):
                pipeline.prepare(output, seed=20261003, variants=['full', 'without_c1'], reference_root=reference,
                                 gpu_ids=[2, 3], gpu_uuids=['GPU-full', 'GPU-without-c1'],
                                 comparison_roots=[reference])
            for variant in ('full', 'without_c1'):
                config = read_json(output/'configs'/f'{variant}.json')
                self.assertEqual(config['train']['seed'], 20261003)
                config['train']['seed'] = 20261002
                self.assertEqual(config, read_json(reference/'configs'/f'{variant}.json'))
            self.assertEqual(pipeline.files_digest(reference), original)
            self.assertTrue((output/'recovery').is_symlink())
            self.assertEqual((output/'recovery').resolve(), reference/'recovery')
            self.assertEqual((output/'source/src/model.py').read_bytes(), (reference/'source/src/model.py').read_bytes())
            protocol = read_json(output/'protocol.json')
            self.assertEqual(list(protocol['variants']), ['full', 'without_c1'])
            self.assertEqual(protocol['gpu_ids'], [2, 3])
            self.assertEqual(pipeline.gpu_bindings(protocol), ['GPU-full', 'GPU-without-c1'])
            self.assertEqual(read_json(output/'runtime_image.json')['gpu_ids'], [2, 3])
            self.assertEqual(protocol['comparison_roots'], [str(reference), str(output)])
            self.assertEqual(protocol['previous_seed'], 20261002)
            self.assertTrue(protocol['shared_recovery'])

    def test_gpu_assignment_rejects_reusing_one_device_for_two_variants(self):
        with self.assertRaises(ValueError):
            pipeline.gpu_bindings(dict(variants=['full', 'without_c1'], gpu_uuids=['GPU-one', 'GPU-one']))
        self.assertEqual(pipeline.gpu_bindings(dict(variants=['full', 'without_c1'])), ['0', '1'])

    def test_default_preparation_keeps_all_four_variants(self):
        base = read_json(pipeline.PROJECT_ROOT/'configs/cam_var.json')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            project, output = directory/'project', directory/'output'
            output.mkdir()
            self.make_source(project, base)
            with patch.object(pipeline, 'PROJECT_ROOT', project):
                pipeline.prepare(output)
            protocol = read_json(output/'protocol.json')
            self.assertEqual(set(protocol['variants']), set(pipeline.VARIANTS))
            self.assertEqual(protocol['seed'], 20261002)
            self.assertFalse(protocol['shared_recovery'])
            self.assertEqual(read_json(output/'configs/full.json')['train']['recovery_root'], str(output/'recovery'))

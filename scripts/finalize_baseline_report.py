"""Add quantified comparison limits and recovery/validation diagnostics after audit."""
import argparse
import hashlib
from pathlib import Path
import shutil
from tsn.common.config import read_json, write_json


def comparison_figure(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ('diffusion_policy', 'dp3', 'flowpolicy', 'carp')
    names = ['GTSN', 'DP', 'DP3\n(depth)', 'FlowPolicy\n(depth)', 'CARP']
    main = summary['main']
    validation = [main['partitions']['validation']['success_rate']]
    test = [main['partitions']['test']['success_rate']]
    validation += [summary['models'][m]['validation']['success_rate'] for m in methods]
    test += [summary['models'][m]['test']['success_rate'] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    fig.suptitle('tsn-1k-var: compact baseline adaptations, one training seed\n'
        'GTSN: 351.9M parameters, pretrained Pi3; baselines: 2.7–6.0M, fresh weights; depth inputs differ',
        fontsize=10)
    x = np.arange(len(names))
    axes[0].bar(x-.17, np.array(validation)*100, width=.34, label='Validation')
    axes[0].bar(x+.17, np.array(test)*100, width=.34, label='Test')
    axes[0].set(xticks=x, xticklabels=names, ylabel='Collision-free XYZ success (%)', ylim=(0, 100))
    axes[0].legend()
    for method in methods:
        epochs = read_json(root/method/'training/epochs.json')
        axes[1].plot([r['epoch'] for r in epochs], [r['validation_rmse_m']*1000 for r in epochs], label=method)
    axes[1].set(xlabel='Policy epoch', ylabel='Validation waypoint RMSE (mm)')
    axes[1].legend(fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'comparison.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    summary = read_json(root/'summary.json')
    assert summary['audit']['passed'] and summary['audit']['baseline_rollouts'] == 800
    parameters = {'gtsn': summary['main']['initialization']['parameters'],
                  **{m: row['initialization']['parameters'] for m, row in summary['models'].items()}}
    scope = {'parameter_counts': parameters, 'capacity_matched': False, 'pretraining_matched': False,
        'observation_modality_matched': False, 'policy_training_sample_budget_matched': True,
        'published_benchmark_reproduction': False, 'training_seeds': 1,
        'interpretation': 'Results describe the stated compact TSN adaptations. They do not isolate capacity, '
            'pretraining, observation or algorithm effects and do not establish superiority to full published implementations.'}
    summary['comparison_scope'] = scope
    write_json(root/'comparison_scope.json', scope)
    diagnostics = read_json(root/'validation_diagnostics.json')
    summary['validation_diagnostics'] = diagnostics
    recovery = read_json(root/'evaluation_recovery/protocol.json')
    summary['evaluation_recovery'] = recovery
    audit_recovery = root/'audit_recovery/protocol.json'
    if audit_recovery.exists():
        summary['audit_recovery'] = read_json(audit_recovery)
    write_json(root/'summary.json', summary)
    report = root/'RESULTS.md'
    text = report.read_text()
    marker = '\n## Capacity, reconstruction and execution scope\n'
    if marker in text:
        text = text.split(marker)[0]
    scope_intro = ('These are compact TSN adaptations, not full published benchmark reproductions. '
        'GTSN has 351.9M parameters and pretrained Pi3 weights; these baselines have 2.7–6.0M '
        'parameters and fresh weights. DP3 and FlowPolicy also receive depth. The comparison '
        'does not control capacity, pretraining or observation modality.\n')
    if scope_intro not in text:
        text = text.replace('\n| Model | Input | Epoch |', '\n'+scope_intro+'\n| Model | Input | Epoch |', 1)
    lines = [marker, '', 'Network capacity and pretraining are not controlled in this experiment. '
        f"GTSN has {parameters['gtsn']:,} parameters and published Pi3 initialization; all four baseline adaptations initialize fresh. "
        'The matched setting is the policy-training sample budget, held-out partitions and numerical simulator.', '',
        '| Model | Total parameters |', '|---|---:|']
    for name, count in parameters.items():
        lines.append(f'| {name} | {count:,} |')
    prior_state = root/'evaluation_recovery/previous_container_state.json'
    recovered_oom = prior_state.exists() and read_json(prior_state).get('OOMKilled', False)
    execution_note = ('The original four long-lived evaluators exceeded the 32 GiB host-memory limit after all training had finished. '
        'Completed finite episode trajectories were retained; incomplete episodes were archived and rerun. '
        if recovered_oom else 'Native simulator memory is bounded with isolated evaluation processes. ')
    lines += ['', scope['interpretation'], '',
        f"Validation-only hold-action waypoint RMSE is {diagnostics['zero_action_waypoint_rmse_m']*1000:.2f} mm. "
        f"CARP's tokenizer reconstructs expert future joint chunks at "
        f"{diagnostics['carp_tokenizer_teacher_action_reconstruction_waypoint_rmse_m']*1000:.2f} mm waypoint RMSE. "
        'The reconstruction diagnostic receives expert future labels offline; it is neither a deployment policy '
        'nor a closed-loop success upper bound. It was not used to select or tune checkpoints.', '',
        execution_note+'Held-out episodes were evaluated in fresh processes with at most five episodes per batch. '
        'The same selected checkpoint hashes, frozen policy/simulator source, random seeds and thresholds '
        'were used throughout. All 800 resulting rollouts passed the final audit. '
        '`evaluation_recovery/` and `evaluation_batches/` preserve execution provenance.']
    if audit_recovery.exists():
        lines += ['', 'The audit reader was corrected to the archived GTSN report schema. '
            '`audit_recovery/` preserves the failed audit, corrected reader and its hash; '
            'frozen numerical source, model weights and rollout metrics were unchanged.']
    report.write_text(text+'\n'.join(lines)+'\n')
    comparison_figure(root, summary)
    shutil.copyfile(__file__, root/'report_finalization_source.py')
    write_json(root/'report_finalization.json', {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'changed_model_weights': False, 'changed_rollout_metrics': False,
        'added': ['comparison_scope', 'validation_diagnostics', 'evaluation_recovery', 'qualified_comparison_figures'],
        'audit_passed': True})


if __name__ == '__main__':
    main()

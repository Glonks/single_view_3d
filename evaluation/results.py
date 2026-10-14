import json


def summarize(per_sample):
    names = [
        name
        for name
        in per_sample[0]
        if '@' in name
    ]

    summary = {
        name: sum(sample[name] for sample in per_sample) / len(per_sample)
        for name
        in names
    }
    summary['empty'] = sum(sample['empty'] for sample in per_sample)

    return summary


def format_f1(summary):
    return (
        f'F1@0.05: {summary["f1@0.05"]:.2f} | '
        f'F1@0.02: {summary["f1@0.02"]:.2f} | '
        f'F1@0.01: {summary["f1@0.01"]:.2f}'
    )


def save_results(output, config, summary, per_sample):
    with open(output, 'w') as file:
        json.dump({
            'config':     vars(config),
            'summary':    summary,
            'per_sample': per_sample,
        }, file, indent=1)

    print(f'Saved {output}')

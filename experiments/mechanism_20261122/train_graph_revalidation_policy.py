import argparse, json, math, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path('/home/chen/mab2_conditional')
PANELS = ROOT / 'variant_panels'
TASKS = ('qasc', 'sciq', 'commonsense')
VARIANTS = ('short', 'semantic')
DENSITIES = ('rows1', 'rows2', 'rows4')
torch.set_num_threads(2)

def entropy(p):
    p = np.asarray(p, dtype=np.float32)
    return float(-(p * np.log(np.clip(p, 1e-8, 1))).sum())

def load_rows(variant, density, tasks=TASKS):
    rows = []
    for task in tasks:
        base = PANELS / f'{variant}_{density}' / task
        sealed = json.loads((base / 'SEALED.json').read_text())
        for wi in range(3):
            for line in (base / f'WORKER{wi}.jsonl').read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                full = r['full']
                vec = lambda i: np.asarray(full[str(i)], dtype=np.float32)
                rows.append({'key': r['key'], 'task': task, 'variant': variant, 'density': density,
                             'old_source': vec(0), 'new_source': vec(1), 'base': vec(2),
                             'old_relay': np.stack([vec(i) for i in range(3, 7)]),
                             'new_relay': np.stack([vec(i) for i in range(7, 11)]),
                             'labels': np.asarray([int(full[str(i)]) for i in range(11, 27)], dtype=np.int64).reshape(4, 4),
                             'tokens': r['tokens'], 'changed': tuple(int(x) for x in r.get('changed_rows', [])),
                             'gold': sealed[r['key']]})
    return rows

def truth(source, base, relay, labels, tau=.01):
    value = np.zeros(8, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            value[int(labels[i, j])] += float(source[i]) * float(relay[i][int(labels[i, j])])
    for h in range(4):
        value[h] += float(base[h])
    return value > tau

def impact_labels(r):
    target = truth(r['new_source'], r['base'], r['new_relay'], r['labels'])
    y = np.zeros(4, dtype=np.float32)
    for i in range(4):
        candidate = r['new_relay'].copy()
        candidate[i] = r['old_relay'][i]
        y[i] = float(np.any(truth(r['new_source'], r['base'], candidate, r['labels']) != target))
    return y, target

def features(r, i):
    old = r['old_relay'][i]
    return np.concatenate([r['old_source'], r['new_source'], r['base'],
                           r['new_source'] - r['old_source'], old,
                           np.asarray([entropy(old), float(old.max() - old.min()), float(r['tokens'][7 + i]) / 200.0], dtype=np.float32),
                           np.eye(4, dtype=np.float32)[i],
                           np.asarray([float(i in r['changed'])], dtype=np.float32)]).astype(np.float32)

def split_rows(rows, mode):
    train, test = [], []
    for r in rows:
        bucket = int(r['key'][:8], 16) % 5
        if mode == 'density':
            if r['variant'] == 'short' and r['density'] in ('rows1', 'rows2') and bucket < 4: train.append(r)
            if r['variant'] == 'short' and r['density'] == 'rows4' and bucket == 4: test.append(r)
        elif mode == 'interface':
            if r['variant'] == 'short' and bucket < 4: train.append(r)
            if r['variant'] == 'semantic' and bucket == 4: test.append(r)
        elif mode == 'task':
            if r['variant'] == 'short' and r['task'] in ('qasc', 'sciq') and bucket < 4: train.append(r)
            if r['variant'] == 'short' and r['task'] == 'commonsense' and bucket == 4: test.append(r)
    return train, test

class Policy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def eval_policy(model, rows, device, k):
    model.eval(); exact = []; costs = []; recalls = []
    with torch.no_grad():
        for r in rows:
            y, target = impact_labels(r)
            candidates = list(r['changed'])
            if candidates:
                x = torch.tensor(np.stack([features(r, i) for i in candidates]), device=device)
                scores = model(x).detach().cpu().numpy()
                chosen = [candidates[int(j)] for j in np.argsort(-scores)[:min(k, len(candidates))]]
            else:
                chosen = []
            relay = r['old_relay'].copy()
            for i in chosen: relay[i] = r['new_relay'][i]
            pred = truth(r['new_source'], r['base'], relay, r['labels'])
            exact.append(float(np.array_equal(pred, target)))
            impact = {i for i, value in enumerate(y) if value > .5}
            recalls.append(float(len(set(chosen) & impact) / len(impact)) if impact else 1.0)
            costs.append(float(sum(r['tokens'][7 + i] for i in chosen)))
    return {'n': len(rows), 'exact': float(np.mean(exact)), 'impact_recall': float(np.mean(recalls)),
            'tokens': float(np.mean(costs)), 'chosen_rows': float(min(k, 4))}

def eval_baselines(rows, k):
    out = {}
    for name in ('manifest', 'uncertainty', 'random'):
        vals = []; rng = random.Random(19)
        for r in rows:
            y, target = impact_labels(r); candidates = list(r['changed'])
            if name == 'manifest': chosen = candidates[:]
            elif name == 'uncertainty': chosen = sorted(candidates, key=lambda i: entropy(r['old_relay'][i]), reverse=True)[:k]
            else:
                shuffled = candidates[:]; rng.shuffle(shuffled); chosen = shuffled[:k]
            relay = r['old_relay'].copy()
            for i in chosen: relay[i] = r['new_relay'][i]
            pred = truth(r['new_source'], r['base'], relay, r['labels'])
            impact = {i for i, value in enumerate(y) if value > .5}
            vals.append((float(np.array_equal(pred, target)), float(len(set(chosen) & impact) / len(impact)) if impact else 1.0,
                         float(sum(r['tokens'][7 + i] for i in chosen))))
        out[name] = {'n': len(vals), 'exact': float(np.mean([v[0] for v in vals])),
                     'impact_recall': float(np.mean([v[1] for v in vals])), 'tokens': float(np.mean([v[2] for v in vals])),
                     'chosen_rows': float(k)}
    return out

def train_once(train_rows, test_rows, device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x_data = []; y_data = []
    for r in train_rows:
        y, _ = impact_labels(r)
        for i in range(4): x_data.append(features(r, i)); y_data.append(y[i])
    x = torch.tensor(np.asarray(x_data), device=device); y = torch.tensor(np.asarray(y_data), device=device)
    pos = float(y.sum().item()); neg = float(len(y) - pos)
    model = Policy(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    model.train()
    for _ in range(250):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 512):
            idx = order[start:start + 512]
            loss = loss_fn(model(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    result = {'seed': seed, 'train_rows': len(train_rows), 'test_rows': len(test_rows), 'positive_rate': float(pos / len(y)), 'feature_dim': int(x.shape[1])}
    for k in (1, 2, 4):
        result[f'policy_top{k}'] = eval_policy(model, test_rows, device, k)
        result[f'baseline_top{k}'] = eval_baselines(test_rows, k)
    return result

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu', type=int, default=0); parser.add_argument('--mode', choices=('density', 'interface', 'task'), required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = [r for variant in VARIANTS for density in DENSITIES for r in load_rows(variant, density)]
    train, test = split_rows(rows, args.mode)
    print(json.dumps({'mode': args.mode, 'device': str(device), 'train': len(train), 'test': len(test)}), flush=True)
    results = [train_once(train, test, device, seed) for seed in (7, 19, 43)]
    payload = {'schema': 'graph_conditioned_revalidation_policy_training_v1', 'mode': args.mode, 'device': str(device), 'results': results,
               'interpretation': 'Exploratory learned row-selection policy. Labels are derived from conditional outputs and this is not an independent final benchmark.'}
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8'); print(json.dumps(payload, indent=2), flush=True)

if __name__ == '__main__': main()


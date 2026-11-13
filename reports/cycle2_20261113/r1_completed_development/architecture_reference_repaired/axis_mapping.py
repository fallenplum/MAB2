"""Explicit frozen R1 replay semantics; no positional A/B inference."""
import numpy as np

CONDITION_BACKGROUND = {'change': 'B', 'no_change': 'A'}
CONDITION_AXIS = {'change': 0, 'no_change': 1}
BACKGROUND_AXIS = {background: CONDITION_AXIS[condition]
                   for condition, background in CONDITION_BACKGROUND.items()}


def verify_stays(raw_quality, raw_cost, replay_quality, replay_cost):
    """Require independently rescored aggregate and all-nine-node cost equality."""
    if set(raw_quality) != {'A', 'B'} or set(raw_cost) != {'A', 'B'}:
        raise ValueError('Explicit A and B evidence required')
    if replay_quality.shape != (8, 2, 24, 7) or replay_cost.shape != (8, 2, 7, 5):
        raise ValueError('Wrong frozen replay dimensions')
    for background, axis in BACKGROUND_AXIS.items():
        quality, cost = np.asarray(raw_quality[background]), np.asarray(raw_cost[background])
        if quality.shape != (8, 24) or cost.shape != (8, 3):
            raise ValueError('Incomplete 192-question aggregate or cost evidence')
        if not np.isin(quality, [0, 1]).all() or not np.isfinite(cost).all():
            raise ValueError('Invalid score or cost')
        if not np.array_equal(quality, replay_quality[:, axis, :, 0]):
            raise ValueError(background + ' aggregate score does not match its explicit replay axis')
        if not np.array_equal(cost, replay_cost[:, axis, 0, :3]):
            raise ValueError(background + ' all-node logical cost does not match its explicit replay axis')
        if not np.all(cost[:, 2] == 216):
            raise ValueError('Each episode requires 24 complete nine-node graphs')
    return {'checked_aggregate_graphs': 384, 'checked_episode_background_costs': 16,
            'condition_background': CONDITION_BACKGROUND, 'background_axis': BACKGROUND_AXIS,
            'quality_equal': True, 'complete_logical_cost_equal': True}

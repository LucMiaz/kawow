import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from kawow.model import _build_Xy_mqg, _MODEL_MQG_LOGKOW, _MODEL_MQG_LOGKOA

root = Path('.')
s01 = root / 'tests' / 'test_data' / 'S01. Compounds List for logPow-Parameters Calculations.sdf'
s02 = root / 'tests' / 'test_data' / 'S02. Compounds List for logKoa-Parameters Calculations.sdf'

Xk, yk, _ = _build_Xy_mqg(s01, 'logP', fp_size=64)
Xa, ya, _ = _build_Xy_mqg(s02, 'logKoa', fp_size=64)

kow = pickle.load(open(_MODEL_MQG_LOGKOW, 'rb'))
koa = pickle.load(open(_MODEL_MQG_LOGKOA, 'rb'))

cols_k = kow.get('feature_cols', list(range(Xk.shape[1])))
cols_a = koa.get('feature_cols', list(range(Xa.shape[1])))

pk = kow['model'].predict(Xk[:, cols_k])
pa = koa['model'].predict(Xa[:, cols_a])

out = {
    'logKow': {
        'r2_cv_payload': kow.get('r2_cv'),
        'rmse_cv_payload': kow.get('rmse_cv'),
        'r2_train': float(r2_score(yk, pk)),
    },
    'logKoa': {
        'r2_cv_payload': koa.get('r2_cv'),
        'rmse_cv_payload': koa.get('rmse_cv'),
        'r2_train': float(r2_score(ya, pa)),
    }
}

yk_bin = (yk > 5.0).astype(int)
ya_bin = (ya < 6.0).astype(int)
out['logKow']['roc_auc_B_gt5'] = float(roc_auc_score(yk_bin, pk)) if len(np.unique(yk_bin)) > 1 else None
out['logKoa']['roc_auc_Pproxy_lt6'] = float(roc_auc_score(ya_bin, -pa)) if len(np.unique(ya_bin)) > 1 else None

print(json.dumps(out, indent=2))

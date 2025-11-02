import json

def infer_columns(df, target, id_like_cols=None):
    id_like_cols = set(id_like_cols or [])
    feats = [c for c in df.columns if c not in id_like_cols | {target}]
    cat = [c for c in feats if df[c].dtype == 'object']
    num = [c for c in feats if c not in cat]
    return num, cat

def save_schema(path, feature_names, dtypes):
    schema = { 'features': feature_names, 'dtypes': dtypes }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(schema, f, indent=2)

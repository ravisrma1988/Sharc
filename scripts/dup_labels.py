import pandas as pd, numpy as np
W = "/path/to/sharc_work"
d = pd.read_csv(W + "/results/duplicates_confirmed_strict.csv"); M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet")
g = M.set_index("img_id").grade; p = M.set_index("img_id").patient
d["grade_i"] = g.reindex(d.id_i).values; d["grade_j"] = g.reindex(d.id_j).values
d["ref_i"] = (d.grade_i >= 2).astype(int); d["ref_j"] = (d.grade_j >= 2).astype(int)
print("grade disagreement among duplicate pairs:"); print(d.groupby("ds_i").apply(lambda x: pd.Series(dict(pairs=len(x), grade_differs=int((x.grade_i != x.grade_j).sum()), referable_differs=int((x.ref_i != x.ref_j).sum())))).to_string())
# drop list: keep the first image of each duplicate group, drop the rest (union-find over pairs)
parent = {}
def find(x):
    while parent.setdefault(x, x) != x: x = parent[x]
    return x
for a, b in zip(d.id_i, d.id_j): parent[find(a)] = find(b)
groups = {}
for x in list(parent): groups.setdefault(find(x), []).append(x)
drop = sorted(x for grp in groups.values() for x in sorted(grp)[1:])
pd.DataFrame(dict(img_id=drop, patient=p.reindex(drop).values)).to_csv(W + "/results/dedup_drop_list.csv", index=False)
print("duplicate groups:", len(groups), "| images to drop for the deduplicated sensitivity run:", len(drop))

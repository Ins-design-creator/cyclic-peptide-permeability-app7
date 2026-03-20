
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from io import BytesIO
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

st.set_page_config(page_title="Cyclic Peptide Permeability Triage", page_icon="🧪", layout="wide")

DEFAULT_DATA = "cyclic_peptide_final_outputs_plus_validated100.csv"

MODEL_FEATURES = [
    "tpsa_rdkit",
    "rotatable_bonds_rdkit",
    "hba_rdkit",
    "hbd_rdkit",
    "clogp_rdkit",
    "mw_rdkit",
    "ring_count_rdkit",
    "fraction_csp3_rdkit",
    "nhoh_count_rdkit",
    "no_count_rdkit",
    "formal_charge_rdkit",
    "tpsa_over_rb1",
    "n_methylation_count_model",
    "ring_size_model",
]

HIGH_THRESHOLD = -5.7
BORDERLINE_THRESHOLD = -6.3
POOR_CLASSIFIER_THRESHOLD = -6.5

# Conservative decision thresholds
HIGH_MAX_POOR_PROB = 0.15
HIGH_MIN_SIM = 0.60
HIGH_MIN_NN_MEAN = -6.0
NEIGHBOR_POOR_SIM = 0.45
NEIGHBOR_POOR_MEAN = -6.5
POOR_PROB_VETO = 0.35


def canon_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def calc_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return {
        "mw_rdkit": Descriptors.MolWt(mol),
        "tpsa_rdkit": tpsa,
        "hba_rdkit": rdMolDescriptors.CalcNumHBA(mol),
        "hbd_rdkit": rdMolDescriptors.CalcNumHBD(mol),
        "rotatable_bonds_rdkit": rb,
        "clogp_rdkit": Crippen.MolLogP(mol),
        "heavy_atom_count_rdkit": mol.GetNumHeavyAtoms(),
        "fraction_csp3_rdkit": rdMolDescriptors.CalcFractionCSP3(mol),
        "ring_count_rdkit": rdMolDescriptors.CalcNumRings(mol),
        "formal_charge_rdkit": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "nhoh_count_rdkit": float(Lipinski.NHOHCount(mol)),
        "no_count_rdkit": float(Lipinski.NOCount(mol)),
        "tpsa_over_rb1": tpsa / (rb + 1.0),
    }


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    return mol


def fp_from_mol(mol):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaf if scaf else smiles
    except Exception:
        return smiles


def infer_peptide_specific_features_from_smiles(smiles: str):
    mol = mol_from_smiles(smiles)
    atom_rings = mol.GetRingInfo().AtomRings()
    ring_size_guess = max((len(r) for r in atom_rings), default=0)

    n_methyl_count_guess = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        carbon_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6]
        if len(carbon_neighbors) >= 2:
            methyl_like = sum(1 for nbr in carbon_neighbors if nbr.GetTotalNumHs() >= 3)
            if methyl_like > 0:
                n_methyl_count_guess += 1.0

    return n_methyl_count_guess, float(ring_size_guess)


def category_from_logpapp(logpapp: float) -> str:
    if logpapp > HIGH_THRESHOLD:
        return "High"
    if logpapp >= BORDERLINE_THRESHOLD:
        return "Borderline"
    return "Poor"


def confidence_from_similarity(max_sim: float) -> str:
    if max_sim >= 0.60:
        return "In-domain"
    if max_sim >= 0.40:
        return "Moderate confidence"
    return "Low confidence"


@st.cache_data
def load_training_data(uploaded_bytes=None, uploaded_name=None):
    if uploaded_bytes is not None:
        df = pd.read_csv(BytesIO(uploaded_bytes))
        source = uploaded_name or "uploaded dataset"
    else:
        path = Path(DEFAULT_DATA)
        if not path.exists():
            raise FileNotFoundError(f"Could not find {DEFAULT_DATA}.")
        df = pd.read_csv(path)
        source = str(path)

    target_col = "PAMPA_log10_final"
    if target_col not in df.columns:
        raise ValueError("CSV must contain PAMPA_log10_final.")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["smiles", target_col]).copy()
    df["canonical_smiles"] = df["smiles"].astype(str).apply(canon_smiles)
    return df, source


@st.cache_resource
def train_models(df_hash_key, _df):
    df = _df.copy()

    desc_df = pd.DataFrame([calc_descriptors(s) for s in df["smiles"]])
    model_df = pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

    if "n_methylation_count" in model_df.columns:
        model_df["n_methylation_count_model"] = pd.to_numeric(
            model_df["n_methylation_count"], errors="coerce"
        ).fillna(0.0)
    else:
        model_df["n_methylation_count_model"] = 0.0

    if "ring_size" in model_df.columns:
        model_df["ring_size_model"] = pd.to_numeric(
            model_df["ring_size"], errors="coerce"
        ).fillna(model_df["ring_count_rdkit"])
    else:
        model_df["ring_size_model"] = model_df["ring_count_rdkit"]

    model_df = model_df.dropna(subset=MODEL_FEATURES + ["PAMPA_log10_final"]).copy()
    model_df["poor_perm"] = model_df[target_col] < POOR_CLASSIFIER_THRESHOLD
    model_df["scaffold_group"] = model_df["smiles"].apply(scaffold_from_smiles)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(model_df, groups=model_df["scaffold_group"]))
    train = model_df.iloc[train_idx].copy()
    test = model_df.iloc[test_idx].copy()

    X_train = train[MODEL_FEATURES]
    X_test = test[MODEL_FEATURES]

    reg = RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=3, n_jobs=-1
    )
    reg.fit(X_train, train[target_col])
    reg_pred = reg.predict(X_test)

    clf = RandomForestClassifier(
        n_estimators=250, random_state=42, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, train["poor_perm"])
    poor_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "reg_r2": r2_score(test[target_col], reg_pred),
        "reg_mae": mean_absolute_error(test[target_col], reg_pred),
        "poor_auc": roc_auc_score(test["poor_perm"], poor_prob) if test["poor_perm"].nunique() > 1 else np.nan,
        "poor_rate": float(train["poor_perm"].mean()),
    }

    reg_importance = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance": reg.feature_importances_,
    }).sort_values("importance", ascending=False)

    clf_importance = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    train_mols = [mol_from_smiles(s) for s in train["smiles"]]
    train_fps = [fp_from_mol(m) for m in train_mols]
    neighbor_table = train[["canonical_smiles", "smiles", "PAMPA_log10_final"]].reset_index(drop=True).copy()

    lookup_cols = ["canonical_smiles", "smiles", "PAMPA_log10_final"]
    if "compound_id" in model_df.columns:
        lookup_cols.insert(0, "compound_id")
    lookup_table = model_df[lookup_cols].copy()

    return reg, clf, metrics, reg_importance, clf_importance, train_fps, neighbor_table, lookup_table


def make_feature_row(smiles: str, user_n_methyl=None, user_ring_size=None):
    desc = calc_descriptors(smiles)
    inferred_n_methyl, inferred_ring_size = infer_peptide_specific_features_from_smiles(smiles)
    n_methyl = float(user_n_methyl) if user_n_methyl is not None else inferred_n_methyl
    ring_size = float(user_ring_size) if user_ring_size is not None else inferred_ring_size

    row = {
        "tpsa_rdkit": desc["tpsa_rdkit"],
        "rotatable_bonds_rdkit": desc["rotatable_bonds_rdkit"],
        "hba_rdkit": desc["hba_rdkit"],
        "hbd_rdkit": desc["hbd_rdkit"],
        "clogp_rdkit": desc["clogp_rdkit"],
        "mw_rdkit": desc["mw_rdkit"],
        "ring_count_rdkit": desc["ring_count_rdkit"],
        "fraction_csp3_rdkit": desc["fraction_csp3_rdkit"],
        "nhoh_count_rdkit": desc["nhoh_count_rdkit"],
        "no_count_rdkit": desc["no_count_rdkit"],
        "formal_charge_rdkit": desc["formal_charge_rdkit"],
        "tpsa_over_rb1": desc["tpsa_over_rb1"],
        "n_methylation_count_model": n_methyl,
        "ring_size_model": ring_size,
    }
    return row, desc, inferred_n_methyl, inferred_ring_size


def nearest_neighbors(smiles, train_fps, neighbor_table, top_k=3):
    mol = mol_from_smiles(smiles)
    query_fp = fp_from_mol(mol)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    order = np.argsort(sims)[::-1][:top_k]
    nn = neighbor_table.iloc[order].copy().reset_index(drop=True)
    nn["similarity"] = [float(sims[i]) for i in order]
    return nn


def exact_lookup(smiles, lookup_table):
    cs = canon_smiles(smiles)
    if not cs:
        return None
    hits = lookup_table[lookup_table["canonical_smiles"] == cs].copy()
    if hits.empty:
        return None
    # If duplicates exist, keep first but show count
    result = hits.iloc[0].to_dict()
    result["n_matches"] = len(hits)
    return result


def conservative_decision(reg_pred, poor_prob, nn_mean, max_sim):
    reg_cat = category_from_logpapp(reg_pred)
    nn_cat = category_from_logpapp(nn_mean) if not np.isnan(nn_mean) else "Borderline"
    confidence = confidence_from_similarity(max_sim)

    if poor_prob >= POOR_PROB_VETO:
        final_cat = "Poor"
        rationale = "Poor-permeability veto triggered."
    elif nn_mean <= NEIGHBOR_POOR_MEAN and max_sim >= NEIGHBOR_POOR_SIM:
        final_cat = "Poor"
        rationale = "Neighbor veto triggered by similar poor compounds."
    elif (
        reg_pred > HIGH_THRESHOLD and
        reg_cat == "High" and
        poor_prob < HIGH_MAX_POOR_PROB and
        max_sim >= HIGH_MIN_SIM and
        nn_mean > HIGH_MIN_NN_MEAN and
        confidence == "In-domain"
    ):
        final_cat = "High"
        rationale = "High call allowed only after regression, classifier, and neighbors all agreed."
    else:
        final_cat = "Borderline"
        rationale = "Conservative downgrade because evidence was not strong enough for High."
    return final_cat, rationale, reg_cat, nn_cat, confidence


def predict_smiles(smiles, reg, clf, train_fps, neighbor_table, lookup_table, user_n_methyl=None, user_ring_size=None):
    lookup_hit = exact_lookup(smiles, lookup_table)
    if lookup_hit is not None:
        known_logpapp = float(lookup_hit["PAMPA_log10_final"])
        return {
            "mode": "lookup",
            "known_logpapp": known_logpapp,
            "known_category": category_from_logpapp(known_logpapp),
            "lookup_hit": lookup_hit,
        }

    feature_row, desc, inferred_n_methyl, inferred_ring_size = make_feature_row(
        smiles, user_n_methyl=user_n_methyl, user_ring_size=user_ring_size
    )
    X = pd.DataFrame([feature_row])

    pred_logpapp = float(reg.predict(X)[0])
    poor_prob = float(clf.predict_proba(X)[0, 1])

    nn = nearest_neighbors(smiles, train_fps, neighbor_table, top_k=3)
    nn_mean = float(nn["PAMPA_log10_final"].mean()) if len(nn) else np.nan
    max_sim = float(nn["similarity"].max()) if len(nn) else 0.0

    final_cat, rationale, reg_cat, nn_cat, confidence = conservative_decision(
        pred_logpapp, poor_prob, nn_mean, max_sim
    )

    return {
        "mode": "model",
        "pred_logpapp": pred_logpapp,
        "pred_papp_cm_s": 10 ** pred_logpapp,
        "pred_category": final_cat,
        "regression_category": reg_cat,
        "neighbor_category": nn_cat,
        "poor_prob": poor_prob,
        "confidence": confidence,
        "max_similarity": max_sim,
        "neighbor_mean_logpapp": nn_mean,
        "neighbors": nn,
        "features": feature_row,
        "raw_descriptors": desc,
        "inferred_n_methyl": inferred_n_methyl,
        "inferred_ring_size": inferred_ring_size,
        "rationale": rationale,
    }


st.title("🧪 Cyclic Peptide Permeability Triage")
st.write(
    "Lookup-first revision: known compounds are returned directly from the dataset, "
    "similar poor neighbors can veto optimistic calls, and High labels are intentionally hard to achieve."
)

with st.sidebar:
    st.header("Training data")
    uploaded = st.file_uploader("Optional: upload corrected training CSV", type=["csv"])
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_name = uploaded.name if uploaded is not None else None

df, source = load_training_data(uploaded_bytes, uploaded_name)
df_hash_key = f"{len(df)}_{hash(tuple(df.columns))}_{source}"
reg, clf, metrics, reg_importance, clf_importance, train_fps, neighbor_table, lookup_table = train_models(df_hash_key, df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regression R²", f"{metrics['reg_r2']:.3f}")
c2.metric("Regression MAE", f"{metrics['reg_mae']:.3f}")
c3.metric("Poor-perm ROC AUC", "NA" if pd.isna(metrics["poor_auc"]) else f"{metrics['poor_auc']:.3f}")
c4.metric("Poor-perm rate (train)", f"{100*metrics['poor_rate']:.1f}%")

with st.expander("Decision policy", expanded=False):
    st.write("1. Exact known compounds are looked up first.")
    st.write("2. Similar poor neighbors can veto an optimistic call.")
    st.write("3. A High label is only allowed when regression, classifier, similarity, and neighbors all support it.")
    st.write(f"Training source: `{source}`")

st.subheader("Predict from SMILES")
default_smiles = df["smiles"].dropna().iloc[0]
smiles = st.text_area("Paste cyclic peptide SMILES", value=default_smiles, height=140)

col_a, col_b = st.columns(2)
with col_a:
    user_n_methyl = st.number_input(
        "Optional N-methylation count",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Recommended if known. Leave at 0 to use the SMILES-based estimate."
    )
with col_b:
    user_ring_size = st.number_input(
        "Optional ring size",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Recommended if known. Leave at 0 to use the largest ring inferred from SMILES."
    )

if st.button("Predict permeability", type="primary"):
    try:
        nm = None if user_n_methyl == 0 else user_n_methyl
        rs = None if user_ring_size == 0 else user_ring_size

        result = predict_smiles(
            smiles, reg, clf, train_fps, neighbor_table, lookup_table,
            user_n_methyl=nm, user_ring_size=rs
        )

        if result["mode"] == "lookup":
            st.success("Exact dataset match found. Returning the known value instead of a model prediction.")
            a, b, c = st.columns(3)
            a.metric("Known log10(Papp, cm/s)", f'{result["known_logpapp"]:.3f}')
            b.metric("Known category", result["known_category"])
            c.metric("Matches in dataset", int(result["lookup_hit"]["n_matches"]))

            hit_df = pd.DataFrame([result["lookup_hit"]])
            st.dataframe(hit_df, hide_index=True, width="stretch")

            if result["known_category"] == "Poor":
                st.error("Known dataset result: poor permeability.")
            elif result["known_category"] == "Borderline":
                st.warning("Known dataset result: borderline permeability.")
            else:
                st.info("Known dataset result: higher permeability.")
        else:
            a, b, c = st.columns(3)
            a.metric("Predicted log10(Papp, cm/s)", f'{result["pred_logpapp"]:.3f}')
            b.metric("Predicted Papp (cm/s)", f'{result["pred_papp_cm_s"]:.2e}')
            c.metric("Final category", result["pred_category"])

            d, e, f = st.columns(3)
            d.metric("Probability of poor permeability", f'{100*result["poor_prob"]:.1f}%')
            e.metric("Confidence", result["confidence"])
            f.metric("Nearest-train similarity", f'{result["max_similarity"]:.2f}')

            g, h = st.columns(2)
            g.metric("Regression-only category", result["regression_category"])
            h.metric("Neighbor evidence category", result["neighbor_category"])

            st.info(result["rationale"])

            if nm is None or rs is None:
                st.write("### Inferred peptide-specific fields")
                st.write(
                    f'SMILES-based estimate — N-methylation count: **{result["inferred_n_methyl"]:.0f}**, '
                    f'ring size: **{result["inferred_ring_size"]:.0f}**'
                )

            st.write("### Top nearest neighbors")
            nn_show = result["neighbors"].copy()
            nn_show["PAMPA_log10_final"] = nn_show["PAMPA_log10_final"].round(3)
            nn_show["similarity"] = nn_show["similarity"].round(3)
            st.dataframe(nn_show, hide_index=True, width="stretch")

            st.write("### Features used for prediction")
            feat_df = pd.DataFrame({"feature": list(result["features"].keys()), "value": list(result["features"].values())})
            st.dataframe(feat_df, hide_index=True, width="stretch")

            if result["confidence"] == "Low confidence":
                st.warning("Low confidence: chemistry unlike the training set.")
            if result["pred_category"] == "Poor":
                st.error("Conservative triage result: likely poor permeability.")
            elif result["pred_category"] == "Borderline":
                st.warning("Conservative triage result: borderline / needs experimental confirmation.")
            else:
                st.success("Conservative triage result: higher permeability is supported by lookup and model evidence.")
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption(
    "Recommended inputs beyond SMILES: N-methylation count and ring size. "
    "Known compounds are returned directly from the dataset when an exact match is found."
)

"""ML Assignment 1: deployed models.

Run locally:   streamlit run streamlit_app.py
Deploy free:   push this folder to GitHub, then create an app on https://share.streamlit.io
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import gas_turbine_features, intrusion_features

HERE = Path(__file__).parent
MODELS = HERE / "models"

st.set_page_config(page_title="ML Assignment 1", layout="wide")


@st.cache_resource
def load(name):
    return joblib.load(MODELS / f"{name}.joblib"), json.loads((MODELS / f"{name}_meta.json").read_text())


st.title("Machine Learning Assignment 1")
tab_ids, tab_nox = st.tabs(["Intrusion detection", "Gas turbine NOx"])

# --------------------------------------------------------------------------- classification
with tab_ids:
    model, meta = load("ids_model")
    m = meta["test_metrics"]
    st.markdown(
        f"Classifies a network connection as **Normal** or one of **9 attack types** "
        f"(model: tuned {meta['family']}, trained on UNSW-NB15).  \n"
        f"Official test set at the saved alarm threshold ({meta.get('alarm_threshold', 0.5):.2f}): detection rate **{m['detection rate']:.1%}**, false-alarm rate **{m['false alarm rate']:.1%}**.  \n"
        f"Across classes, macro-F1 is **{m['macro F1']:.3f}**. Across alarm thresholds, "
        f"binary ROC-AUC **{m['binary ROC-AUC']:.3f}**."
    )
    examples = pd.read_csv(MODELS / "ids_example_connections.csv")
    source = st.radio("Connections to classify", ["Example connections from the test set", "Upload a CSV file"], horizontal=True)
    if source == "Upload a CSV file":
        up = st.file_uploader("CSV with the UNSW-NB15 feature columns (proto, service, state, dur, sbytes, …)", type="csv",
                              max_upload_size=2, help="Up to 2 MB and 1,000 connections per upload.")
        data = None
        if up is not None:
            try:
                frame = pd.read_csv(up, nrows=1001)
                if not isinstance(frame.index, pd.RangeIndex):
                    raise ValueError("Each row must have the same number of fields as the header.")
                if not 1 <= len(frame) <= 1000:
                    raise ValueError("Upload between 1 and 1,000 connections.")
                if all(c in frame for c in meta["raw_features"]):
                    categorical = ["proto", "service", "state"]
                    numeric = [c for c in meta["raw_features"] if c not in categorical]
                    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
                    if not np.isfinite(values).all().all() or (values < 0).any().any() or (values > 1e30).any().any():
                        raise ValueError("Numeric connection features must be finite nonnegative numbers within range.")
                    if frame[categorical].isna().any().any():
                        raise ValueError("Every row needs a protocol, service and state.")
                    frame = frame.assign(**values, **frame[categorical].astype(str))
                data = frame
            except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                st.error("Upload a UTF-8 CSV with column headers and connection rows.")
            except ValueError as exc:
                st.error(str(exc))
    else:
        data = examples

    if data is not None:
        missing = [c for c in meta["raw_features"] if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = intrusion_features(data[meta["raw_features"]])[meta["input_columns"]]
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            attack_probability = 1 - proba[:, classes.index("Normal")]
            out = pd.DataFrame({
                "prediction": model.predict(X),
                "confidence": proba.max(axis=1).round(3),
                "P(attack)": attack_probability.round(3),
            })
            if "true_label" in data.columns:
                out.insert(0, "true label", data["true_label"])
            threshold = st.slider("Alarm threshold on P(attack)", 0.0, 1.0, float(meta.get("alarm_threshold", 0.5)), 0.01,
                                  help="Chosen on validation data to keep the detection rate ≥ 95%. Lower = more alarms.")
            out["alarm"] = np.where(attack_probability >= threshold, "🚨 ALERT", "ok")
            st.dataframe(pd.concat([out, data[["proto", "service", "state", "dur", "sbytes", "dbytes"]]], axis=1),
                         width="stretch")
            c1, c2 = st.columns(2)
            c1.metric("Connections", len(out))
            c2.metric("Alarms raised", int((out["alarm"] != "ok").sum()))
            row = st.selectbox("Show class probabilities for row", out.index)
            st.bar_chart(pd.Series(proba[row], index=classes).sort_values(ascending=False))

# --------------------------------------------------------------------------- regression
with tab_nox:
    model, meta = load("nox_model")
    m = meta["test_metrics"]
    example = json.loads((MODELS / "nox_example.json").read_text())
    st.subheader("Gas turbine emissions")
    with st.container(border=True):
        prediction = st.empty()
        st.caption(f"Example reading from {example['year']}. Adjust the sensors below to update the prediction.")

    def reset_example():
        for sensor, value in example["sensors"].items():
            st.session_state[f"nox_{sensor}"] = value

    st.button("Reset to example", on_click=reset_example)
    groups = {
        "Ambient conditions": {"AT": "Ambient temperature (°C)", "AP": "Ambient pressure (mbar)",
                               "AH": "Relative humidity (%)"},
        "Pressure": {"AFDP": "Air filter differential pressure (mbar)", "GTEP": "Exhaust pressure (mbar)",
                     "CDP": "Compressor discharge pressure (assumed bar)"},
        "Turbine operation": {"TIT": "Turbine inlet temperature (°C)", "TAT": "Turbine exhaust temperature (°C)",
                              "TEY": "Energy yield (MWh)"},
    }
    values = {}
    for col, (title, sensors) in zip(st.columns(3), groups.items()):
        panel = col.container(border=True)
        panel.markdown(f"**{title}**")
        for sensor, label in sensors.items():
            lo, hi, _ = meta["sensor_ranges"][sensor]
            key = f"nox_{sensor}"
            st.session_state.setdefault(key, example["sensors"][sensor])
            values[sensor] = panel.slider(label, float(lo), float(hi), step=0.01, format="%.2f", key=key)

    at = np.unique(np.append(np.linspace(*meta["sensor_ranges"]["AT"][:2], 60), values["AT"]))
    sweep = pd.DataFrame([values] * len(at)).assign(AT=at)
    emissions = model.predict(gas_turbine_features(sweep)[meta["features"]])
    selected = at == values["AT"]
    prediction.metric("Predicted NOx (mg/m³)", f"{emissions[selected][0]:.1f}")

    st.subheader("Temperature sensitivity")
    st.caption("The dot marks your selected temperature. The other eight sensors stay fixed along the curve.")
    chart = pd.DataFrame({"temperature": at, "nox": emissions, "selected": selected})
    st.vega_lite_chart(chart, {
        "encoding": {
            "x": {"field": "temperature", "type": "quantitative", "title": "Ambient temperature (°C)",
                  "scale": {"zero": False}},
            "y": {"field": "nox", "type": "quantitative", "title": "Predicted NOx (mg/m³)",
                  "scale": {"zero": False}},
        },
        "layer": [{"mark": "line"}, {"transform": [{"filter": "datum.selected"}],
                                     "mark": {"type": "point", "filled": True, "size": 100}}],
    }, width="stretch")

    with st.expander("How accurate is the model?"):
        st.write(f"Same-period held-out readings (2011–2015): R² {m['R2']:.3f}, "
                 f"MAE {m['MAE']:.2f} mg/m³.")
        st.write("In a separate chronological test, a model trained on 2011–2013 scored R² −0.827 "
                 "in 2015. Monthly refits using the previous six months scored R² 0.788 in that "
                 "simulation. Emissions shifted in 2014. The measurements do not identify the cause.")

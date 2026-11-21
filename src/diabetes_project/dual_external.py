"""MEPS and NHANES demographic grouping and external-only multitask features.

MEPS next-year utilization and NHANES past-year hospitalization are different
outcomes. Their labels stay in separate source-specific heads and group rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from diabetes_project.harmonized_meps import RACES


DEMO = ("age", "male") + tuple(f"race_{name}" for name in RACES)
TARGETS = ("meps_er", "meps_ip", "nhanes_hosp")


def nhanes_frame(nhanes: pd.DataFrame) -> pd.DataFrame:
    required = {"SEQN", "age", "male", "race_common", "hospitalized_past_year", "person_weight"}
    if missing := required - set(nhanes):
        raise ValueError(f"NHANES missing columns: {sorted(missing)}")
    output = pd.DataFrame(index=nhanes.index)
    output["person_key"] = nhanes.SEQN.astype(str)
    output["age"] = pd.to_numeric(nhanes.age, errors="coerce").fillna(65).clip(0, 100) / 100
    output["male"] = pd.to_numeric(nhanes.male, errors="coerce").fillna(.5).clip(0, 1)
    for race in RACES:
        output[f"race_{race}"] = nhanes.race_common.eq(race).astype(np.float32)
    output["nhanes_hosp"] = pd.to_numeric(nhanes.hospitalized_past_year, errors="coerce")
    output["survey_weight"] = pd.to_numeric(nhanes.person_weight, errors="coerce")
    output = output.loc[output.nhanes_hosp.isin([0, 1])].reset_index(drop=True)
    if len(output) < 100 or output.nhanes_hosp.nunique() != 2:
        raise ValueError("NHANES source has insufficient valid labels")
    return output


def meps_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"person_key", "target_er", "target_ip", "survey_weight", *DEMO}
    if missing := required - set(pairs):
        raise ValueError(f"MEPS pairs missing columns: {sorted(missing)}")
    output = pairs.loc[:, ["person_key", *DEMO, "target_er", "target_ip", "survey_weight"]].copy()
    output = output.rename(columns={"target_er": "meps_er", "target_ip": "meps_ip"})
    return output


def normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    weight = pd.to_numeric(frame.survey_weight, errors="coerce").to_numpy(dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, np.nan)
    median = np.nanmedian(weight)
    if not np.isfinite(median):
        raise ValueError("External source has no positive survey weights")
    weight = np.where(np.isfinite(weight), weight, median)
    repeats = frame.groupby("person_key").person_key.transform("size").to_numpy(dtype=float)
    weight = weight / repeats
    weight /= weight.mean()
    return np.clip(weight, 0.1, 10.0).astype(np.float32)


def group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable demographic cells shared by both external sources and UCI."""
    race_columns = [f"race_{race}" for race in RACES]
    age = pd.to_numeric(frame.age, errors="coerce").fillna(.65).clip(0, 1)
    male = pd.to_numeric(frame.male, errors="coerce").fillna(.5)
    race = frame[race_columns].to_numpy(dtype=float)
    output = pd.DataFrame(index=frame.index)
    output["age_band"] = np.minimum((age.to_numpy() * 100 // 20).astype(int), 4)
    output["sex"] = np.where(male > .75, 1, np.where(male < .25, 0, 2))
    output["race"] = np.where(race.sum(axis=1) > 0,
                              np.array(RACES, dtype=object)[race.argmax(axis=1)], "Other")
    return output


@dataclass
class GroupRate:
    global_rate: float
    age: dict
    age_sex: dict
    age_sex_race: dict
    support: dict
    alpha: float

    def transform(self, frame: pd.DataFrame, name: str) -> pd.DataFrame:
        keys = group_keys(frame)
        rate = np.empty(len(keys), dtype=np.float32)
        support = np.empty(len(keys), dtype=np.float32)
        for i, (age, sex, race) in enumerate(keys.itertuples(index=False, name=None)):
            key = (age, sex, race)
            rate[i] = self.age_sex_race.get(key, self.age_sex.get((age, sex),
                                                                    self.age.get(age, self.global_rate)))
            support[i] = self.support.get(key, 0)
        p = np.clip(rate, 1e-4, 1 - 1e-4)
        base = np.clip(self.global_rate, 1e-4, 1 - 1e-4)
        lift = np.log(p / (1 - p)) - np.log(base / (1 - base))
        return pd.DataFrame({f"{name}_rate": rate, f"{name}_logit_lift": lift.astype(np.float32),
                             f"{name}_log_support": np.log1p(support)}, index=frame.index)


def fit_group_rate(frame: pd.DataFrame, target: str, alpha: float) -> GroupRate:
    keys = group_keys(frame)
    target_values = frame[target].to_numpy(dtype=float)
    weight = normalized_weights(frame)
    global_rate = float(np.average(target_values, weights=weight))
    table = keys.copy()
    table["weighted_y"] = weight * target_values
    table["weight"] = weight
    table["n"] = 1

    def aggregate(columns, parent):
        grouped = table.groupby(columns, sort=False).agg(
            sum_y=("weighted_y", "sum"), sum_w=("weight", "sum"), n=("n", "sum"))
        rates, counts = {}, {}
        for key, row in grouped.iterrows():
            if not isinstance(key, tuple):
                key = (key,)
            prior = parent(key)
            rates[key if len(key) > 1 else key[0]] = float((row.sum_y + alpha * prior) / (row.sum_w + alpha))
            counts[key if len(key) > 1 else key[0]] = int(row.n)
        return rates, counts

    age, _ = aggregate(["age_band"], lambda _: global_rate)
    age_sex, _ = aggregate(["age_band", "sex"], lambda key: age.get(key[0], global_rate))
    full, counts = aggregate(["age_band", "sex", "race"],
                             lambda key: age_sex.get(key[:2], age.get(key[0], global_rate)))
    return GroupRate(global_rate, age, age_sex, full, counts, alpha)


def grouping_features(hospital_demo: pd.DataFrame, meps: pd.DataFrame,
                      nhanes: pd.DataFrame, alpha: float) -> dict[str, pd.DataFrame]:
    rates = {
        "meps_er": fit_group_rate(meps, "meps_er", alpha),
        "meps_ip": fit_group_rate(meps, "meps_ip", alpha),
        "nhanes_hosp": fit_group_rate(nhanes, "nhanes_hosp", alpha),
    }
    source = {name: rate.transform(hospital_demo, name) for name, rate in rates.items()}
    meps_features = pd.concat([source["meps_er"], source["meps_ip"]], axis=1)
    nhanes_features = source["nhanes_hosp"]
    dual = pd.concat([meps_features, nhanes_features], axis=1)
    dual["dual_logit_lift_mean"] = (dual.meps_ip_logit_lift + dual.nhanes_hosp_logit_lift) / 2
    dual["dual_logit_lift_gap"] = np.abs(dual.meps_ip_logit_lift - dual.nhanes_hosp_logit_lift)
    return {"meps": meps_features, "nhanes": nhanes_features, "dual": dual}


def multitask_features(hospital_demo: pd.DataFrame, meps: pd.DataFrame,
                       nhanes: pd.DataFrame, epochs: int, batch_size: int,
                       seed: int, threads: int = 4) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Train external-only shared encoders with source-specific task heads."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(threads)
    x_meps = torch.tensor(meps.loc[:, list(DEMO)].to_numpy(dtype=np.float32))
    x_nhanes = torch.tensor(nhanes.loc[:, list(DEMO)].to_numpy(dtype=np.float32))
    x_hospital = torch.tensor(hospital_demo.loc[:, list(DEMO)].to_numpy(dtype=np.float32))
    labels = {
        "meps_er": torch.tensor(meps.meps_er.to_numpy(dtype=np.float32)),
        "meps_ip": torch.tensor(meps.meps_ip.to_numpy(dtype=np.float32)),
        "nhanes_hosp": torch.tensor(nhanes.nhanes_hosp.to_numpy(dtype=np.float32)),
    }
    weights = {
        "meps": torch.tensor(normalized_weights(meps)),
        "nhanes": torch.tensor(normalized_weights(nhanes)),
    }

    class Network(nn.Module):
        def __init__(self, tasks):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(len(DEMO), 24), nn.ReLU(),
                                         nn.Linear(24, 8), nn.ReLU())
            self.heads = nn.ModuleDict({task: nn.Linear(8, 1) for task in tasks})

        def forward(self, x):
            return self.encoder(x)

    output, logs = {}, []
    for number, mode in enumerate(("meps", "nhanes", "dual")):
        tasks = ("meps_er", "meps_ip") if mode == "meps" else (
            ("nhanes_hosp",) if mode == "nhanes" else TARGETS)
        rng = np.random.default_rng(seed + number * 100)
        torch.manual_seed(seed + number * 100)
        model = Network(tasks)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
        steps = max(1, int(np.ceil(max(len(meps) if mode != "nhanes" else 0,
                                        len(nhanes) if mode != "meps" else 0) / batch_size)))
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for _ in range(steps):
                optimizer.zero_grad()
                losses = []
                if mode != "nhanes":
                    ids = rng.choice(len(meps), size=batch_size, replace=len(meps) < batch_size)
                    hidden = model(x_meps[ids])
                    per_task = []
                    for task in ("meps_er", "meps_ip"):
                        loss = F.binary_cross_entropy_with_logits(
                            model.heads[task](hidden).squeeze(1), labels[task][ids], reduction="none")
                        per_task.append((loss * weights["meps"][ids]).mean())
                    losses.append(torch.stack(per_task).mean())
                if mode != "meps":
                    ids = rng.choice(len(nhanes), size=batch_size, replace=len(nhanes) < batch_size)
                    hidden = model(x_nhanes[ids])
                    loss = F.binary_cross_entropy_with_logits(
                        model.heads["nhanes_hosp"](hidden).squeeze(1),
                        labels["nhanes_hosp"][ids], reduction="none")
                    losses.append((loss * weights["nhanes"][ids]).mean())
                objective = torch.stack(losses).mean()
                objective.backward()
                optimizer.step()
                total += objective.item()
            logs.append({"Mode": mode, "Epoch": epoch + 1, "Loss": total / steps})
        model.eval()
        with torch.no_grad():
            hidden = model(x_hospital)
            data = {f"mtl_{mode}_z_{index}": hidden[:, index].numpy()
                    for index in range(hidden.shape[1])}
            for task in tasks:
                data[f"mtl_{mode}_{task}_risk"] = torch.sigmoid(
                    model.heads[task](hidden).squeeze(1)).numpy()
        output[mode] = pd.DataFrame(data, index=hospital_demo.index).astype(np.float32)
    return output, logs

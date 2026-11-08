"""Module D: Bayesian experimentation lab."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from ...db.models import Experiment, ExperimentVariant, Observation, Profile
from ...db.session import session_scope
from ...stats import bayesian
from ..deps import current_user, rate_limiter
from ..schemas import ExperimentIn, ObservationIn, VariantIn

router = APIRouter(prefix="/experiments", tags=["experiments"],
                   dependencies=[Depends(rate_limiter)])


def _variants(s, experiment: Experiment) -> list[bayesian.Variant]:
    return [bayesian.Variant(id=v.id, label=v.label, successes=v.successes,
                             failures=v.failures, impressions_missing=v.missing)
            for v in experiment.variants]


def _serialise(experiment: Experiment) -> dict:
    return {
        "id": experiment.id, "name": experiment.name, "hypothesis": experiment.hypothesis,
        "metric": experiment.metric, "prior": experiment.prior,
        "min_observations": experiment.min_observations,
        "loss_tolerance": experiment.loss_tolerance,
        "family_level": experiment.family_level, "seed": experiment.seed,
        "status": experiment.status,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "variants": [{"id": v.id, "label": v.label, "successes": v.successes,
                      "failures": v.failures, "missing": v.missing,
                      "configuration": v.configuration} for v in experiment.variants],
    }


def _owned(s, experiment_id: str, user_id: str) -> Experiment:
    exp = s.get(Experiment, experiment_id)
    if exp is None or exp.user_id != user_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    return exp


@router.post("", status_code=201)
def create(body: ExperimentIn, user_id: str = Depends(current_user)):
    with session_scope() as s:
        if body.profile_id:
            profile = s.get(Profile, body.profile_id)
            if profile is None or profile.user_id != user_id:
                raise HTTPException(status_code=404, detail="profile not found")
        exp = Experiment(user_id=user_id, profile_id=body.profile_id, name=body.name,
                         hypothesis=body.hypothesis, metric=body.metric, prior=body.prior,
                         min_observations=body.min_observations,
                         loss_tolerance=body.loss_tolerance, family_level=body.family_level,
                         seed=body.seed)
        s.add(exp)
        s.flush()
        for v in body.variants:
            s.add(ExperimentVariant(experiment_id=exp.id, label=v.label,
                                    configuration=v.configuration, successes=v.successes,
                                    failures=v.failures, missing=v.missing, weight=v.weight))
        s.flush()
        s.refresh(exp)
        return _serialise(exp)


@router.get("")
def list_experiments(user_id: str = Depends(current_user)):
    with session_scope() as s:
        rows = s.scalars(select(Experiment).where(Experiment.user_id == user_id)
                         .order_by(Experiment.created_at.desc())).all()
        return {"experiments": [_serialise(e) for e in rows]}


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        return _serialise(_owned(s, experiment_id, user_id))


@router.delete("/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        s.delete(_owned(s, experiment_id, user_id))
    return Response(status_code=204)


@router.post("/{experiment_id}/observations")
def add_observation(experiment_id: str, body: ObservationIn,
                    user_id: str = Depends(current_user)):
    """Record data for every variant of the experiment.

    The schema separates ``missing`` (never observed) from ``failures`` (observed
    non-conversions) so a conversion rate is only computed when a denominator
    actually exists.
    """
    with session_scope() as s:
        exp = _owned(s, experiment_id, user_id)
        variants = sorted(exp.variants, key=lambda v: v.label)
        for v in variants:
            v.successes += body.successes
            v.failures += body.failures
            v.missing += body.missing
            if body.successes + body.failures + body.missing:
                s.add(Observation(variant_id=v.id, observed=(body.successes + body.failures) > 0,
                                  success=body.successes > 0 if (body.successes + body.failures) else None,
                                  context={"successes": body.successes, "failures": body.failures,
                                           "missing": body.missing, "note": body.note,
                                           "observed_at": body.observed_at.isoformat()
                                           if body.observed_at else None}))
        s.flush()
        return {"experiment_id": exp.id,
                "variants": [{"id": v.id, "label": v.label, "successes": v.successes,
                              "failures": v.failures, "missing": v.missing} for v in variants]}


@router.post("/{experiment_id}/observe-variant/{variant_id}")
def add_variant_observation(experiment_id: str, variant_id: str, body: ObservationIn,
                            user_id: str = Depends(current_user)):
    with session_scope() as s:
        exp = _owned(s, experiment_id, user_id)
        variant = s.get(ExperimentVariant, variant_id)
        if variant is None or variant.experiment_id != exp.id:
            raise HTTPException(status_code=404, detail="variant not found")
        variant.successes += body.successes
        variant.failures += body.failures
        variant.missing += body.missing
        s.add(Observation(variant_id=variant.id,
                          observed=(body.successes + body.failures) > 0,
                          success=body.successes > 0 if (body.successes + body.failures) else None,
                          context={"successes": body.successes, "failures": body.failures,
                                   "missing": body.missing, "note": body.note}))
        return {"variant_id": variant.id, "successes": variant.successes,
                "failures": variant.failures, "missing": variant.missing}


@router.get("/{experiment_id}/analysis")
def analysis(experiment_id: str, user_id: str = Depends(current_user)):
    with session_scope() as s:
        exp = _owned(s, experiment_id, user_id)
        variants = _variants(s, exp)
        out = bayesian.analyse(variants, prior_name=exp.prior,
                               min_observations=exp.min_observations,
                               loss_tolerance=exp.loss_tolerance,
                               family_level=exp.family_level, seed=exp.seed)
        out["density"] = bayesian.density_curve(out["variants"])
        out["hierarchical"] = bayesian.hierarchical(variants, prior_name=exp.prior)
        observed_total = sum(v.successes + v.failures for v in variants)
        out["capacity"] = {
            "observed_total": observed_total,
            "min_observations": exp.min_observations,
            "planning": bayesian.required_n(
                max(0.05, out["variants"][0]["posterior"]["mean"]),
                max(0.06, out["variants"][-1]["posterior"]["mean"] + 0.05)),
        }
        out["experiment"] = _serialise(exp)
        out["timeline"] = _timeline(s, exp)
        return out


def _timeline(s, exp: Experiment) -> list[dict]:
    rows = s.execute(
        select(Observation, ExperimentVariant)
        .join(ExperimentVariant, Observation.variant_id == ExperimentVariant.id)
        .where(ExperimentVariant.experiment_id == exp.id)
        .order_by(Observation.observed_at)
    ).all()
    return [{
        "variant": v.label,
        "observed_at": o.observed_at.isoformat() if o.observed_at else None,
        "context": o.context,
        "observed": o.observed,
    } for o, v in rows][-200:]


@router.get("/{experiment_id}/posterior")
def posterior(experiment_id: str, prior: str | None = None,
              level: float = 0.95, user_id: str = Depends(current_user)):
    with session_scope() as s:
        exp = _owned(s, experiment_id, user_id)
        out = bayesian.analyse(_variants(s, exp), prior_name=prior or exp.prior,
                               credible_level=level, min_observations=exp.min_observations,
                               seed=exp.seed)
        out["density"] = bayesian.density_curve(out["variants"])
        return out


@router.post("/simulate")
def simulate(body: dict, user_id: str = Depends(current_user)):
    """Synthetic experiment runner for testing the statistics without real data.

    This is explicitly labelled synthetic in the response and in the UI.
    """
    from ...stats import synthetic as syn

    spec = syn.SyntheticSpec(seed=int(body.get("seed", 7)))
    rng = syn._rng(spec.seed, "experiment")
    variants = body.get("variants") or [
        {"label": "A", "true_rate": 0.30, "observations": 400},
        {"label": "B", "true_rate": 0.36, "observations": 400},
    ]
    observed = []
    for v in variants:
        n = int(v.get("observations", 200))
        s_ = int(rng.binomial(n, float(v.get("true_rate", 0.3))))
        observed.append(bayesian.Variant(id=v["label"], label=v["label"], successes=s_,
                                        failures=n - s_))
    out = bayesian.analyse(observed, prior_name=body.get("prior", "weak_centred"),
                           seed=int(body.get("seed", 7)))
    out["density"] = bayesian.density_curve(out["variants"])
    out["synthetic"] = True
    out["true_rates"] = {v["label"]: v.get("true_rate") for v in variants}
    out["label"] = "SYNTHETIC simulation: posteriors computed from generated binomial draws"
    return out

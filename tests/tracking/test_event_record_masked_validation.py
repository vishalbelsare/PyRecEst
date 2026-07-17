from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import TrackingEvent, event_from_measurement, record_from_update


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("time", np.ma.array(4.0, mask=True), "time must be finite"),
        (
            "measurement",
            np.ma.array([1.0], mask=[True]),
            "measurement must contain real-valued numeric entries",
        ),
        (
            "covariance",
            np.ma.array([[1.0]], mask=[[True]]),
            "covariance must contain real-valued numeric entries",
        ),
        (
            "accepted",
            np.ma.array(True, mask=True),
            "accepted must be a boolean or None",
        ),
    ),
)
def test_tracking_event_rejects_masked_inputs(field, value, message) -> None:
    kwargs = {
        "time": 0.0,
        "source": "radar",
        "measurement": [1.0],
        "covariance": np.eye(1),
        "accepted": True,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        TrackingEvent(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "prior_mean",
            np.ma.array([0.0, 0.0], mask=[True, False]),
            "prior_mean must contain real-valued numeric entries",
        ),
        (
            "prior_cov",
            np.ma.array(np.eye(2), mask=[[False, True], [False, False]]),
            "prior_cov must contain real-valued numeric entries",
        ),
        (
            "posterior_mean",
            np.ma.array([0.0, 0.0], mask=[False, True]),
            "posterior_mean must contain real-valued numeric entries",
        ),
        (
            "posterior_cov",
            np.ma.array(np.eye(2), mask=[[False, False], [True, False]]),
            "posterior_cov must contain real-valued numeric entries",
        ),
        (
            "innovation",
            np.ma.array([0.0], mask=[True]),
            "innovation must contain real-valued numeric entries",
        ),
        (
            "innovation_cov",
            np.ma.array([[1.0]], mask=[[True]]),
            "innovation_cov must contain real-valued numeric entries",
        ),
        (
            "nis",
            np.ma.array(1.0, mask=True),
            "nis must be finite and nonnegative",
        ),
        (
            "accepted",
            np.ma.array(False, mask=True),
            "accepted must be a boolean or None",
        ),
    ),
)
def test_tracking_record_rejects_masked_inputs(field, value, message) -> None:
    event = event_from_measurement(time=0.0, source="radar")
    kwargs = {
        "event": event,
        "prior_mean": [0.0, 0.0],
        "prior_cov": np.eye(2),
        "posterior_mean": [0.0, 0.0],
        "posterior_cov": np.eye(2),
        "innovation": [0.0],
        "innovation_cov": np.eye(1),
        "nis": 1.0,
        "accepted": True,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        record_from_update(**kwargs)


def test_event_records_accept_unmasked_masked_arrays() -> None:
    event = TrackingEvent(
        time=np.ma.array(1.0, mask=False),
        source="radar",
        measurement=np.ma.array([2.0], mask=[False]),
        covariance=np.ma.array([[3.0]], mask=[[False]]),
        accepted=np.ma.array(True, mask=False),
    )
    record = record_from_update(
        event=event,
        prior_mean=np.ma.array([0.0], mask=[False]),
        prior_cov=np.ma.array([[1.0]], mask=[[False]]),
        posterior_mean=np.ma.array([1.0], mask=[False]),
        posterior_cov=np.ma.array([[2.0]], mask=[[False]]),
        innovation=np.ma.array([1.0], mask=[False]),
        innovation_cov=np.ma.array([[3.0]], mask=[[False]]),
        nis=np.ma.array(0.5, mask=False),
        accepted=np.ma.array(False, mask=False),
    )

    assert event.time == 1.0
    assert event.accepted is True
    assert np.array_equal(event.measurement, np.array([2.0]))
    assert record.nis == 0.5
    assert record.accepted is False
    assert np.array_equal(record.posterior_mean, np.array([1.0]))

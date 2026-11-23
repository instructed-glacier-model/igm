"""Parameters as constants or as schedules over model time (``igm.utils.schedule``)."""

import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.utils.schedule import (
    Schedule,
    check_schedule,
    is_count,
    is_positive,
    parse_schedule,
    value_at,
)

SCHEDULE = [[0.0, 20], [1.0, 5], [10.0, 0]]

CASES = [
    (-3.0, 20),
    (0.0, 20),
    (0.999, 20),
    (1.0, 5),
    (9.5, 5),
    (10.0, 0),
    (1.0e6, 0),
]


class _Unreadable:
    """A model time that fails when read."""

    def __float__(self):
        raise AssertionError("t was read")


@pytest.mark.parametrize("param", [5, 0, -1, 2.5, "5", None])
def test_a_constant_is_returned_as_it_is_and_t_is_not_read(param):
    assert value_at(param) == param
    assert value_at(param, _Unreadable()) == param


def test_a_schedule_gives_its_first_value_without_a_time():
    assert value_at(SCHEDULE) == 20
    assert value_at(OmegaConf.create(SCHEDULE), None) == 20


@pytest.mark.parametrize("t, expected", CASES)
def test_a_schedule_is_piecewise_constant_from_each_start(t, expected):
    for schedule in (SCHEDULE, OmegaConf.create(SCHEDULE)):
        value = value_at(schedule, t)
        assert value == expected
        assert type(value) is int


@pytest.mark.parametrize("t, expected", CASES)
def test_a_tensor_time_is_read_on_the_host_when_eager(t, expected):
    assert value_at(SCHEDULE, tf.Variable(t)) == expected
    assert value_at(SCHEDULE, tf.constant(t, tf.float64)) == expected


@pytest.mark.parametrize("parse", [False, True])
def test_inside_a_graph_the_schedule_is_traced_once(parse):
    """Also with a schedule parsed anew at each call, as long as it is equal."""
    traces = []

    @tf.function(autograph=False, reduce_retracing=True)
    def scheduled(param, t):
        traces.append(1)
        return value_at(param, t)

    t = tf.Variable(0.0)
    for t_value, expected in CASES:
        schedule = OmegaConf.create(SCHEDULE)
        if parse:
            schedule = parse_schedule(schedule)
        t.assign(t_value)
        value = scheduled(schedule, t)
        assert value.dtype == tf.float32
        assert float(value) == expected
    assert len(traces) == 1


def test_parse_schedule():
    parsed = parse_schedule(OmegaConf.create({"p": SCHEDULE}).p)
    assert parsed == Schedule(starts=(0.0, 1.0, 10.0), values=(20, 5, 0))
    assert parse_schedule(parsed) is parsed
    assert parse_schedule(SCHEDULE) == parsed
    assert hash(parse_schedule(SCHEDULE)) == hash(parsed)
    assert parse_schedule([[0.0, 1]]) != parsed

    for param in (5, 0.3, None):
        assert parse_schedule(param) is param


def test_a_single_entry_schedule_is_constant():
    schedule = [[5.0, 0.1]]
    assert value_at(schedule, 0.0) == 0.1
    assert value_at(schedule, 50.0) == 0.1
    graph_value = tf.function(lambda t: value_at(schedule, t))(tf.constant(50.0))
    assert float(graph_value) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "param", [5, 0.3, None, SCHEDULE, [[2000.0, 3]], [[0, 1], [50.5, 2]]]
)
def test_valid_parameters_pass_the_check(param):
    check_schedule(param, "nbit", is_count, "non-negative integers")
    wrapped = OmegaConf.create({"param": param}).param
    check_schedule(wrapped, "nbit", is_count, "non-negative integers")


@pytest.mark.parametrize(
    "param",
    [
        [],
        [[1.0, 5], [0.0, 20]],
        [[0.0, 5], [0.0, 20]],
        [[0.0, -1]],
        [[0.0, 2.5]],
        [[0.0, True]],
        [[0.0]],
        [[0.0, 1, 2]],
        [["a", 1]],
        [[True, 1]],
        [5],
    ],
)
def test_invalid_schedules_are_rejected(param):
    with pytest.raises(ValueError, match="nbit"):
        check_schedule(param, "nbit", is_count, "non-negative integers")


@pytest.mark.parametrize(
    "is_valid, accepted, rejected",
    [
        (is_count, [0, 7], [-1, 1.5, False]),
        (is_positive, [1, 0.01, 3.5], [0, -0.1, True, "1"]),
    ],
)
def test_value_checks(is_valid, accepted, rejected):
    assert all(is_valid(value) for value in accepted)
    assert not any(is_valid(value) for value in rejected)

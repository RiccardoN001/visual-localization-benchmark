import pytest

from loc.matchers import Matcher
from loc.utils import check_options, parse_options


class Dummy(Matcher):
    def __init__(self, device, max_kpts=2048, resize=None, **cfg):
        super().__init__(device, max_kpts=max_kpts, resize=resize, **cfg)


def test_values_are_parsed_as_python_literals():
    assert parse_options(["max_kpts=1024", "resize=None", "setting=turbo", "ratio=0.9"]) == {
        "max_kpts": 1024, "resize": None, "setting": "turbo", "ratio": 0.9}


def test_unknown_options_are_rejected():
    check_options(Dummy, {"max_kpts": 1})
    with pytest.raises(SystemExit):
        check_options(Dummy, {"max_keypoints": 1})

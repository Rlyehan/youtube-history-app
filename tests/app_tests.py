from typing import List

import polars as pl
import pytest

from app import get_video_keywords, get_video_length


def test_get_video_length_valid_url():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert get_video_length(url) == 212


def test_get_video_length_invalid_url():
    url = "https://www.youtube.com/watch?v=invalid_url"
    assert get_video_length(url) is None


def test_get_video_length_no_url():
    assert get_video_length(None) is None


def test_get_video_keywords_with_valid_url() -> None:
    # Test case for a valid URL
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    expected_keywords = ["rick astley", 'Never Gonna Give You Up', 'nggyu']
    actual_keywords = get_video_keywords(url)
    if actual_keywords:
        assert expected_keywords == actual_keywords[0:3]


def test_get_video_keywords_with_invalid_url() -> None:
    # Test case for an invalid URL
    url = "invalid_url"
    assert get_video_keywords(url) is None


def test_get_video_keywords_with_none_url() -> None:
    # Test case for None input
    url = None
    assert get_video_keywords(url) is None


def test_get_video_keywords_return_type() -> None:
    # Test case for checking the return type
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    keywords = get_video_keywords(url)
    assert isinstance(keywords, List) or keywords is None


@pytest.fixture
def sample_data():
    data = {
        "title": ["Title 1", "Title 2", "Title 3", "Title 4"],
        "channel": ["Channel A", "Channel B", "Channel A", "Channel C"],
        "titleUrl": ["http://url1.com", "http://url2.com", "http://url3.com", "http://url4.com"],
        "time": ["2022-02-01 12:00:00", "2022-02-02 12:00:00", "2022-02-03 12:00:00", "2022-02-04 12:00:00"],
        "month": [2, 2, 2, 2],
        "week": [5, 5, 5, 5],
        "day": ["Tuesday", "Wednesday", "Thursday", "Friday"],
        "video_length": [120, 240, 180, 300],
        "keywords": [["keyword1", "keyword2"], ["keyword2", "keyword3"], ["keyword1"], ["keyword3", "keyword4"]]
    }
    df = pl.DataFrame(data)
    return df


import numpy as np

from visualize_e16_reconstruction_results import choose_view, matched_m_labels


def test_choose_view_uses_largest_frozen_support_count():
    prediction = {
        "view_indices": np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int16)
    }
    assert choose_view(prediction) == 2


def test_m_display_colours_follow_maximum_b_support_overlap():
    b_labels = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
    m_labels = np.asarray([7, 7, 8, 9, 9, 8], dtype=np.int32)
    mapping = matched_m_labels(b_labels, m_labels)
    assert mapping[7] == 0
    assert mapping[9] == 1
    assert mapping[8] == 2

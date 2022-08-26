#!/usr/bin/env python3

import json


def flip_position(data_obj):
    flipped_data_obj = {}
    flipped_data_obj["turn"] = -data_obj["turn"]
    flipped_data_obj["winner"] = -data_obj["winner"]

    flipped_data_obj["position"] = [None] * 121
    for idx, val in enumerate(data_obj["position"]):
        r, c = idx // 11, idx % 11
        flipped_idx = c * 11 + r
        flipped_data_obj["position"][flipped_idx] = -val

    flipped_data_obj["moves_probabilities"] = [None] * 121
    for idx, val in enumerate(data_obj["moves_probabilities"]):
        r, c = idx // 11, idx % 11
        flipped_idx = c * 11 + r
        flipped_data_obj["moves_probabilities"][flipped_idx] = val

    return flipped_data_obj


def load_data_entry(path):
    with open(path, "rb") as f:
        data_obj = json.load(f)

    # Network always accept position as
    if data_obj["turn"] != 1:
        data_obj = flip_position(data_obj)
    assert data_obj["turn"] == 1

    return data_obj

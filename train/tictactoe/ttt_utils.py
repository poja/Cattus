#!/usr/bin/env python3

import json


def flip_position(data_obj):
    return {
        "turn": -data_obj["turn"],
        "winner": -data_obj["winner"],
        "position": [-x for x in data_obj["position"]],
        "moves_probabilities": data_obj["moves_probabilities"]
    }


def load_data_entry(path):
    with open(path, "rb") as f:
        data_obj = json.load(f)

    # Network always accept position as
    if data_obj["turn"] != 1:
        data_obj = flip_position(data_obj)
    assert data_obj["turn"] == 1

    return data_obj

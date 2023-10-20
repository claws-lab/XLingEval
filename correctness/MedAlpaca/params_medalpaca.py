greedy_search = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": False
}

beam_search = {
    "num_beams": 4,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": True,
}

sampling_top_k = {
    "do_sample": True,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_k": 50
}

sampling_top_p = {
    "do_sample": True,
    "top_k": 0,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_p": 0.9
}

sampling = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 512,
    "early_stopping": True,
    "top_p": 0.9
}

# sampling = {
#     "do_sample": True,
#     "top_k": 50,
#     "num_beams": 1,
#     "max_new_tokens": 512,
#     "early_stopping": True,
#     "temperature": 0.4,
#     "top_p": 0.9
# }

params = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": None,
    "top_p": 0.9
}
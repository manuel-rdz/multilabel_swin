import random


def get_mask(testing, num_labels):
    if testing:  # if testing hide all
        return range(num_labels)

    random.seed()
    num_known = random.randint(0, int(num_labels * 0.75))

    unk_mask_indices = random.sample(range(num_labels), (num_labels - num_known))

    return unk_mask_indices

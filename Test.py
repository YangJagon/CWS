import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk + start of a chunk!
        tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
        if chunk_type is None:
            chunk_type, chunk_start = tok_chunk_type, i
        elif tok_chunk_type != chunk_type or tok_chunk_class == "B" or tok_chunk_class == "S":
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = tok_chunk_type, i

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

print(set(get_chunks([0,3,3,1,2,2], {'B': 0, 'E': 1, 'S': 2, 'I': 3})))



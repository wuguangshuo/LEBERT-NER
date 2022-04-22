
def get_entities(seq):
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = i
            chunk[0] = tag.split('-')[1]
            chunk[2] = i
            if i == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = i
            if i == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1

def f1_score(label_paths, pred_paths):
    origins,founds,rights=[],[],[]
    for label_path, pre_path in zip(label_paths, pred_paths):
        label_entities = get_entities(label_path)
        pre_entities = get_entities(pre_path)
        origins.extend(label_entities)
        founds.extend(pre_entities)
        rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)
    return recall, precision, f1

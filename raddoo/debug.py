import time as _time
import contextlib as _contextlib
from .core import find, prop_eq, uniq, concat, curry


@curry(2)
def p(label, v):
    print(label, v)
    return v


@_contextlib.contextmanager
def timed(label):
    now = _time.time()

    yield

    print(label, _time.time() - now)


def diff_collections(old, new):
    diff = {
        'added': [],
        'changed': [],
        'removed': [
            old_item for old_item in old
            if not find(prop_eq.c('id', old_item['id']), new)
        ],
    }

    for new_item in new:
        old_item = find(prop_eq.c('id', new_item['id']), old)

        if not old_item:
            diff['added'].append(new_item)
        elif new_item != old_item:
            diff['changed'].append({
                'from': old_item,
                'to': new_item,
            })

    return diff


def diff_dicts(a, b):
    diff = []

    for k in uniq(concat(a.keys(), b.keys())):
        if k not in a and k in b:
            diff.append(f'key "{k}" is new. New value: {b[k]}')
        elif k not in b and k in a:
            diff.append(f'key "{k}" was removed. Old value: {a[k]}')
        elif a[k] != b[k]:
            diff.append(
                f'key "{k}" was changed. Old value: {a[k]}; New value: {b[k]}'
            )

    return diff

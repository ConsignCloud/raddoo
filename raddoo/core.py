import uuid as _uuid
import itertools as _itertools
import random as _random
import collections as _collections
import inspect as _inspect
import copy as _copy
import _collections_abc


UUID_PATTERN = "\
^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"


def _curry(n, fn, carryover_args=(), carryover_kwargs=()):
    def _curried(*args, **kwargs):
        cur_n = len(args) + len(kwargs)

        # Rebuild the carryover dict to avoid mutation
        combined_args = carryover_args + args
        combined_kwargs = {}
        combined_kwargs.update(carryover_kwargs)
        combined_kwargs.update(kwargs)

        if cur_n >= n:
            return fn(*combined_args, **combined_kwargs)
        else:
            return _curry(n - cur_n, fn, combined_args, combined_kwargs)

    _curried._arity = n

    return _curried


def curry(n):
    def _curry_fn(fn):
        fn.c = _curry(n, fn)

        return fn

    return _curry_fn


def partial(fn, *args, **kwargs):
    def _partialed(*other_args, **other_kwargs):
        return fn(*(args + other_args), **merge(kwargs, other_kwargs))

    return _partialed


def arity(fn):
    return getattr(fn, '_arity', len(_inspect.signature(fn).parameters))


@curry(2)
def n_ary(arity, fn):
    return lambda *args: fn(*args[:arity])


def self_arity(fn):
    return n_ary(arity(fn), fn)


def unary(fn):
    return n_ary(1, fn)


@curry(2)
def mapl(fn, data):
    return (
        {key: fn(value) for key, value in data.items()}
        if type(data) == dict else
        [fn(item) for item in data]
    )


@curry(2)
def filterl(fn, data):
    return (
        {key: value for key, value in data.items() if fn(value)}
        if type(data) == dict else
        [item for item in data if fn(item)]
    )


@curry(3)
def reducel(fn, value, data):
    for item in data:
        value = fn(value, item)

    return value


@curry(2)
def reject(fn, data):
    return (
        {key: value for key, value in data.items() if not fn(value)}
        if type(data) == dict else
        [item for item in data if not fn(item)]
    )


@curry(3)
def when(test, fn, x):
    return fn(x) if test(x) else x


@curry(2)
def find(fn, data):
    for item in data:
        if fn(item):
            return item


@curry(2)
def find_index(fn, xs):
    for idx, x in enumerate(xs):
        if fn(x):
            return idx


def uniq(data):
    return list(set(data))


@curry(2)
def uniq_by(fn, data):
    result = _collections.OrderedDict()
    for item in data:
        key = fn(item)
        result.setdefault(key, item)

    return list(result.values())


@curry(2)
def index_by(get_key, coll):
    return {get_key(item): item for item in coll}


@curry(2)
def index_of(value, xs):
    if type(xs) != dict:
        xs = enumerate(xs)

    for k, v in xs:
        if v == value:
            return k


@curry(2)
def equals(value1, value2):
    return value1 == value2


@curry(2)
def identical(value1, value2):
    return value1 is value2


@curry(2)
def lt(a, b):
    return a < b


@curry(2)
def lte(a, b):
    return a <= b


@curry(2)
def gt(a, b):
    return a > b


@curry(2)
def gte(a, b):
    return a >= b


@curry(2)
def prop(prop_name, obj):
    try:
        return obj[prop_name]
    except (IndexError, KeyError):
        return None
    except TypeError:
        return getattr(obj, prop_name, None)


@curry(2)
def get_path(_path, obj):
    while _path:
        head, *_path = _path
        obj = prop(head, obj)

    return obj


@curry(3)
def prop_eq(prop_name, val, obj):
    return prop(prop_name, obj) == val


@curry(3)
def prop_ne(prop_name, val, obj):
    return prop(prop_name, obj) != val


@curry(3)
def path_eq(path, val, obj):
    return get_path(path, obj) == val


@curry(3)
def path_ne(path, val, obj):
    return get_path(path, obj) != val


@curry(2)
def concat(*lists):
    return [item for sublist in lists for item in sublist]


def flatten(lists):
    if not hasattr(lists, '__iter__') or isinstance(lists, (dict, str)):
        return [lists]

    r = []
    for x in lists:
        r.extend(flatten(x))

    return r


@curry(2)
def without(sans, xs):
    sans = set(sans)

    return [x for x in xs if x not in sans]


@curry(2)
def contains(x, xs):
    return x in xs


@curry(2)
def starts_with(x, xs):
    return xs.startswith(x)


@curry(2)
def split(delimiter, value):
    return value.split(delimiter)


@curry(2)
def join(delimiter, value):
    return delimiter.join(value)


def merge_all(dicts):
    result = {}
    for d in dicts:
        result.update(d)

    return result


@curry(2)
def merge(*dicts):
    result = {}
    for d in dicts:
        result.update(d)

    return result


@curry(2)
def zip_dict(keys, values):
    result = {}
    for idx, key in enumerate(keys):
        result[key] = values[idx]

    return result


@curry(2)
def pick(keys, data):
    return {key: data[key] for key in keys if key in data}


@curry(2)
def omit(keys, data):
    result = {}
    for key, value in data.items():
        if key in keys:
            continue

        result[key] = value

    return result


@curry(3)
def assoc(key, value, data):
    result = _copy.copy(data)

    if type(data) == dict:
        result.update({key: value})
    else:
        setattr(result, key, value)

    return result


@curry(3)
def assoc_path(_path, value, data):
    if not hasattr(data, '__setitem__'):
        data = {}

    head, *tail = _path

    value = assoc_path(tail, value, data.get(head, {})) if tail else value

    result = _copy.copy(data)
    result[head] = value

    return result


@curry(2)
def pluck(key, coll):
    return mapl(prop.c(key), coll)


@curry(2)
def all_fn(fn, coll):
    for x in coll:
        if not fn(x):
            return False

    return True


@curry(2)
def any_fn(fn, coll):
    for x in coll:
        if fn(x):
            return True

    return False


@curry(2)
def none_fn(fn, coll):
    for x in coll:
        if fn(x):
            return False

    return True


@curry(2)
def all_pass(fx, x):
    for fn in fx:
        if not fn(x):
            return False

    return True


@curry(2)
def any_pass(fx, x):
    for fn in fx:
        if fn(x):
            return True

    return False


@curry(2)
def none_pass(fx, x):
    for fn in fx:
        if fn(x):
            return False

    return True


@curry(2)
def none(fn, coll):
    for x in coll:
        if fn(x):
            return False

    return True


@curry(2)
def where(spec, obj):
    for key, fn in spec.items():
        if not fn(prop(key, obj)):
            return False

    return True


@curry(2)
def where_eq(spec, obj):
    for key, value in spec.items():
        if value != prop(key, obj):
            return False

    return True


@curry(2)
def group_by(fn, coll):
    result = {}
    for item in coll:
        key = fn(item)
        result.setdefault(key, [])
        result[key].append(item)

    return result


@curry(2)
def count_by(fn, coll):
    result = {}
    for item in coll:
        key = fn(item)
        result.setdefault(key, 0)
        result[key] += 1

    return result


def pipe(*fns):
    def _piped(value):
        for fn in fns:
            value = fn(value)

        return value

    return _piped


@curry(3)
def replace(find, replace, value):
    return value.replace(find, replace)


def reverse(xs):
    return list(reversed(xs))


def always(val):
    return lambda *args, **kwargs: val


def identity(val, *args, **kwargs):
    return val


def difference(l1, l2):
    return set(l1).difference(l2)


def intersection(l1, l2):
    return set(l1).intersection(l2)


def union(l1, l2):
    return set(l1).union(l2)


@curry(2)
def nth(idx, xs):
    return xs[idx]


def last(xs):
    return xs[-1] if xs else None


@curry(2)
def add(x, y):
    return x + y


def inc(x):
    return x + 1


@curry(2)
def subtract(x, y):
    return x - y


def dec(x):
    return x - 1


@curry(2)
def dict_of(k, v):
    return {k: v}


def values(m):
    return m.values()


def keys(m):
    return m.keys()


@curry(2)
def partition(f, xs):
    a = []
    b = []

    for x in xs:
        if f(x):
            a.append(x)
        else:
            b.append(x)

    return a, b


@curry(2)
def sort_by(f, xs):
    return sorted(xs, key=f)


@curry(2)
def default_to(default, x):
    return default if x is None else x


@curry(2)
def drop(n, xs):
    return _itertools.islice(xs, n, None)


@curry(2)
def take(n, xs):
    return _itertools.islice(xs, n)


def complement(f):
    return lambda *a, **kw: not f(*a, **kw)


def noop(*args, **kwargs):
    pass


def first(xs):
    # Handle sequences that don't support indexing
    for x in xs:
        return x


def random_uuid():
    return str(_uuid.uuid4())


def ensure_list(x):
    t = type(x)

    if t == list:
        return x

    if t in {set, _collections_abc.dict_keys, _collections_abc.dict_values}:
        return list(x)

    return [x]


def concat_all(lists):
    return concat(*lists)


def safe_divide(x, y):
    return x / y if y != 0 else 0.0


def is_none(x):
    return x is None


def is_not_none(x):
    return x is not None


def slurp(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def spit(path, s, mode="w+"):
    with open(path, mode) as f:
        f.write(s)


@curry(2)
def create_map(key, collection):
    return index_by(prop.c(key), collection)


@curry(3)
def create_map_of(key, value_key, collection):
    result = {}
    for item in collection:
        result[prop(key, item)] = prop(value_key, item)

    return result


def union_keys(*dicts):
    result = set()
    for d in dicts:
        result = result.union(set(d.keys()))

    return result


@curry(2)
def merge_right(*d):
    return merge(*reversed(d))


@curry(3)
def merge_in(key, overrides, data):
    return update_in(key, merge_right.c(overrides), data)


@curry(3)
def merge_in_right(key, overrides, data):
    return update_in(key, merge.c(overrides), data)


@curry(2)
def merge_right_fn(result, d2):
    """
    Merge defaults in but lazily evaluate them in case they're expensive
    """
    for key, get_value in d2.items():
        if key not in result:
            result = merge(result, {key: get_value()})

    return result


@curry(3)
def rename_prop(from_prop, to_prop, data):
    return omit([from_prop], merge(data, {to_prop: data[from_prop]}))


@curry(3)
def copy_prop(from_prop, to_prop, data):
    return merge(data, {to_prop: data[from_prop]})


@curry(3)
def copy_path(from_path, to_path, data):
    return assoc_path(to_path, get_path(from_path, data), data)


@curry(3)
def rename_path(from_path, to_path, data):
    # find the value
    value = get_path(from_path, data)

    # Remove it from the data structure
    update_path(from_path[:-1], omit.c(from_path[-1]), data)

    # Stick it where it belongs
    return assoc_path(to_path, value, data)


@curry(3)
def update_path(path, fn, data):
    value = get_path(path, data)
    parent = get_path(path[:-1], data) if len(path) > 1 else data
    args = [value, parent, data][:arity(fn)]

    return assoc_path(path, fn(*args), data)


@curry(3)
def update_in(key, fn, data):
    return update_path([key], fn, data)


def pop(key, data):
    return data.get(key), omit([key], data)


@curry(2)
def modify_keys_recursive(fn, value):
    if isinstance(value, list):
        return mapl(modify_keys_recursive.c(fn), value)

    if isinstance(value, dict):
        return zip_dict(
            mapl(fn, value.keys()),
            mapl(modify_keys_recursive.c(fn), value.values())
        )

    return value


@curry(2)
def modify_values_recursive(fn, value):
    if isinstance(value, list):
        return mapl(modify_values_recursive.c(fn), value)

    if isinstance(value, dict):
        return zip_dict(
            value.keys(),
            mapl(modify_values_recursive.c(fn), value.values())
        )

    return fn(value)


def do_pipe(value, fns):
    for fn in fns:
        if callable(fn):
            value = fn(value)
        else:
            fn, *args = fn
            value = fn(*args + [value])

    return value


@curry(2)
def fill(keys, value):
    return {key: value for key in keys}


def ichunk(n, iterable):
    iterator = iter(iterable)

    while True:
        chunk = list(_itertools.islice(iterator, n))

        if not chunk:
            return

        yield chunk


def do_all(iterable):
    for x in iterable:
        pass


class Obj(object):
    def __init__(self, **data):
        self.data = data

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(name)


def ensure_bytes(s, encoding='utf-8'):
    try:
        return s.encode(encoding)
    except AttributeError:
        return s


def ensure_str(s, encoding='utf-8'):
    try:
        return s.decode(encoding)
    except AttributeError:
        return s


def listify(*xs):
    return [x for x in flatten(xs) if x is not None]


def eager(fn):
    def wrapper(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return wrapper


def switcher(k, m):
    if k in m:
        return m[k]

    if 'default' in m:
        return m['default']

    raise ValueError(f'Unknown key for switcher: {k}')


def switcher_fn(k: str, m: dict):
    """
    Calls function in m which is the value of key k.

    Example::

        swtcher_fn('a', {
            'a': lambda: print('first'),
            'b': lambda: print('second')
        })
        # Output: first
    """
    f = switcher(k, m)

    return f()


def strip(v):
    return v.strip()


def ffirst(x):
    try:
        return x[0][0]
    except KeyError:
        pass


@curry(2)
def plop(coll, k):
    return prop(k, coll)


@curry(2)
def contained(xs, x):
    return x in xs


# https://stackoverflow.com/a/13018842/1467342
def bidi_hash(n):
    return ((0x0000FFFF & n) << 16) + ((0xFFFF0000 & n) >> 16)


@curry(2)
def mapcat(f, xs):
    return [y for x in xs for y in f(x)]


def subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from subclasses(subclass)


@curry(2)
def pick_values(ks, x):
    return [x.get(k, None) for k in ks]


def shuffled(xs):
    xs = list(xs)

    _random.shuffle(xs)

    return xs

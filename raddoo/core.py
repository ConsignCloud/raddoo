import uuid, re
from copy import copy
from collections import OrderedDict
from inspect import signature
from itertools import islice
from contextlib import contextmanager
import itertools, os, sys, random, time
from _collections_abc import dict_keys, dict_values


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
    return getattr(fn, '_arity', len(signature(fn).parameters))


@curry(2)
def n_ary(arity, fn):
    return lambda *args: fn(*args[:arity])


def self_arity(fn):
    return n_ary(arity(fn), fn)


def unary(fn):
    return n_ary(1, fn)


def binary(fn):
    return n_ary(2, fn)


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
    result = OrderedDict()
    for item in data:
        key = fn(item)
        result.setdefault(key, item)

    return list(result.values())


@curry(2)
def index_by(get_key, coll):
    return {get_key(item): item for item in coll}


@curry(2)
def index_of(item, coll):
    for idx, value in enumerate(coll):
        if value == item:
            return idx

    return -1


@curry(2)
def map_obj_indexed(fn, obj):
    fn = self_arity(fn)

    return {key: fn(value, key, obj) for key, value in obj.items()}


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
def path_eq(_path, val, obj):
    return path(_path, obj) == val


@curry(2)
def concat(*l):
    return [item for sublist in l for item in sublist]


def flatten(l):
    if not hasattr(l, '__iter__') or isinstance(l, (dict, str)):
        return [l]

    r = []
    for x in l:
        r.extend(flatten(x))

    return r


@curry(2)
def without(sans, target):
    sans = set(sans)

    return [element for element in target if element not in sans]


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
def zip_obj(keys, values):
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
    result = copy(data)

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

    result = copy(data)
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


def invoker(arity, method_name):
    @curry(arity)
    def invoke(*args):
        obj = args[-1]
        method = getattr(obj, method_name)

        return method(*args[:-1])

    return invoke


@curry(3)
def replace(find, replace, value):
    return value.replace(find, replace)


def reverse(xs):
    return list(reversed(xs))


def always(val):
    return lambda *args, **kwargs: val


def identity(val):
    return val


def difference(l1, l2):
    return [item for item in l1 if item not in l2]


@curry(3)
def flip(fn, arg1, arg2):
    return fn(arg2, arg1)


@curry(2)
def nth(idx, xs):
    return xs[idx]


def last(l):
    return l[-1] if l else None


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
def obj_of(k, v):
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
    return islice(xs, n, None)


@curry(2)
def take(n, xs):
    return islice(xs, n)


def complement(f):
    return lambda *a, **kw: not f(*a, **kw)


def noop(*args, **kwargs):
    pass


def first(xs):
    # Handle sequences that don't support indexing
    for x in xs:
        return x


def uuid4():
    return str(uuid.uuid4())


def ensure_list(x):
    t = type(x)

    if t == list:
        return x

    if t in {set, dict_keys, dict_values}:
        return list(x)

    return [x]


def concat_all(lists):
    return concat(*lists)


def diff_collections(old, new):
    diff = {
        'added': [],
        'changed': [],
        # Anything in old that's gone in new
        'removed': [
            old_item for old_item in old
            if not find(prop_eq.c('id', old_item['id']), new)
        ],
    }

    for new_item in new:
        old_item = find(prop_eq.c('id', new_item['id']), old)

        # Anything in new that's gone in old
        if not old_item:
            diff['added'].append(new_item)
        # Anything that changed but is still around
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
    return modify(data, omit=[from_prop], overrides={to_prop: data[from_prop]})


@curry(3)
def copy_prop(from_prop, to_prop, data):
    return merge(data, {to_prop: data[from_prop]})


@curry(3)
def copy_path(from_path, to_path, data):
    return assoc_path(to_path, path(from_path, data), data)


@curry(3)
def rename_path(from_path, to_path, data):
    # find the value
    value = path(from_path, data)

    # Remove it from the data structure
    update_path(from_path[:-1], omit.c(from_path[-1]), data)

    # Stick it where it belongs
    return assoc_path(to_path, value, data)


@curry(3)
def update_path(path, fn, data):
    value = path(path, data)
    parent = path(path[:-1], data) if len(path) > 1 else data
    args = [value, parent, data][:arity(fn)]

    return assoc_path(path, fn(*args), data)


@curry(3)
def update_in(key, fn, data):
    return update_path([key], fn, data)


def extract(key, data):
    return data.get(key), omit([key], data)


@curry(2)
def modify_keys_recursive(fn, value):
    if isinstance(value, list):
        return mapl(modify_keys_recursive.c(fn), value)

    if isinstance(value, dict):
        return zip_obj(
            mapl(fn, value.keys()),
            mapl(modify_keys_recursive.c(fn), value.values())
        )

    return value


@curry(2)
def modify_values_recursive(fn, value):
    if isinstance(value, list):
        return mapl(modify_values_recursive.c(fn), value)

    if isinstance(value, dict):
        return zip_obj(
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
def clog(label, v):
    print(label, v)
    return v


@curry(2)
def fill_dict(keys, value):
    return {key: value for key in keys}


@curry(2)
def find_key(value, d):
    for k, v in d:
        if v == value:
            return k


def ichunk(n, iterable):
    iterator = iter(iterable)

    while True:
        chunk = list(itertools.islice(iterator, n))

        if not chunk:
            return

        yield chunk


def do_all(iterable):
    for x in iterable:
        pass


def parse_bool(value):
    if type(value) == bool:
        return value

    value = str(value).lower()

    if value in ['1', 'yes', 'true', 'y', 't']:
        return True
    elif value in ['0', 'no', 'false', 'n', 'f']:
        return False


def delimit(values, conjunction='and'):
    if len(values) == 0:
        return ''
    elif len(values) == 1:
        return values[0]
    elif len(values) == 2:
        return "{} {} {}".format(values[0], conjunction, values[1])
    elif len(values) > 8:
        return "{}, {} {} others".format(
            ", ".join(values[:6]),
            conjunction,
            len(values) - 6
        )

    return "{}, {} {}".format(", ".join(values[:-1]), conjunction, values[-1])


def pluralize(n, label, pluralLabel=None):
    return label if n == 1 else (pluralLabel or '{}s'.format(label))


class Obj(object):
    def __init__(self, **data):
        self.__data = data

    def __getattr__(self, name):
        if name == '__data':
            return super(Obj, self).__getattr__(name)

        try:
            return self.__data[name]
        except KeyError:
            raise AttributeError(name)


@curry(3)
def prop_ne(prop_name, val, obj):
    return prop(prop_name, obj) != val


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


def ellipsize(s, l, suffix='...'):
    if len(s) < l * 1.1:
        return s

    while len(s) > l and ' ' in s:
        s, *_ = s.rpartition(' ')

    return s + suffix


def to_snake(value):
    return re.sub(
        '([a-z0-9])([A-Z])',
        r'\1_\2',
        re.sub(
            '(^_)_*([A-Z][a-z]+)',
            r'\1_\2',
            re.sub(r' +', '_', value),
        )
    ).lower()


def to_human(value):
    return to_snake(value).replace("_", " ").title()


def to_kebab(value):
    return to_snake(value).replace('_', '-')


def to_screaming_snake(value):
    return to_snake(value).upper()


def to_camel(value):
    first, *rest = to_snake(value).split('_')

    return first + "".join([word.capitalize() for word in rest])


def to_pascal(value):
    return "".join([word.capitalize() for word in to_snake(value).split('_')])


def switcher(k, m):
    if k in m:
        return m[k]

    if 'default' in m:
        return m['default']

    raise ValueError(f'Unknown key for switcher: {k}')


def switcher_fn(k, m):
    f = switcher(k)

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


def env(k, **kw):
    if 'default' in kw and k not in os.environ:
        return kw['default']

    v = os.environ[k]

    if v.lower() == 'true':
        return True

    if v.lower() == 'false':
        return False

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    return v


# https://stackoverflow.com/a/13018842/1467342
def bidi_hash(n):
    return ((0x0000FFFF & n) << 16) + ((0xFFFF0000 & n) >> 16)


class Progress(object):
    width = 80

    def __init__(self, total):
        self.total = max(1, total)

    def show(self, i):
        j = round(i) / self.total
        spinner = ["# ", " #"][round(i) % 2]
        filled_w = self.width * j
        empty_w = self.width - filled_w
        fill = '=' * round(filled_w)
        space = ' ' * round(empty_w)

        sys.stdout.write('\r')
        sys.stdout.write(f"{spinner}[{fill}{space}] {int(100 * j)}%")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * (self.width + 8))
        sys.stdout.write('\r')
        sys.stdout.flush()


@curry(2)
def mapcat(f, xs):
    return [y for x in xs for y in f(x)]


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from get_subclasses(subclass)


@curry(2)
def pick_values(ks, x):
    return [x.get(k, None) for k in ks]


def shuffled(xs):
    xs = list(xs)

    random.shuffle(xs)

    return xs


@contextmanager
def timed(label):
    now = time.time()

    yield

    print(label, time.time() - now)

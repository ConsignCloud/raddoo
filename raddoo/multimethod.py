from .core import partial, identity

Default = object()

no_method_t = 'No method found for "{}" and dispatch value "{}"'
not_callable_t = "Method provided to {} for {} is not callable (got {})"


class CompositeMethod(object):
    """
    This is a way to register multiple function to a single dispatch key.
    If mode == replace, no value will be returned, so this should only be
    used for effect-ful functions.
    """
    def __init__(self, *methods):
        self.methods = methods

    def __iter__(self):
        return iter(self.methods)

    def __call__(self, *args, **kwargs):
        for fn in self.methods:
            fn(*args, **kwargs)


class MultiMethod(object):
    def __init__(self, name, dispatch=identity, mode='replace', cache=None):
        self.name = name
        self.dispatch = dispatch
        self.decorator = identity
        self.methods = {}
        self.mode = mode
        self.cache = cache

        assert self.mode in {'replace', 'append'}

    def __repr__(self):
        return "<MultiMethod {}>".format(self.name)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        # Nested try-excepts are so we give the caller a more helpful message
        # with the dispatch value in it
        try:
            method = self.get_method(self.dispatch(*args, **kwargs))
        except KeyError as orig_exc:
            try:
                method = self.get_method(Default)
            except KeyError:
                raise orig_exc

        return method(*args, **kwargs)

    def call_default(self, *args, **kwargs):
        method = self.get_method(Default)

        return method(*args, **kwargs)

    def add_method(self, value, fn):
        if not callable(fn):
            raise ValueError(not_callable_t.format(self.name, value, fn))

        if self.cache:
            fn = self.cache(fn)

        self.methods[value] = self.decorator(
            fn if self.mode == 'replace' else
            CompositeMethod(*self.methods.get(value, []), fn)
        )

        return self

    def add_default_method(self, fn):
        return self.add_method(Default, fn)

    def method(self, value):
        return partial(self.add_method, value)

    def default_method(self):
        return self.method(Default)

    def get_method(self, value):
        try:
            return self.methods[value]
        except KeyError:
            raise KeyError(no_method_t.format(self.name, value))

    def get_default_method(self):
        return self.get_method(Default)

    def call_default_method(self, *args, **kw):
        return self.get_method(Default)(*args, **kw)

    def remove_method(self, value):
        del self.methods[value]

    def remove_default_method(self):
        self.remove_method(Default)

    def decorate(self, decorator):
        self.decorator = decorator


def defmulti(name, **kwargs):
    return partial(MultiMethod, name, **kwargs)

import contextlib as _ctx
from .core import get_path, merge, Obj


class Context(Obj):
    ns = NotImplemented

    def __init__(self, ns=None, **data):
        self.ns = ns or self.__class__.ns
        self.data = data

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.data.__str__())

    def clone(self, **data):
        return self.__class__(**merge(self.data, data))

    def get(self, *path):
        try:
            return get_path(path, self.data)
        except (AttributeError, KeyError):
            pass

    def register(self, force=False):
        if self.ns in g.data and not force:
            raise KeyError('{} is already registered'.format(self.ns))

        g.data[self.ns] = self

        return self

    def unregister(self):
        del g.data[self.ns]

        return self

    def update(self, **data):
        self.data = merge(self.data, data)
        self.unregister()
        self.register()

    def on_exception(self, exc):
        pass

    @_ctx.contextmanager
    def scoped(self):
        with scoped(self):
            yield


@_ctx.contextmanager
def scoped(*ctxs):
    outer = {}
    for ctx in ctxs:
        outer[ctx.ns] = g.data.get(ctx.ns)

        if outer[ctx.ns]:
            outer[ctx.ns].unregister()

        ctx.register()

    try:
        yield
    except Exception as exc:
        for ctx in ctxs:
            ctx.on_exception(exc)

        raise
    finally:
        for ctx in ctxs:
            ctx.unregister()

            if outer[ctx.ns]:
                outer[ctx.ns].register()


class Global(Obj):
    def __setattr__(self, name, value):
        if name != 'data':
            self.data[name] = value

        super(Global, self).__setattr__(name, value)

    def get(self, name, *path):
        ctx = self.data.get(name)

        return ctx.get(*path) if ctx else None

    def clear(self):
        self.data.clear()

    def unregister(self, *keys):
        for key in keys:
            if key in self.data:
                self.data[key].unregister()


g = Global()

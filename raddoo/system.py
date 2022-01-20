import os, sys
from datetime import datetime
from .text import parse_bool


def now():
    return datetime.utcnow().isoformat() + 'Z'


def env(k, **kw):
    if 'default' in kw and k not in os.environ:
        return kw['default']

    v = os.environ[k]

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    try:
        return parse_bool(v)
    except ValueError:
        pass

    return v


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

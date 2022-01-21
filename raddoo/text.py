import re


def parse_bool(value):
    if type(value) == bool:
        return value

    value = str(value).lower()

    if value in ['1', 'yes', 'true', 'y', 't']:
        return True
    elif value in ['0', 'no', 'false', 'n', 'f']:
        return False

    raise ValueError(value)


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


def ellipsize(text, length, suffix='...'):
    if len(text) < length * 1.1:
        return text

    while len(text) > length and ' ' in text:
        text, *_ = text.rpartition(' ')

    return text + suffix


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


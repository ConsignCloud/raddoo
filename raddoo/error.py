class Error(Exception):
    def __init__(self, code, message, meta={}):
        self.code = code
        self.message = message
        self.meta = meta

    def __repr__(self):
        return '<{} {}: {}>'.format(
            self.__class__.__name__,
            self.code,
            self.message
        )

    def has_code(self, code):
        return code in self.code

    def has_message(self, substring):
        return substring in self.message

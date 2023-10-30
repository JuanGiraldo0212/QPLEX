from collections.abc import Mapping


class Options(Mapping):
    __slots__ = "_fields"

    def __init__(self, **kwargs):
        # self._fields = kwargs
        super().__setattr__("_fields", kwargs)

    def __getitem__(self, key):
        return self._fields[key]

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            # raise AttributeError(f"Option {name} is not defined") from ex
            return None

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def __setattr__(self, key, value):
        self._fields[key] = value

    def __setitem__(self, key, value):
        self.update_options(**{key: value})

    def update_options(self, **fields):
        self._fields.update(fields)

    def __repr__(self):
        items = [f"{k}={v!r}" for k, v in self._fields.items()]
        return f"{type(self).__name__}({', '.join(items)})"

import logging
lg = logging.getLogger(__name__)

class LoggingDict(dict):
    def setdefault(self, key, default = None):
        if key not in self: 
            lg.warning(f'{key} not set: default value used ({default}).')
        return super().setdefault(key, default)


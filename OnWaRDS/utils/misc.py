import logging
lg = logging.getLogger(__name__)

class LoggingDict(dict):
    def setdefault(self, key, default = None):
        if key not in self: 
            lg.warning(f'{key} not set: default value used ({default}).')
        return super().setdefault(key, default)
    # ------------------------------------------------------------------------ #

def dict2txt(my_dict, filename):
    from datetime import datetime
    with open(filename, "w") as fid:
        date_id_str = datetime.now().strftime("%d/%m/%Y, %H:%M")
        fid.write('# OnWaRDS input file (generated on the {})\n'.format(date_id_str))
        fid.write(dict2txtbuffer(my_dict, 1))
    # ------------------------------------------------------------------------ #

def dict2txtbuffer(my_dict, level):
    buffer = ''

    keyFields  = [key for key in my_dict if not isinstance(my_dict[key],dict)]
    dictFields = [key for key in my_dict if     isinstance(my_dict[key],dict)]
    
    for key in keyFields:
        buffer += '{} = {}\n'.format(key, my_dict[key], level)

    for subDictKey in dictFields:
        buffer += '\n' + '['*level + subDictKey + ']'*level + '\n'
        buffer += dict2txtbuffer(my_dict[subDictKey], level+1)
    
    return buffer
    # ------------------------------------------------------------------------ #

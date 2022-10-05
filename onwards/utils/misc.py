# Copyright (C) <2022> <Université catholique de Louvain (UCLouvain), Belgique>

# List of the contributors to the development of OnWaRDS: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OnWaRDS) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.
 
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

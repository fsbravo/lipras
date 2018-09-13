"""
I/O for NMR-STAR files.
"""

import pandas as pd
import re
from collections import OrderedDict, defaultdict


def careful_split(string, delimiter='\s'):

    """
    Function to split on delimiter but not within substrings.

    Defaults to white-space splitting.

    E.g. Given
        s = "'chemical shift assignment' 16006 7 "
    Then
        >>> t = careful_split(s, None)
    Returns
        t = ['chemical shift assignment', '16006', '7']
    rather than
        t = ['chemical', 'shift', 'assignment', '16006', '7']
    """

    r = re.compile(r"(?:['\"](?P<substring>.*?)['\"]|(?P<normal>[\S]+))\s*")
    return [match.group('substring') if match.group('substring') else
            match.group('normal') for match in r.finditer(string)]


class NmrStarSaveBlock(object):

    """
    A SAVE block from an NMR-STAR file.
    """

    def __init__(self, name):

        self.name = name
        self.properties = {}
        self.loops = {}

    def add_property(self, var, value):

        self.properties[var] = value

    def add_loop(self, name, data, columns):

        self.loops[name] = pd.DataFrame(data, columns=columns)

    def __getitem__(self, k):

        if k in self.properties.keys():
            return self.properties[k]
        elif k in self.loops.keys():
            return self.loops[k]

        raise KeyError('No such property or loop in this save block.')

    def __str__(self):

        properties = ', '.join(
            ['{}={}'.format(k, v) for k, v in self.properties.iteritems()])

        loops = ', '.join(
            ['{}=DataFrame{}'.format(k, v.shape) for k, v in self.loops.iteritems()])

        return 'SaveBlock({}):\n\tproperties: \n\t\t{}\n\tloops: \n\t\t{}'.format(
            self.name, properties, loops)

    def __repr__(self):

        properties = '\n\t\t'.join(
            ['{:35} {}'.format(k, v) for k, v in self.properties.iteritems()])

        loops = '\n\t\t'.join(
            ['{:35} DataFrame{}'.format(k, v.shape) for k, v in self.loops.iteritems()])

        return 'SaveBlock({}):\n\tproperties: \n\t\t{}\n\tloops: \n\t\t{}'.format(
            self.name, properties, loops)

    def short_string(self):

        return 'SaveBlock({}) with {} properties and {} loops'.format(
            self.name, len(self.properties), len(self.loops))


class NmrStarMultiSave(list):

    """
    A list of multiple save blocks.
    """

    def __init__(self, name):

        self.name = name
        super(NmrStarMultiSave, self).__init__()

    def __str__(self):

        return '{} save blocks\n\t\t{}'.format(
            len(self), '\n\t\t'.join([c.short_string() for c in self]))

    def __repr__(self):

        return '{} save blocks\n\t\t{}'.format(
            len(self), '\n\t\t'.join([c.short_string() for c in self]))

    def short_string(self):

        return self.__str__()


class NmrStarAccessor(OrderedDict):

    """
    Accessor for NMR-STAR file format.

    Parses file into human-readable format and presents it as object
    that can be accessed like a dictionary.
    """

    def __init__(self, file):

        super(NmrStarAccessor, self).__init__()

        self.file = file
        self.__parse()      # parses file into native Python structures
        self.__id__ = int(self['Entry']['ID'])

    @property
    def id(self):

        return self.__id__

    def __parse(self):

        # flags
        IN_LOOP = False
        IN_SAVE = False
        IN_LABL = False

        entry_p = re.compile('_(?P<name>\S+?)\.(?P<var>\S+)\s*(?P<value>\S+)?')

        # process
        with open(self.file, 'r') as f:

            c_save = OrderedDict()
            c_name = ''
            l_vars = []
            l_name = ''
            l_data = []

            for line in f:
                line = line.strip()
                # ### special block-denomination lines
                # save_ block start
                if line.startswith('save_') and not IN_SAVE:
                    IN_SAVE = True
                    c_save = NmrStarSaveBlock(line.split('save_')[1])
                    c_name = ''
                    continue
                # loop_ block start
                elif line.startswith('loop_'):
                    IN_LOOP = True
                    l_vars = []
                    l_name = ''
                    l_data = []
                    continue
                # loop_ block end
                elif line.startswith('stop_'):
                    IN_LOOP = False
                    c_save.add_loop(l_name, l_data, l_vars)
                    continue
                # save_ block end
                elif line.startswith('save_') and IN_SAVE:
                    IN_SAVE = False
                    if c_name in self.keys():
                        if not isinstance(self[c_name], NmrStarMultiSave):
                            tmp = self[c_name]
                            self[c_name] = NmrStarMultiSave(c_name)
                            self[c_name].append(tmp)
                        self[c_name].append(c_save)
                    else:
                        self[c_name] = c_save
                    continue
                # ### regular (data) line
                # line within loop
                if IN_LOOP:
                    if line == '':
                        continue
                    # loop variable definition
                    elif line.startswith('_'):
                        r = entry_p.search(line)
                        l_name = r.group('name')
                        l_vars.append(r.group('var'))
                    # loop data line
                    else:
                        l_data.append(careful_split(line))
                # multi-line entry data
                elif line.startswith(';'):
                    IN_LABL = not IN_LABL
                    if not IN_LABL:
                        c_save.add_property(var, value)
                        continue
                    value = ''
                # regular entry in save_ block
                elif line.startswith('_'):
                    r = entry_p.search(line)
                    c_name = r.group('name')
                    var, value = r.group('var'), r.group('value')
                    c_save.add_property(var, value)
                elif IN_LABL:
                    if len(value) != 0:
                        value += ' '
                    value += line

    def __str__(self):

        return 'NMR-STAR file:\n\t' + '\n\t'.join(
            [str(k) + ' :::' + v.short_string() for k, v in self.iteritems()])

    def __repr__(self):

        return 'NMR-STAR file:\n\t' + '\n\t'.join(
            [str(k) + ' ::: ' + v.short_string() for k, v in self.iteritems()])

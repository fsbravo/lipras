import numpy as np


######################################
# SPIN SYSTEMS #######################
######################################

class SpinSystem(object):

    def __init__(self, value, i, empty=False):

        self.value = value
        if empty:
            self.value *= np.nan
            self.empty = True
            self.__u__ = []
        else:
            self.empty = False
            self.__u__ = [i]
        self.__id__ = i

    @property
    def u(self):

        return self.__u__

    @property
    def id(self):

        return self.__id__

    def __repr__(self):

        return 'SpinSystem({}, id={})'.format(self.value, self.id)

    def __str__(self):

        return 'SpinSystem({}, id={})'.format(self.value, self.id)


class SpinSystemSet(list):

    def __init__(self, data, scheme):

        super(SpinSystemSet, self).__init__()

        self.scheme = scheme

        for i, values in enumerate(data):
            self.append(SpinSystem(values, i))

        # add empty spin system
        self.append(SpinSystem(np.ones(values.shape)*np.nan, i+1, empty=True))

        self.__value__ = data

        self.__u__ = []
        for spin in self:
            self.__u__ += spin.u

    @property
    def value(self):

        return self.__value__

    @property
    def u(self):

        return self.__u__

    def __repr__(self):

        return 'Spin system set with {} spin systems.\n\tScheme: {}'.format(
            len(self)-1, self.scheme)

    def __str__(self):

        return 'Spin system set with {} spin systems.\n\tScheme: {}'.format(
            len(self)-1, self.scheme)
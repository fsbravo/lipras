import numpy as np

######################################
# SPIN SYSTEMS #######################
######################################

class SpinSystem(object):

    def __init__(self, value, i, u, empty=False):

        self.value = value
        if empty:
            self.value *= np.nan
            self.empty = True
        else:
            self.empty = False
        self.__u__ = u
        self.__id__ = i

    @property
    def u(self):

        return self.__u__

    @property
    def id(self):

        return self.__id__

    def __repr__(self):

        return 'SpinSystem({}, id={})'.format(
            super(SpinSystem, self).__repr__(), self.id)

    def __str__(self):

        return 'SpinSystem({}, id={})'.format(
            super(SpinSystem, self).__str__(), self.id)


class SpinSystemSet(list):

    def __init__(self, data):

        super(SpinSystemSet, self).__init__()

        u_max = data.shape[0]
        for i, values in enumerate(data):
            u = np.zeros(u_max, dtype=np.int32)
            u[i] = 1
            self.append(SpinSystem(values, i, u))

        # add empty spin system
        u = np.zeros(u_max, dtype=np.int32)
        self.append(SpinSystem(np.ones(values.shape)*np.nan, i+1, u, empty=True))

        self.__value__ = data

    @property
    def value(self):

        return self.__value__
import gdsvml

class MapService(object):

    def __init__(self, config=None, dataservice=None, verbose=False):

        self.ds     = dataservice
        self.lang   = gdsvml.GDSVML()
        self.config = config
        self.verbose= verbose

#------------------------------------------------------------------------------

    def get_capabilities(self, request):
        return None

#------------------------------------------------------------------------------

    def get_map(self, plot):
        return None

#------------------------------------------------------------------------------

    def register(self, config=None, dataservice=None):

        if config is not None:
            self.config = config

        if dataservice is not None:
            self.ds = dataservice

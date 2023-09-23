try:
    import pint
except ImportError as err:
    print("WARNING: {}".format(err))
else:
    ureg = pint.UnitRegistry()

    # define shortcut for quantity specifier using Q_. e.g. Q_("1 m/s") define 1 meter/second
    Q_ = ureg.Quantity

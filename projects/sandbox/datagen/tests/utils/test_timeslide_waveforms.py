import datagen.utils.timeslide_waveforms as utils


def test_calc_shifts_required():
    # one shift has 30 seconds of livetime
    segments = ((0, 10), (20, 30), (40, 50))

    # test that requiring 0 background time returns 0 shifts
    shifts_required = utils.calc_shifts_required(segments, 0, 1)
    assert shifts_required == 0

    # need an extra shift to get 60 seconds of background
    # due to the chopping off of livetime at the end of each segment
    shifts_required = utils.calc_shifts_required(segments, 60, 1)
    assert shifts_required == 3

from bbhnet.trainer.wrapper import trainify

# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from deepclean.trainer.trainer.train to command line
# execution and parsing

# note that this function is trivial:
# it simply just returns the data paths passed to it.
# however, future projects may have more complicated
# data discovery/generation processes.


@trainify
def main(
    glitch_dataset: str,
    signal_dataset: str,
    val_glitch_dataset: str,
    val_signal_dataset: str,
    hanford_background: str,
    livingston_background: str,
    val_hanford_background: str,
    val_livingston_background: str,
    **kwargs
):

    return (
        glitch_dataset,
        signal_dataset,
        val_glitch_dataset,
        val_signal_dataset,
        hanford_background,
        livingston_background,
        val_hanford_background,
        val_livingston_background,
    )

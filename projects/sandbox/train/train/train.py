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
    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity

    # is this packaging into dict possibly redundant or unnecessary?
    # idea was to simplify arguments to train function

    # package training files into dictionary
    train_files = {
        "glitch dataset": glitch_dataset,
        "signal dataset": signal_dataset,
        "hanford background": hanford_background,
        "livingston background": livingston_background,
    }

    # package validation files into dictionary
    val_files = {
        "glitch dataset": val_glitch_dataset,
        "signal dataset": val_signal_dataset,
        "hanford background": val_hanford_background,
        "livingston background": val_livingston_background,
    }

    return train_files, val_files

from typing import List, Optional


def get_state_flags(ifos: List[str], state_flag: Optional[str] = None):
    if state_flag is None:
        flags = [f"{ifo}_DATA" for ifo in ifos]  # open data flags
    else:
        flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    return flags

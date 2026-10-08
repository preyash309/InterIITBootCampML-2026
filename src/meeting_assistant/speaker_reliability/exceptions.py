"""Errors confined to the optional shadow diarizer."""


class SpeakerReliabilityError(Exception):
    pass


class InvalidReliability(SpeakerReliabilityError):
    pass


class SecondaryUnavailable(SpeakerReliabilityError):
    pass


class SecondaryTimeout(SecondaryUnavailable):
    pass

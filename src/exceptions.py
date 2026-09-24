"""Project-specific errors with messages intended for the user interface."""


class LunarRegError(Exception):
    """Base error for the registration prototype."""


class ImageLoadError(LunarRegError):
    """Raised when an image or its metadata cannot be read."""


class InsufficientMatchesError(LunarRegError):
    """Raised when too few correspondences exist to estimate a transform."""


class RegistrationError(LunarRegError):
    """Raised when geometric estimation or warping fails."""

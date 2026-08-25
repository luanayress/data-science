"""Application-level failures translated to stable HTTP responses."""

class ApplicationError(Exception):
    """Base class for predictable application failures."""

class ModelUnavailableError(ApplicationError):
    pass

class InvalidModelVersionError(ApplicationError):
    pass

class PredictionError(ApplicationError):
    pass

class MonitoringError(ApplicationError):
    pass

from abc import ABC, abstractmethod


class FeatureEngineer(ABC):
    """Defines the base structure of featurer engineer classes.

    Attributes
    ----------
        -

    Methods
    -------
        exten_data()
            abstract method
    """

    @abstractmethod
    def extend_data(self, *args, **kwargs):
        pass

from abc import ABC, abstractmethod
from enum import Enum


def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class PROBLEM_TYPE(Enum):
    CAUSAL_TREATMENT_EFFECT_ESTIMATION = "causal_treatment_effect_estimation"
    PROPENSITY_ESTIMATION = "propensity_estimation"
    SYNTHETIC_DATA_GENERATION = "syntethic_data_generation"


class Propensity_Estimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build(self, params):
        pass

    @abstractmethod
    def fit(self, X, treatment):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Clones scikitlearn style. Check scickitlearn documentation for details.
        """
        pass


class Model(ABC):

    def __init__(self):
        pass

    @staticmethod
    def create_model(name,
                     params,
                     problem_type=PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION,
                     multiple_treatments=False):
        if problem_type == PROBLEM_TYPE.CAUSAL_TREATMENT_EFFECT_ESTIMATION:
            if multiple_treatments:
                raise Exception("Multiple treatments not supported yet")

            # Qui importi direttamente i moduli PyTorch:
            if name == 'bcauss':
                module_path = 'models.BCAUSS'
            elif name == 'dragonnet':
                module_path = 'models.DragonNet'
            elif name == 'ganite':
                module_path = 'models.Ganite'
            elif name == 'bcaus_dr':
                module_path = 'models.BCAUS_DR'
            else:
                raise Exception(f"Model not supported yet: {name}")

            Klass = dynamic_import(module_path)
            net = Klass()
            net.build(params)
            return net

        elif problem_type == PROBLEM_TYPE.PROPENSITY_ESTIMATION:
            if name == 'bcaus':
                module_path = 'models.BCAUS'
            else:
                raise Exception(f"Model not supported yet: {name}")

            Klass = dynamic_import(module_path)
            net = Klass()
            net.build(params)
            return net

        elif problem_type == PROBLEM_TYPE.SYNTHETIC_DATA_GENERATION:
            raise Exception(f"problem_type not supported yet: {problem_type}")
        else:
            raise Exception(f"Invalid problem_type: {problem_type}")

    @abstractmethod
    def build(self, params):
        pass

    @abstractmethod
    def fit(self, X, treatment, y):
        """
        Fits the model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        pass

    @abstractmethod
    def support_ite(self):
        """
        Whether the model supports individual treatment effect ("ITE") estimation

        Returns:
            (Boolean): Whether the model supports ITE estimation
        """
        pass

    @abstractmethod
    def predict_ite(self, X):
        """
        Predicts the individual treatment effect ("ITE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
        Returns:
            (np.array): treatment effect vector
        """
        pass

    @abstractmethod
    def predict_ate(self, X, treatment, y):
        """
        Predicts the average treatment effect ("ATE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        Returns:
            (np.array): treatment effect vector
        """
        pass


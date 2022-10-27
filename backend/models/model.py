import startup

class Model:
    
    def __init__(self):
        pass
    
    def Fit(self, X, y, lr, iters, optimizer, *params):
        """
        Fits the model to given data

        Parameters
        ----------
        X : training points
        y : training observations
        lr : optimiser learning rate
        iters : optimization learning rate

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError
        
    def Predict(self, X_s):
        """
        Produces predictions at points X_s

        Parameters
        ----------
        X_s : Prediction points

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError
        
    def Sample(self, X_s, S):
        """
        Produces S prediction samples on X_s

        Parameters
        ----------
        X_s : Prediction points
        S : Number of samples

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError
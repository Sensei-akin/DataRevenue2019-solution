from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from datetime import date

class DescriptionTransformer(TransformerMixin, BaseEstimator):
    ''' A class to get the length of each description.'''

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def get_length(self,description):
        return len(description)
    
    def transform(self, X):
        # transform X via code or additional methods
        X = X.apply(lambda description : self.get_length(description))
        return pd.DataFrame(X)
    
    
#Custom Transformer that extracts columns passed as argument to its constructor
class TitleTransformer(BaseEstimator, TransformerMixin):
    ''' A class to get the extract age from Title.''' 
    def __init__( self):
        pass
    
    

    def extract_year(self, title):
        
        year_array = [int(letter) for letter in title.split() if letter.isdigit()]
        year = np.nan if not year_array else int(year_array[0])
        return year

    def validate_year(self, year):
        import math
        
        if np.isnan(year):
             return False

        length = math.floor(math.log10(year)) + 1

        if year > date.today().year:
            return False
        elif (length) == 4:
            return True

        else:
            return False
    def calculate_age(self, year): 
        today = date.today() 
        age = today.year - year
        return age

    def create_age(self, title):
        year = self.extract_year(title)
        year = np.nan if not self.validate_year(year) else year
        age = self.calculate_age(year)
        return age

    #Return self nothing else to do here    
    def fit( self, X, y = None ):

        return self

    #Method that describes what we need this transformer to do
    def transform(self, X, y = None):
        X = X.apply(lambda title : self.create_age(title))
        
        return pd.DataFrame(X)

import pytest
from fastai import *
from fastai.collab import *

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.ML_SAMPLE)
    ratings = pd.read_csv(path/'ratings.csv')
    ratings.head()
    series2cat(ratings, 'userId','movieId')
    data = CollabDataBunch.from_df(ratings)
    learn = collab_learner(data, n_factors=50, pct_val=0.2, y_range=(0.,5.))
    learn.fit_one_cycle(3, 5e-3)
    return learn

def test_val_loss(learn): assert learn.validate()[0] < 0.8
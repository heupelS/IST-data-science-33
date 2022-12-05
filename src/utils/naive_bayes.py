import sys, os

from numpy import ndarray
from pandas import DataFrame, read_csv, unique

from matplotlib.pyplot import figure, savefig, show

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

from ds_charts import plot_evaluation_results, bar_chart

sys.path.append( os.path.join(os.path.dirname(__file__)) )
from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False   
###########################################


def NB(df: DataFrame, target_name: str, save_file_name: str):

    print('-- Calculate Naive Bayes')

    y: ndarray = df.pop(target_name).values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)

    labels = unique(y_train)
    labels.sort()

    plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
    savefig(  os.path.join(get_plot_folder_path(), '%s_result' % save_file_name) )

    nb_model_comparison(X_train, y_train, X_test, y_test)
    savefig(  os.path.join(get_plot_folder_path(), '%s_comparison' % save_file_name) )

    if SHOW_PLOTS:
        show()

    
def nb_model_comparison(X_train, y_train, X_test, y_test):

    estimators = {'GaussianNB': GaussianNB(),
                # 'MultinomialNB': MultinomialNB(),
                'BernoulliNB': BernoulliNB()
                #'CategoricalNB': CategoricalNB
                }

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(X_train, y_train)
        prdY = estimators[clf].predict(X_test)
        yvalues.append(accuracy_score(y_test, prdY))

    figure()
    bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)



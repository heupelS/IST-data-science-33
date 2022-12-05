import os, sys

from numpy import ndarray
from pandas import DataFrame, read_csv, unique

from matplotlib.pyplot import figure, savefig, show

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ds_charts import plot_evaluation_results, multiple_line_chart

sys.path.append( os.path.join(os.path.dirname(__file__)) )
from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################


def KNN(df: DataFrame, target_name: str, save_file_name: str, nvalues, dist):

    y: ndarray = df.pop(target_name).values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

    eval_metric = accuracy_score

    values = {}
    best = (0, '')
    last_best = 0
    
    for d in dist:
        y_tst_values = []
        for n in nvalues:
            print('--- Start %s with value %s' % (d, n))

            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            
            knn.fit(X_train, y_train)

            prd_tst_Y = knn.predict(X_test)
            y_tst_values.append(eval_metric(y_test, prd_tst_Y))

            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]

        values[d] = y_tst_values

    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    
    print('--- Best results with %d neighbors and %s'%(best[0], best[1]))
    savefig( os.path.join(get_plot_folder_path(), '%s_results' % save_file_name) )

    print('--- Evaluation of KNN')
    evaluate_knn(X_train, y_train, X_test, y_test, best)
    savefig(  os.path.join(get_plot_folder_path(), '%s_eval' % save_file_name) )

    print('--- Overfitting of KNN')
    evaluate_overfitting(X_train, y_train, X_test, y_test, 'euclidean', nvalues=nvalues)
    savefig(  os.path.join(get_plot_folder_path(), '%s_overfitting' % save_file_name) )

    if SHOW_PLOTS:
        show()


def evaluate_knn(X_train, y_train, X_test, y_test, best):

    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(X_train, y_train)

    prd_trn = clf.predict(X_train)
    prd_tst = clf.predict(X_test)
    
    labels = unique(y_train)
    labels.sort()
    plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)


def evaluate_overfitting(X_train, y_train, X_test, y_test, dist_func: str, nvalues):
    d = dist_func
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(X_train, y_train)
        
        prd_tst_Y = knn.predict(X_test)
        prd_trn_Y = knn.predict(X_train)

        y_tst_values.append(eval_metric(y_test, prd_tst_Y))
        y_trn_values.append(eval_metric(y_train, prd_trn_Y))

    evals = {'Train': y_trn_values, 'Test': y_tst_values}
    figure()
    multiple_line_chart(nvalues, evals, ax = None, title=f'Overfitting KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric), percentage=True)


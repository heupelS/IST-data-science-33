import sys, os

from numpy import ndarray
from pandas import DataFrame, read_csv, unique

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study

from load_data import read_data_by_filename
from general_utils import get_plot_folder_path

from sklearn.model_selection import train_test_split

def neural_network(X, target, file_tag):

    y: ndarray = X.pop(target).values
    trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=0.2, random_state=0)

    labels = unique(trnY)
    labels.sort()

    lr_type = ['constant']
    # lr_type = ['constant', 'invscaling', 'adaptive']
    max_iter = [100]
    # max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
    learning_rate = [.1]
    # learning_rate = [.1, .5, .9]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(lr_type)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(lr_type)):
        d = lr_type[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in max_iter:
                
                print(f'lr type {k} with lr {lr} and max Iter {n}')

                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)

                mlp.fit(trnX, trnY)
                prdY = mlp.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues

        multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                            xlabel='mx iter', ylabel='accuracy', percentage=True)
    
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_mlp_study' ) )

    plt.plot(best_model.loss_curve_)
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_mlp_loss' ) )


    print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    if len(labels) > 2:
        plot_evaluation_results_multi_label(labels, trnY, prd_trn, tstY, prd_tst)
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)

    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_mlp_best' ) )

    lr_type = 'adaptive'
    lr = 0.9
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in max_iter:
        mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
        mlp.fit(trnX, trnY)
        prd_tst_Y = mlp.predict(tstX)
        prd_trn_Y = mlp.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    
    plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr episodes', ylabel=str(eval_metric))
    
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_mlp_overfitting_study' ) )


def main():
    drought_data = read_data_by_filename('drought_dataset_undersampled.csv')
    diabetic_data = read_data_by_filename('diabetic_dataset_oversampled.csv')

    neural_network(drought_data, 'drought', 'drought')
    neural_network(diabetic_data, 'readmitted', 'diabetic')

if __name__ == '__main__':
    main()
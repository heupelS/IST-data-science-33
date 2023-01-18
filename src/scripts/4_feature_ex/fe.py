import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

from load_data import read_data_by_filename
from general_utils import get_plot_folder_path

from pandas import DataFrame

from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap


from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types


from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots

from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import gca, title

THRESHOLD = 0.9

def fs(data, file_tag):
    
    drop, corr_mtx = select_redundant(data.corr(), THRESHOLD)
    print(drop.keys())

    if corr_mtx.empty:
        raise ValueError('Matrix is empty.')

    figure(figsize=[10, 10])
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
    title('Filtered Correlation Analysis')

    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_filtered_correlation_analysis_{THRESHOLD}.png' ) )

    numeric = get_variable_types(data)['Numeric']
    vars_2drop = select_low_variance(data[numeric], 0.1, file_tag)


def select_low_variance(data: DataFrame, threshold: float, file_tag) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')

    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_filtered_variance_analysis.png' ) )

    return lst_variables


def select_redundant(corr_mtx, threshold: float) -> tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx


def fex(data, file_tag):
    
    variables = data.columns.values
    eixo_x = 0
    eixo_y = 4
    eixo_z = 7

    figure()
    xlabel(variables[eixo_y])
    ylabel(variables[eixo_z])
    scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])
    
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_feature_ex_scatter.png' ) )

    mean = (data.mean(axis=0)).tolist()
    centered_data = data - mean
    cov_mtx = centered_data.cov()
    eigvals, eigvecs = eig(cov_mtx)

    pca = PCA()
    pca.fit(centered_data)
    PC = pca.components_
    var = pca.explained_variance_


    # PLOT EXPLAINED VARIANCE RATIO
    fig = figure(figsize=(4, 4))
    title('Explained variance ratio')
    xlabel('PC')
    ylabel('ratio')
    x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
    bwidth = 0.5
    ax = gca()
    ax.set_xticklabels(x_values)
    ax.set_ylim(0.0, 1.0)
    ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
    ax.plot(pca.explained_variance_ratio_)
    for i, v in enumerate(pca.explained_variance_ratio_):
        ax.text(i, v+0.05, f'{v*100:.1f}', ha='center', fontweight='bold')

    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_pca_bar.png' ) )

    transf = pca.transform(data)

    _, axs = subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
    axs[0,0].set_xlabel(variables[eixo_y])
    axs[0,0].set_ylabel(variables[eixo_z])
    axs[0,0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

    axs[0,1].set_xlabel('PC1')
    axs[0,1].set_ylabel('PC2')
    axs[0,1].scatter(transf[:, 0], transf[:, 1])
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_pca_scatter.png' ) )




def main():
    drought_data = read_data_by_filename('drought_dataset_undersampled.csv')
    diabetic_data = read_data_by_filename('diabetic_dataset_oversampled.csv')

    # fs(drought_data,'drought')
    # fs(diabetic_data, 'diabetic')
    
    fex(drought_data,'drought')
    fex(diabetic_data, 'diabetic')

if __name__ == '__main__':
    main()
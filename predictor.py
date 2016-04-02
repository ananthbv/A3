import pandas as pd
import numpy as np
import random
from sklearn import tree
from sklearn import metrics
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sknn.mlp import Classifier, Layer
from matplotlib import pyplot as plt
import sys
import copy
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics import explained_variance_score
from scipy import stats
import scipy as sp
from sklearn import mixture
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.lda import LDA
from sklearn.random_projection import SparseRandomProjection
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score


def run_clustering(X):
    if clustalg == 'em':
        return run_gmm(X_new)
    if clustalg == 'kmeans':
        return run_kmeans(X_new)

def run_gmm(X):
    #acc = []
    best_acc = 0.0
    for n_components in range(2, num_clusters+1):
    #for n_components in [num_clusters]:
        for covariance_type in ['spherical', 'tied', 'diag', 'full']:
            clf = mixture.GMM(n_components = n_components, covariance_type = covariance_type)
            clf.fit(X.tolist())
            y_pred = clf.predict(X)
            #acc = accuracy_score(y_pred.tolist(), list(y.tolist()))
            acc = silhouette_score(X, y_pred)
            #print 'components =', n_components, 'cov type =', covariance_type, 'accuracy score =', acc
            if acc > best_acc:
                best_acc = acc
                best_n_components = n_components
                best_cov_type = covariance_type

    print 'best = ', best_acc, best_n_components, best_cov_type
    return best_acc, best_n_components #, best_cov_type

def run_kmeans(X):
    #acc = 0.0
    best_acc = 0.0
    best_n_clusters = 0
    for n in range(2, num_clusters+1):
    #for n in [num_clusters]:
        clf = KMeans(n_clusters=n, n_init=50)
        clf.fit(X.tolist())
        y_pred = clf.predict(X)
        #print explained_variance_score(y.tolist(), list(clf.labels_), multioutput='uniform_average')  
        #acc.append(accuracy_score(y.tolist(), list(clf.labels_)))
        #acc = metrics.adjusted_rand_score(y.tolist(), list(clf.labels_))
        #acc = accuracy_score(y.tolist(), list(clf.labels_))
        acc = silhouette_score(X, y_pred)
        #adj_mut.append(metrics.adjusted_mutual_info_score(y.tolist(), list(clf.labels_)))
        #exp_var.append()
        #print "k means with", num_clusters, "clusters"
        #print 'predicted labels labels_ = ', (list(clf.labels_))[:40]
        #print 'actual labels = ', (y.tolist())[:40]
        if acc > best_acc:
            best_acc = acc
            best_n_clusters = n

    return best_acc, best_n_clusters

def kurt(x):
    n, min_max, mean, var, skew, kurt = sp.stats.describe(x)
    '''print 'number of points: ', n
    print 'min/max: ', min_max
    print 'mean: ', mean
    print 'variance: ', var
    print 'skew: ', skew
    print 'kurtosis: ', kurt
    print 'median: ', sp.median(x)'''
    
    return kurt


def param_selection_ica(ica, desc):
    # Plot the PCA spectrum
    ica.fit(X)
    
    plt.figure()
    #plt.figure(1, figsize=(4, 3))
    plt.clf()
    #plt.axes([.2, .2, .7, .7])
    #plt.plot(pca.explained_variance_, linewidth=2)
    #plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    n_components = range(1, len(df.columns) + 1)
    print 'n_components list =', n_components
    

    for n in n_components:
        ica = FastICA(n_components=n)
        X_new = ica.fit_transform(X)  # Reconstruct signals
        clf = KMeans(n_clusters=num_clusters)
        clf.fit(X_new.tolist())
        print "k means after X modified with ICA components =", n
        #print 'predicted labels = ', (list(clf.labels_))[:40]
        #print 'actual labels = ', (y.tolist())[:40]
        print 'kurt of labels_ =', kurt(clf.labels_)
        print 'kurt of X =', kurt(X_new)
        
        print "\n\n"


    #print(pca.explained_variance_ratio_) 
    #print X_new
    #print "\n\n\n\n"
    estimator = GridSearchCV(pipe, dict(ica__n_components=n_components))
    estimator.fit(X)

    print estimator.best_estimator_
    print 'best n components =', estimator.best_estimator_.named_steps[desc].n_components

    plt.axvline(estimator.best_estimator_.named_steps[desc].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    
    return estimator

    

def param_selection_pca(pca, desc):
    # Plot the PCA spectrum
    pca.fit(X)

    plt.figure()
    #plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    n_components = range(1, len(df.columns) + 1)
    n_clusters = range(1, 29)
    print 'n_components list =', n_components

    pca.fit(X)
    X_new = pca.transform(X)

    print(pca.explained_variance_ratio_) 
    #print X_new
    #print "\n\n\n\n"
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__n_clusters=n_clusters))
    estimator.fit(X)

    print estimator.best_estimator_
    print 'best n components =', estimator.best_estimator_.named_steps[desc].n_components

    plt.axvline(estimator.best_estimator_.named_steps[desc].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    
    return estimator
    

def print_list(lst, cmt):
    print cmt, '='
    for row in lst:
        print row
        
def get_list_from_df(df, label_column):
    #y = list(df[label_column])
    y = df[label_column]
    df.drop([label_column], 1, inplace=True)
    #X = df.values.tolist()
    X = df.values
    return X, y


filename = sys.argv[1]
clustalg = sys.argv[2]
dralg = sys.argv[3]
label_column = int(sys.argv[4])
delim = ','
if len(sys.argv) > 5:
    delim = sys.argv[5]


df = pd.read_csv(filename, header=None, delimiter=delim)
if (label_column == -1):
    label_column = len(df.columns) - 1

X, y = get_list_from_df(df, label_column)
#print 'first col =', pd.unique(df[0].ravel())
#X = SelectKBest(f_classif, k=3).fit_transform(X, y)
X = preprocessing.MinMaxScaler().fit_transform(X)

num_clusters = len(pd.unique(y.ravel()))
#num_clusters = 10
print 'num clusters = ', num_clusters


if clustalg == 'kmeans':
    best_acc = 0.0
    best_n_clusters = 0
    best_acc, best_n_clusters = run_kmeans(X)
    print 'Best kmeans score =', best_acc, 'with', best_n_clusters, 'clusters' 

if clustalg == 'em':
    best_acc = 0.0
    best_n_components = 0
    best_cov_type = ''
    #best_acc, best_n_components, best_cov_type = run_gmm(X)
    best_acc, best_n_components = run_gmm(X)
    print 'Best kmeans score =', best_acc, 'with', best_n_components, 'clusters' 

'''
params = {'n_clusters': range(1, 31), 
          #'n_init': [50], 
         }

clf = GridSearchCV(KMeans(), param_grid=params)
clf.fit(X.tolist())

#print clf.best_estimator_
#print clf.best_params_
#print
#print 
#print clf.grid_scores_
print "Grid scores on development set:"
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (
        mean_score, scores.std() / 2, params)
print
print clf.scorer_
print
'''
'''


for n in range(1, 29):
    num_clusters = n
    scores = []
    acc, adj_rand, adj_mut = run_kmeans(X, num_clusters)
    print "average k means scores with", num_clusters, "clusters with X unmodified: acc, adj_rand, adj_mut =", acc, adj_rand, adj_mut 

exit()
'''

pca = PCA()
ica = FastICA()
#logistic = KMeans(n_clusters=num_clusters)

if dralg == 'pca':
    ##################################
    ######## KMeans after PCA ########
    ##################################
    for n in range(1, len(df.columns) + 1):
        pca = PCA(n_components=n)
        X_new = pca.fit_transform(X)
        acc, clusters = run_clustering(X_new)
        print "average kmeans score after X modified with PCA", n, "components, clusters =", clusters, "silhouette score =", acc
    #print "\n\n\n\n"

    
if dralg == 'ica':
    ##################################
    ######## KMeans after ICA ########
    ##################################
    for n in range(1, len(df.columns) + 1):
        ica = FastICA(n_components=n)
        X_new = ica.fit_transform(X)  # Reconstruct signals
        acc, clusters = run_clustering(X_new)
        print "average EM score after X modified with ICA", n, "components, clusters =", clusters, "silhouette score =", acc

        
if dralg == 'rp':
    #######################################################
    ######## KMeans after Sparse Random Projection ########
    #######################################################
    for n in range(1, len(df.columns) + 1):
        # create the random projection
        sp = SparseRandomProjection(n_components = n)
        X_new = sp.fit_transform(X)
        acc, clusters = run_clustering(X_new)
        print "average EM score after X modified with Random Projectsion", n, "components, clusters =", clusters, "silhouette score =", acc

        
if dralg == 'lda':
    ##################################
    ######## KMeans after LDA ########
    ##################################
    for n in range(1, len(df.columns) + 1):
        for solver in ['svd', 'eigen']:
        # create the random projection
            lda = LDA(n_components = n, solver = solver)
            X_new = lda.fit_transform(X, y)
            acc, clusters = run_clustering(X_new)
            print "average EM score after X modified with LDA", n, "components, clusters =", clusters, "silhouette score =", acc


plt.show()

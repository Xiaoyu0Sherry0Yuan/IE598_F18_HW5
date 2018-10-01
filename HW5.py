import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',header=None)
df.columns = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline','Class']
df.head()
cols = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline','Class']

##################################EDA
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()
#################CORELATION HEATMAP
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f'
                 ,annot_kws={'size': 8},yticklabels=cols,xticklabels=cols)
plt.show()
###################################SPLIT

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


log_classfier=LogisticRegression(C=0.01,random_state=42)
log_classfier.fit(X_train_std, y_train)
log_predict_test=log_classfier.predict(X_test_std)
log_predict_train=log_classfier.predict(X_train_std)
scores_log_train=accuracy_score(log_predict_train, y_train)
scores_log_test=accuracy_score(log_predict_test, y_test)

print(scores_log_train)
print(scores_log_test)
# ##########################svm

svm = SVC(kernel='linear', C=0.01, random_state=42)
svm.fit(X_train_std, y_train)
svm_pred_test=svm.predict(X_test_std)
svm_pred_train=svm.predict(X_train_std)
scores_svm_train=accuracy_score(svm_pred_train, y_train)
scores_svm_test=accuracy_score(svm_pred_test, y_test)
print(scores_svm_train)
print(scores_svm_test)

##########################pca_lr
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
log_classfier.fit(X_train_pca, y_train)

log_pred_test_pca=log_classfier.predict(X_test_pca)
log_pred_train_pca=log_classfier.predict(X_train_pca)
print(accuracy_score(log_pred_train_pca, y_train))
print(accuracy_score(log_pred_test_pca, y_test))

##########################pca_svm

svm.fit(X_train_pca, y_train)
svm_pred_test_pca=svm.predict(X_test_pca)
svm_pred_train_pca=svm.predict(X_train_pca)
print(accuracy_score(svm_pred_train_pca, y_train))
print(accuracy_score(svm_pred_test_pca, y_test))

##########################lda_lr
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda= lda.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
lr_pred_train=lr.predict(X_train_lda)
lr_pred_test=lr.predict(X_test_lda)
print(accuracy_score(lr_pred_train, y_train))
print(accuracy_score(lr_pred_test, y_test))

##########################lda_svm

svm.fit(X_train_lda, y_train)
svm_pred_test_lda=svm.predict(X_test_lda)
svm_pred_train_lda=svm.predict(X_train_lda)
print(accuracy_score(svm_pred_train_lda, y_train))
print(accuracy_score(svm_pred_test_lda, y_test))
########################################kpca_lr
gamma_space =np.arange(0.01,5,0.05)
acc_lp_kpca_train=np.empty(len(gamma_space))
acc_lp_kpca_test=np.empty(len(gamma_space))

for j,i in enumerate(gamma_space):
    
    scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=i)
    X_train_kpca = scikit_kpca.fit_transform(X_train_std)
    X_test_kpca = scikit_kpca.transform(X_test_std)
    lr_kpca = lr.fit(X_train_kpca, y_train)
    lr_pred_train=lr.predict(X_train_kpca)
    lr_pred_test=lr.predict(X_test_kpca)
    acc_lp_kpca_train[j]=accuracy_score(lr_pred_train, y_train)
    acc_lp_kpca_test[j]=accuracy_score(lr_pred_test, y_test)
    
plt.title('lr accuracy varies according to gamma')
plt.plot(gamma_space, acc_lp_kpca_train,label='training accuracy')
plt.plot(gamma_space, acc_lp_kpca_test,label='testing accuracy')
_=plt.xlabel('gamma')
_=plt.ylabel('accuracy')
plt.show()

print(max(acc_lp_kpca_train))
print(max(acc_lp_kpca_test))
########################################kpca_svm
gamma_space =np.arange(0.01,5,0.05)
acc_svm_kpca_train=np.empty(len(gamma_space))
acc_svm_kpca_test=np.empty(len(gamma_space))
svm = SVC(kernel='linear', C=0.5, random_state=42)
for k,b in enumerate(gamma_space):
   
    kpca = KernelPCA(n_components=2,kernel='rbf', gamma=b)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    svm_kpca = svm.fit(X_train_kpca, y_train)
    svm_pred_train=svm.predict(X_train_kpca)
    svm_pred_test=svm.predict(X_test_kpca)
    acc_svm_kpca_train[k]=accuracy_score(svm_pred_train, y_train)
    acc_svm_kpca_test[k]=accuracy_score(svm_pred_test, y_test)
    
plt.title('svm accuracy varies according to gamma')
plt.plot(gamma_space, acc_svm_kpca_train,label='training accuracy')
plt.plot(gamma_space, acc_svm_kpca_test,label='testing accuracy')
_=plt.xlabel('gamma')
_=plt.ylabel('accuracy')
plt.show()
print(max(acc_svm_kpca_train))
print(max(acc_svm_kpca_test))

print("My name is Xiaoyu Yuan")
print("My NetID is: 664377413")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

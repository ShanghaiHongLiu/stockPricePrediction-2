import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import statistics
import seaborn as sns


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def data_gathering():

    df = pd.read_csv('admission_data.csv')
    pd.set_option('display.max_columns', 20)
    # serial number is redundant in our project, we can use index instead
    del df['Serial No.']

    # need to cast dtype as float to avoid data conversion warning
    df['CGPA'] = df['CGPA'].astype(float)
    df['GRE Score'] = df['GRE Score'].astype(float)
    df['TOEFL Score'] = df['TOEFL Score'].astype(float)
    df['University Rating'] = df['University Rating'].astype(float)
    df['SOP'] = df['SOP'].astype(float)
    df['LOR '] = df['LOR '].astype(float)
    df['CGPA'] = df['CGPA'].astype(float)
    df['Research'] = df['Research'].astype(float)

    return df

# change the chance of admit into 0 or 1
def label_chance_median(df):

    # assuming applicant with change over median will be admitted
    median_value = statistics.median(df['Chance of Admit '])
    binary_admit = []
    for row in df['Chance of Admit ']:
        if row >= median_value:
            binary_admit.append(1)
        else:
            binary_admit.append(0)

    df['binary_admit'] = binary_admit
    return df


'''
split data into training and testing set
enter 0 or 1 for no_features. 
0: automatically get preset features which are 'CGPA', 'GRE Score', 'TOEFL Score'
1: use all features
'''
def plot_heat_map(df):
    # heatmap generation on the correlation of each features
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, cmap='Blues')
    plt.show()

def correlation_table(df):
    # find the correlation between each feature and admission rate
    correlation_chart = pd.DataFrame(df.corr()['Chance of Admit '])
    print(correlation_chart, "\n")

def plot_correlation(df):
    #CGPA
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 2)
    sns.regplot(df['CGPA'], df['Chance of Admit '])
    plt.title('CGPA vs Chance of Admit')
    plt.show()

    #GRE Score
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 2)
    sns.regplot(df['GRE Score'], df['Chance of Admit '])
    plt.title('GRE Score vs Chance of Admit')
    plt.show()

    #TOEFL Score
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,2)
    sns.regplot(df['TOEFL Score'], df['Chance of Admit '])
    plt.title('TOEFL Scores vs Chance of Admit')
    plt.show()
# decide to get all features or top features in dataset
def data_processing(df, no_feautres):
    # check if there are any any null values in data
    df.isnull().values.any()

    admission_chance = df['binary_admit']

    if no_feautres == 0:
        features = df[['CGPA', 'GRE Score', 'TOEFL Score']]

    else:
        features = df.drop(columns = ['Chance of Admit ','binary_admit' ])

    # split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(features, admission_chance , test_size=0.2)

    # data scaling
    model = StandardScaler()
    X_train = model.fit_transform(X_train)
    X_test = model.fit_transform(X_test)

    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=1)
    clf.fit(X_train, y_train)
    pred_result = clf.predict(X_test)
    log_reg_accuracy = clf.score(X_test, y_test) *100
    # print('Logistic Regression accuracy', log_reg_accuracy, "%")
    return log_reg_accuracy

def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0, max_depth=6)
    clf.fit(X_train, y_train)
    pred_result = clf.predict(X_test)
    dec_tree_accuracy = clf.score(X_test, y_test) *100
    # print('Decision Tree accuracy', dec_tree_accuracy, "%")
    return dec_tree_accuracy

def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=150,max_depth=6,random_state=0)
    clf.fit(X_train, y_train)
    pred_result = clf.predict(X_test)
    ran_forest_accuracy = clf.score(X_test,y_test ) *100

    # print('Random Forest accuracy', ran_forest_accuracy, "%")
    return ran_forest_accuracy

def support_vector_machine(X_train, X_test, y_train, y_test):
    # using linear kernel
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    pred_result = clf.predict(X_test)
    svm_accuracy = metrics.accuracy_score(y_test, pred_result, normalize=True)*100
    # print('SVM accuracy', svm_accuracy, "%")
    return svm_accuracy

def gaussian_naive_bayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pred_result = clf.predict(X_test)
    gnb_accuracy = metrics.accuracy_score(y_test, pred_result, normalize = True)*100
    # print('Gaussian Naive Bayes accuracy', gnb_accuracy, "%")
    return gnb_accuracy


# run all 5 model with 3 features in dataset
def three_features_test(df, iteration):
    X_train, X_test, y_train, y_test = data_processing(df,0)

    log_list = []
    decision_list = []
    forest_list = []
    svm_list = []
    gnb_list = []

    # append each iteration's accuracy into list
    for x in range(iteration):
        log_acc = logistic_regression(X_train, X_test, y_train, y_test)
        des_acc = decision_tree(X_train, X_test, y_train, y_test)
        ran_acc = random_forest(X_train, X_test, y_train, y_test)
        svm_acc = support_vector_machine(X_train, X_test, y_train, y_test)
        gnb_acc = gaussian_naive_bayes(X_train, X_test, y_train, y_test)

        log_list.append(log_acc)
        decision_list.append(des_acc)
        forest_list.append(ran_acc)
        svm_list.append(svm_acc)
        gnb_list.append(gnb_acc)

    # result
    print("****************************Start of 3 features test****************************")

    print('\nLogistic Regression accuracy', statistics.mean(log_list), "%")
    print('Decision Tree accuracy', statistics.mean(decision_list), "%")
    print('Random Forest accuracy', statistics.mean(forest_list), "%")
    print('SVM accuracy', statistics.mean(svm_list), "%")
    print('Gaussian Naive Bayes accuracy', statistics.mean(gnb_list), "%")
    print("****************************End of 3 features test****************************")

    all_features = [statistics.mean(log_list), statistics.mean(decision_list), statistics.mean(forest_list),
                    statistics.mean(svm_list), statistics.mean(gnb_list)]

    return all_features

# run all 5 model with all features in dataset
def all_feature_test(df, iteration):
    X_train, X_test, y_train, y_test = data_processing(df,1)

    log_list = []
    decision_list = []
    forest_list = []
    svm_list = []
    gnb_list = []

    for x in range(iteration):
        log_acc = logistic_regression(X_train, X_test, y_train, y_test)
        des_acc = decision_tree(X_train, X_test, y_train, y_test)
        ran_acc = random_forest(X_train, X_test, y_train, y_test)
        svm_acc = support_vector_machine(X_train, X_test, y_train, y_test)
        gnb_acc = gaussian_naive_bayes(X_train, X_test, y_train, y_test)

        # append each iteration's accuracy into list
        log_list.append(log_acc)
        decision_list.append(des_acc)
        forest_list.append(ran_acc)
        svm_list.append(svm_acc)
        gnb_list.append(gnb_acc)

    # result
    print("****************************Start of all features test****************************")

    print('\nLogistic Regression accuracy', statistics.mean(log_list), "%")
    print('Decision Tree accuracy', statistics.mean(decision_list), "%")
    print('Random Forest accuracy', statistics.mean(forest_list), "%")
    print('SVM accuracy', statistics.mean(svm_list), "%")
    print('Gaussian Naive Bayes accuracy', statistics.mean(gnb_list), "%")

    print("****************************End of all features test****************************")

    all_features = [statistics.mean(log_list), statistics.mean(decision_list), statistics.mean(forest_list), statistics.mean(svm_list), statistics.mean(gnb_list)]

    return all_features

# visualize test result through bar chart
def plot_result(three_feature_result, all_feature_result):
    index = np.arange(5)
    bar_width = 0.35

    fig, ax = plt.subplots()
    summer = ax.bar(index, three_feature_result, bar_width,
                    label="3 features")

    winter = ax.bar(index + bar_width, all_feature_result,
                    bar_width, label="all features")

    ax.set_xlabel('Prediction Method')
    ax.set_ylabel('Accuracy')
    ax.set_title('Prediction accuracy by 3 features and all features')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(["Logistic Reg", "Decision Tree", "Random Forest", "SVM ", "GNB"])
    ax.legend()
    plt.show()

# execute the whole program
def main():
    df = data_gathering()
    plot_heat_map(df)
    correlation_table(df)
    plot_correlation(df)
    df = label_chance_median(df)
    three_feature_result = three_features_test(df, 200)
    all_feature_result = all_feature_test(df, 200)
    plot_result(three_feature_result, all_feature_result)


main()

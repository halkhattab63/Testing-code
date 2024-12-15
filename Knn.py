import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Dinamik Analizin$$$$$$$$$$
# Profiling yapmak için cProfile modülü kullanın
# **************************
# cProfile 
import cProfile
# /******************
# line_profiler  
# تقرير عن الزمن المستغرق في كل سطر داخل الدوال.
# كود التشغيل هو 
# kernprof -l -v KNN.py
# ************************
#  استخدام tracemalloc لتتبع تسريبات الذاكرة
import tracemalloc

tracemalloc.start()

# تجاهل التحذيرات المتعلقة بالقياسات غير المحددة
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# تحميل مجموعة البيانات
def load_dataset(path):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, [2, 3]].values  # اختر ميزتين فقط
    y = dataset.iloc[:, 4].values       # اختر العمود الهدف
    return X, y

# معالجة البيانات (تقسيم وتطبيع)
def preprocess_data(X, y, bins=5, test_size=0.25, random_state=0):
    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    y_binned = kbd.fit_transform(y.reshape(-1, 1)).astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=test_size, random_state=random_state, stratify=y_binned)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test, y_binned

# تدريب النموذج
# @profile
def train_knn(X_train, y_train, n_neighbors=5, metric='minkowski', p=2):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)
    classifier.fit(X_train, y_train)
    return classifier

# تقييم النموذج
# لإضافة التحليل باستخدام أداة line_profiler إلى هذه الدالة، ستحتاج إلى تثبيت المكتبة كما ذكرت سابقًا، ثم إضافة زينة (@profile) فوق الدالة. الكود المعدل يكون كالتالي:
# الكود مع @profile لتفعيل line_profiler
# python
# Copy code

# @profile
def evaluate_model(classifier, X_test, y_test, y_binned):
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # حساب ROC و AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_binned))
    n_classes = y_test_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        if np.sum(y_test_bin[:, i]) > 0:  # تحقق من وجود عينات إيجابية
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], (y_pred == i).astype(int))
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = [0], [0], 0  # تجاهل الفئة

    return cm, accuracy, precision, recall, f1, fpr, tpr, roc_auc

# عرض مصفوفة الالتباس
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# رسم منحنى ROC
def plot_roc_curve(fpr, tpr, roc_auc, num_classes):
    plt.figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i in range(num_classes):
        if len(fpr[i]) > 1:  # تجاهل الفئات التي ليس لها قيم ذات معنى
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2, label=f'ROC Curve (Class {i}, AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# عرض نتائج التصنيف
def visualize_results(X, y, classifier, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('blue', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c=ListedColormap(('yellow', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar()
    plt.show()

# البرنامج الرئيسي
def main():
    dataset_path = 'winequality-red.csv'  # تأكد من وضع مسار صحيح للملف
    
    # تحميل البيانات
    X, y = load_dataset(dataset_path)
    
    # معالجة البيانات
    X_train, X_test, y_train, y_test, y_binned = preprocess_data(X, y)
    
    # تدريب النموذج
    classifier = train_knn(X_train, y_train)
    
    # تقييم النموذج
    cm, accuracy, precision, recall, f1, fpr, tpr, roc_auc = evaluate_model(classifier, X_test, y_test, y_binned)
    
    # طباعة النتائج
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # عرض الرسومات
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc, num_classes=len(np.unique(y_binned)))
    visualize_results(X_train, y_train, classifier, 'KNN (Training Set)')
    visualize_results(X_test, y_test, classifier, 'KNN (Test Set)')

# تشغيل البرنامج
if __name__ == "__main__":
    # هون عنا تشغيل الامر  الخاصة الديناميك اناليسس لمكتبة cProfile 
    # بيعمل اناليسيس لكل الكود وبيعطينا كل دالة شقد شتغلت وشقد اخذت وقت 
      cProfile.run("main()")
      
# أخذ لقطة من استهلاك الذاكرة بعد انتهاء البرنامج
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 5 Memory Consumers ]")
for stat in top_stats[:5]:
    print(stat)

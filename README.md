. أدوات تحليل الأداء (Profiling)
cProfile: أداة مدمجة في Python توفر تقارير حول الوظائف التي تستغرق وقتًا أطول.
line_profiler: تحلل الزمن الذي يستغرقه كل سطر من الكود أثناء التنفيذ.
py-spy: أداة لتحليل الأداء توفر رسومات بيانية في الوقت الفعلي.
2. أدوات تحليل استخدام الذاكرة
memory_profiler: تحلل استهلاك الذاكرة لكل سطر من الكود.
tracemalloc: تتبع استخدام الكائنات في الذاكرة وتساعد في اكتشاف التسريبات.
سنقوم بإجراء **تحليل ديناميكي** على الكود الخاص بك باستخدام الأدوات التي ذكرتها بالترتيب: `cProfile`، و`line_profiler`، و`py-spy`، و`memory_profiler`. الخطوات كالتالي:

---

### **1. استخدام `cProfile` لتحليل الأداء العام**

أضف الكود التالي إلى ملف `KNN.py` لتحليل أداء البرنامج بالكامل:

```python
import cProfile

if __name__ == "__main__":
    cProfile.run("main()")  # استدعِ الوظائف الرئيسية هنا
```

ثم شغّل الكود عبر:
```bash
python KNN.py
```

**النتيجة**: 
ستحصل على تقرير مفصل يظهر:
- أسماء الدوال.
- عدد مرات استدعائها.
- الوقت الذي استغرقته كل دالة.

---

### **2. استخدام `line_profiler` لتحليل الأداء سطرًا بسطر**

#### التثبيت:
قم بتثبيت مكتبة `line_profiler`:
```bash
pip install line-profiler
```

#### الخطوات:
1. أضف زينة (`@profile`) على الدوال التي تريد تحليلها:
   ```python
   @profile
   def classifier_training():
       classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
       classifier.fit(X_train, y_train)
       return classifier
   ```

2. شغّل البرنامج باستخدام `kernprof`:
   ```bash
   kernprof -l -v KNN.py
   ```

**النتيجة**:
- تقرير عن الزمن المستغرق في كل سطر داخل الدوال.

---

### **3. استخدام `py-spy` لتحليل الأداء بشكل متزامن**

#### التثبيت:
قم بتثبيت `py-spy`:
```bash
pip install py-spy
```

#### الخطوات:
1. شغّل البرنامج عبر:
   ```bash
   py-spy top -- python KNN.py
   ```

2. إذا كنت تريد تقريرًا تفصيليًا، استخدم:
   ```bash
   py-spy record -o profile.svg -- python KNN.py
   ```

**النتيجة**:
- رسم بياني يظهر تفاصيل استدعاءات الدوال والأداء في الوقت الفعلي.

---

### **4. استخدام `memory_profiler` لتحليل استهلاك الذاكرة**

#### التثبيت:
قم بتثبيت مكتبة `memory_profiler`:
```bash
pip install memory-profiler
```

#### الخطوات:
1. أضف زينة (`@profile`) على الدوال التي تريد تحليلها:
   ```python
   @profile
   def load_dataset():
       dataset = pd.read_csv('winequality-red.csv')
       return dataset
   ```

2. شغّل البرنامج باستخدام:
   ```bash
   python -m memory_profiler KNN.py
   ```

**النتيجة**:
- تقرير يوضح استهلاك الذاكرة لكل سطر داخل الدالة.

---

### **5. استخدام `tracemalloc` لتتبع تسريبات الذاكرة**

#### الخطوات:
1. أضف الكود التالي إلى بداية البرنامج:
   ```python
   import tracemalloc

   tracemalloc.start()
   ```

2. أضف التعليمات التالية عند نقطة النهاية:
   ```python
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')

   print("[ Top 5 Memory Consumers ]")
   for stat in top_stats[:5]:
       print(stat)
   ```

**النتيجة**:
- ستظهر السطور التي استهلكت أكبر قدر من الذاكرة.

---

### **التنفيذ العملي**

يمكنني تعديل الكود لتحليل الأدوات المطلوبة مباشرة. هل ترغب في تطبيق التحليلات بالكامل على الكود الآن؟


dataset == https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download 


pip install kaggle

pip install seaborn

pip install scikit-learn matplotlib seaborn

pip install scikit-learn

pip install numpy

pip install pandas

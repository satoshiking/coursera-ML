<h3>Задание по программированию: Нормализация признаков</h3>
<h4>Вы научитесь:</h4>
- работать с персептроном — простейшим вариантом линейного классификатора
- повышать качество линейной модели путем нормализации признаков

<h4>Введение</h4>
<p>Линейные алгоритмы — распространенный класс моделей, которые отличается своей простотой и скоростью работы. Их можно обучать за разумное время на очень больших объемах данных, и при этом они могут работать с любыми типами признаков — вещественными, категориальными, разреженными. В этом задании мы предлагаем вам воспользоваться персептроном — одним из простейших вариантов линейных моделей.</p>
<p>Как и в случае с метрическими методами, качество линейных алгоритмов зависит от некоторых свойств данных. В частности, признаки должны быть нормализованы, то есть иметь одинаковый масштаб. Если это не так, и масштаб одного признака сильно превосходит масштаб других, то качество может резко упасть.</p>
<p>Один из способов нормализации заключается в стандартизации признаков. Для этого берется набор значений признака на всех объектах, вычисляется их среднее значение и стандартное отклонение. После этого из всех значений признака вычитается среднее, и затем полученная разность делится на стандартное отклонение.</p>

<h4>Реализация в Scikit-Learn</h4>
<p>В библиотеке scikit-learn линейные методы реализованы в пакете sklearn.linear_model. Мы будем работать с реализацией персептрона sklearn.linear_model.Perceptron. Как и у большинства моделей, обучение производится с помощью функции fit, построение прогнозов — с помощью функции predict.</p>

Пример использования:
1. import numpy as np
2. from sklearn.linear_model import Perceptron
3. X = np.array([[1, 2], [3, 4], [5, 6]])
4. y = np.array([0, 1, 0])
5. clf = Perceptron()
6. clf.fit(X, y)
7. predictions = clf.predict(X)

<p>В качестве метрики качества мы будем использовать долю верных ответов (accuracy). Для ее подсчета можно воспользоваться функцией sklearn.metrics.accuracy_score, первым аргументом которой является вектор правильных ответов, а вторым — вектор ответов алгоритма.</p>
<p>Для стандартизации признаков удобно воспользоваться классом sklearn.preprocessing.StandardScaler. Функция fit_transform данного класса находит параметры нормализации (средние и дисперсии каждого признака) по выборке, и сразу же делает нормализацию выборки с использованием этих параметров. Функция transform делает нормализацию на основе уже найденных параметров.</p>

Пример использования:
1. from sklearn.preprocessing import StandardScaler
2. scaler = StandardScaler()
3. X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
4. X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
5. X_train_scaled = scaler.fit_transform(X_train)
6. X_test_scaled = scaler.transform(X_test)
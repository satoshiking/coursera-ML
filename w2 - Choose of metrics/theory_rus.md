<h3>Задание по программированию: Выбор метрики</h3>

Данное задание основано на лекциях по метрическим методам и посвящено выбору наилучшей метрики для конкретной задачи.

<h4>Вы научитесь:</h4>
- выбирать оптимальную метрику из параметрического семейства

<h4>Введение</h4>
Главным параметром любого метрического алгоритма является функция расстояния (или метрика), используемая для измерения сходства между объектами. Можно использовать стандартный вариант (например, евклидову метрику), но гораздо более эффективным вариантом является подбор метрики под конкретную задачу. Один из подходов — использование той же евклидовой метрики, но с весами: каждой координате ставится в соответствие определенный коэффициент; чем он больше, тем выше вклад признака в итоговое расстояние. Веса настраиваются с целью оптимизации качества на отложенной выборке. Другой подход, о котором и пойдет речь в данном задании — выбор метрики из некоторого класса метрик. Мы возьмем за основу метрику Минковского:

Параметром метрики Минковского является число p, которое мы и будем настраивать.

<h4>Реализация в sklearn</h4>
Нам понадобится решать задачу регрессии с помощью метода k ближайших соседей — воспользуйтесь для этого классом sklearn.neighbors.KNeighborsRegressor. Метрика задается с помощью параметра metric, нас будет интересовать значение ’minkowski’. Параметр метрики Минковского задается с помощью параметра p данного класса.
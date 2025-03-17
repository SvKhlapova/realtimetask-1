# Задание  по Spark

Задачи можно выполнять как на RDD, так и на DF API.

## Задача 1
#### Исходные данные

* /data/wiki/en_articles_part - статьи Википедии. Засылать на тестирование нужно на частичном датасете (en_articles_part чтоб не перегружать кластер).

#### Формат данных:
<pre>article ID \<tab> article text</pre>

#### Условие  
Найдите все пары двух последовательных слов (биграмм), где первое слово «narodnaya». Для каждой пары подсчитайте количество вхождений в тексте статей Википедии. Выведите все пары с их частотой вхождений в лексикографическом порядке. Формат вывода - word_pair  count.

#### Пример результата:
<pre>
narodnaya_station 100500
narodnaya_street 42
</pre>
##
Обратите внимание, что два слова в паре объединяются с символом нижнего подчеркивания, а результат - в нижнем регистре. В датасете слово narodnaya может встречаться с большой буквы, поэтому сначала приведите всё к нижнему регистру.

#### Техническая информация
При парсинге отбрасывайте все символы, которые не являются латинскими буквами:
<pre>text = re.sub("^\W+|\W+$", "", text) </pre>




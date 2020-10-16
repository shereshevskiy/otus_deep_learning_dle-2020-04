# Проект:
# Прогнозирование индекса S&P100 на своих предыдущих значениях и курсов входящих в него акций. 
# Сравнение архитектур сетей на базе LSTM и Transformers

- котировки скачены с https://www.alphavantage.co/ и https://finance.yahoo.com/
- построена архитектура нейросети, которая может настраиваться на работу с LSTM или Transformer слоями, 
со свертками (одномерными) или без
- прогнозируется одиночное значение и временной период
- сделаны варианты с дневными рядами и 15-ти минутками

Один из итоговых ноутбуков   
https://nbviewer.jupyter.org/github/shereshevskiy/otus_deep_learning_dle-2020-04/blob/master/project1_SP100_forecast/jupyter_notebooks_15min/03_model_LSTM_and_Transformers_conv_notconv_pytorch_15min.ipynb
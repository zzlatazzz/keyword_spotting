# keyword_spotting

В папках dark_knowledge_distillation, dark_knowledge_distillation_fp16 и dkd_small_attention лежат модели учителя и учеников и соответствующие ноутбуки с обучением и графиками. 

В ноутбуке streaming.py есть визуализация стримминга.

stream.py использует модель kws.pth, и запускается следующим образом: 
```
python stream.py path_to_wav
```
Путь к аудио можно не указывать, тогда будет использоваться audio.wav по умолчанию

Также тут находится отчет со всеми экспериментами и графиками.

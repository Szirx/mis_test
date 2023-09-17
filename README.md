# mis-test

## Результаты работы над заданием:
Работа над заданием велась в jupyter notebook. В нем наглядно, поэтапно пройден весь процесс от загрузки датасета до написания функции инференса, а так же пристуствует возможность дообучения модели и перевод модели в формат ONNX.

Проект контейнезирован в docker.

## Чтобы собрать образ: 
**!Важно:** папки images и predict необходимо очистить от файлов .gitkeep (эти файлы нужны для демонстрации этих папок в репозитории)
```bash
git clone https://github.com/Szirx/mis_test.git
cd mis_test
docker build -t predict .
```
## Чтобы запустить необходимо:
- Чтобы по пути `<path/to/host/image_dir>` на хосте хранилось одно изображение, которое подается в нейронную сеть в контейнере;
- `<path/to/host/predicted_dir>` - по этому пути на хост сохранится маска исходного изображения. 
```bash
docker run -d -v <path/to/host/predicted_dir>:/app/predicted -v <path/to/host/image_dir>:/app/images <docker image-id>
```

 

## [![Generic badge](https://img.shields.io/badge/LANG-EN-blue.svg)](./README.md) Описание проекта FALPRS
Этот проект является заменой [старого](https://github.com/rosteleset/frs). Основные отличия:
* В качестве СУБД используется PostgreSQL.
* Проект использует [userver](https://github.com/userver-framework/userver) — асинхронный фреймворк с открытым исходным кодом.
* Строгое соблюдение типов данных в API запросах. Например, если ожидается числовое поле, то его нельзя заключать в кавычки.
* Добавлена система распознавания автомобильных номеров (license plate recognition system - LPRS).

### Содержание
* [LPRS](#lprs)
  * [Используемые модели нейронных сетей](#lprs_used_dnn)
  * [Общая схема взаимодействия с LPRS](#lprs_scheme)
  * [Автоматический бан номеров](#lprs_auto_ban)
* [FRS](#frs)
  * [Используемые модели нейронных сетей](#frs_used_dnn)
  * [Общая схема взаимодействия с FRS](#frs_scheme)
* [Сборка и настройка проекта](#build_and_setup_falprs)
   * [Системные требования](#system_requirements)
   * [Установка драйверов NVIDIA](#install_drivers)
   * [Установка Docker Engine](#install_de)
   * [Установка NVIDIA Container Toolkit](#install_ct)
   * [Установка PostgreSQL](#install_pg)
   * [Сборка проекта](#build_falprs)
   * [Создание TensorRT планов моделей нейронных сетей](#create_models)
   * [Настройка проекта](#config_falprs)
* [Примеры](#examples)
   * [LPRS](#lprs_examples)
   * [FRS](#frs_examples)
* [Тесты](#tests)
   * [LPRS](#lprs_tests)
   * [FRS](#frs_tests)
* [Импорт данных из старого проекта FRS](#frs_import_data)


<a id="lprs"></a>
## LPRS
Система предназначена для распознавания автомобильных номеров  и обнаружения специального транспорта с мигалками: скорая помощь, МЧС, полиция и т.п. На данный момент поддерживаются российские регистрационные знаки типа 1 (ГОСТ Р 50577-93) и 1А (ГОСТ Р 50577-2018):
![](./docs/ru1.webp)
![](./docs/ru1a.webp)

<a id="lprs_used_dnn"></a>
### Используемые модели нейронных сетей
Сервис работает с четырмя нейронными сетями: VDNet, VCNet, LPDNet и LPRNet. VDNet предназначена для поиска транспортных средств по полученному снимку с видео камеры. VCNet определяет, является ли каждое найденное транспортное средство специальным. LPDNet предназначена для поиска автомобильных номеров. LPRNet предназначена для распознавания номеров из полученных LPDNet данных. Модели VDNet, LPDNet и LPRNet обучены с помощью [Ultralytics](https://github.com/ultralytics/ultralytics). Модель VCNet получена путем "дообучения" (transfer learning with fine-tuning). За основу взята [эта](https://huggingface.co/WinKawaks/vit-small-patch16-224) модель.
Для инференса нейронных сетей используется [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server).

<a id="lprs_scheme"></a>
### Общая схема взаимодействия с LPRS 
 Для взаимодействия с сервисом используется [API](https://rosteleset.github.io/falprs/), подробное описание которого находится в репозитории в файле **docs/openapi.yaml**. Сначала нужно зарегистрировать видео потоки, с которыми предстоит работать системе. Для этого используется API метод **addStream**. Основные параметры:
 - **streamId** - внутренний для вашего бэкенда (внешний для LPRS) идентификатор видео потока;
 - **config** - конфигурационные параметры.
Пример тела запроса:
```json
{
  "streamId": "1234",
  "config": {
    "screenshot-url": "https://my.video.stream/capture",
    "callback-url": "https://my.host/callback?streamId=1"
  }
}
```
Теперь LPRS знает, что захват кадров надо делать с помощью указанного параметра **screenshot-url**, а при обнаружении специального транспорта или распознавании автомобильных номеров отправлять краткую информацию HTTP методом POST на **callback-url**.
Чтобы LPRS стала обрабатывать кадры, нужно вызвать API метод **startWorkflow**.
Пример тела запроса:
```json
{
  "streamId": "1234"
}
```
Система начнёт делать цикличный процесс: получение кадра, его обработка с помощью нейронных сетей, отправление, если требуется, информации на **callback-url**. Пауза на некоторое время и вновь получение кадра, обработка и т.д. Для остановки процесса нужно вызвать API метод **stopWorkflow**.
Пример тела запроса:
```json
{
  "streamId": "1234"
}
```
Чтобы снизить нагрузку на систему, мы рекомендуем вызывать **startWorkflow** и **stopWorkflow**, например, в соответствии с видеоаналитикой камеры: детектор движения, пересечение линий, обнаружение вторжений и т.п. При получении краткой информации о распознанных номерах, ваш бэкенд должен её обработать. Например, cверить полученные номера со списком разрешённых и открыть ворота или просто проигнорировать. События распознавания номеров некоторое время хранятся в LPRS. Если требуется полная информация о событии, то нужно использовать API метод **getEventData**.
Для удаления видео потока используется API метод **removeStream**. С помощью **listStreams** можно получить информацию о зарегистрированных в LPRS видео потоках.

<a id="lprs_auto_ban"></a>
### Автоматический бан номеров
Для предотвращения спама запросов на **callback-url** используется двухэтапный бан номеров. Если система видит номер впервые, то после обработки он окажется на первом этапе бана: на некоторое время (парметр конфигурации **ban-duration**) номер игнорируется вне зависимости от его расположения в кадре. Если система снова увидит этот номер на первой стадии, то время бана продлевается. По окончании первого этапа номер попадёт в следующий. На втором этапе бана (параметр конфигурации **ban-duration-area**) номер игнорируется до тех пор, пока не поменяет свое местоположение в кадре. При смене местоположения (параметр конфигурации **ban-iou-threshold**) номер будет обработан и снова попадёт на первый этап. По окончании второго этапа номер удаляется из бана. Если вы не хотите, чтобы применялся автоматический бан номеров, то необходимо установить значение конфигурационного параметра **ban-duration** или **ban-duration-area** в нулевое значение, например 0s (0 секунд).

<a id="frs"></a>
## FRS
Система предназначена для распознавания лиц.

<a id="frs_used_dnn"></a>
### Используемые модели нейронных сетей
На данный момент FRS использует в работе три модели:
- **scrfd** - предназначена для поиска лиц на изображении. [Ссылка](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) на проект.
- **genet** - предназначена для определения наличия на лице маски или тёмных очков. За основу взят [этот](https://github.com/idstcv/GPU-Efficient-Networks) проект. Модель получена путем "дообучения" (transfer learning with fine-tuning) на трех классах: открытое лицо, лицо в маске, лицо в тёмных очках.
- **arcface** - предназначена для вычисления биометрического шаблона лица. [Ссылка](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) на проект.

<a id="frs_scheme"></a>
### Общая схема взаимодействия с FRS
Для взаимодействия с сервисом используется [API](https://rosteleset.github.io/falprs/), подробное описание которого находится в репозитории в файле **docs/openapi.yaml**. Первым делом, ваш бэкенд с помощью вызова API метода **addStream** регистрирует видео потоки. Основными параметрами метода являются:
- **streamId** - внутренний для бэкенда (внешний для FRS) идентификатор видео потока;
- **url** - это URL для захвата кадра с видео потока. FRS не декодирует видео, а работает с отдельными кадрами (скриншотами). Например, URL может выглядеть как ***http://<имя хоста>/cgi-bin/images_cgi?channel=0&user=admin&pwd=<пароль>***
- **callback** - это URL, который FRS будет использовать, когда распознает зарегистрированное лицо. Например, ***http://<адрес-бэкенда>/face_recognized?stream_id=1***

Для регистрации лиц используется метод **registerFace**. Основные параметры метода:
- **streamId** - идентификатор видео потока;
- **url** - это URL с изображением лица, которое нужно зарегистрировать. Например, ***http://<адрес-бэкенда>/image_to_register.jpg***
В случае успешной регистрации возвращается внутренний для FRS (внешний для бэкенда) уникальный идентификатор зарегистрированного лица - **faceId**.

Для старта и окончания обработки кадров используется метод **motionDetection**. Основная идея заключается в том, чтобы FRS обрабатывала кадры только тогда, когда зафиксировано движение перед видео камерой. Параметры метода:
- **streamId** - идентификатор видео потока;
- **start** - признак начала или окончания движения. Если **start=true**, то FRS начинает через каждую секунду (задаётся параметром *delay-between-frames*) обрабатывать кадр с видео потока. Если **start=false**, то FRS прекращает обработку.

Под обработкой кадра подразумевается следующая цепочка действий:
1. Поиск лиц с помощью нейронной сети *scrfd*.
2. Если лица найдены, то каждое проверяется на "размытость" и "фронтальность".
3. Если лицо не размыто (чёткое изображение) и фронтально, то с помощью нейронной сети *genet* определяется наличие маски и тёмных очков.
4. Для каждого лица без маски и без тёмных очков, с помощью нейронной сети *arcface*, вычисляется биометрический шаблон лица, он же дескриптор. Математически, дескриптор представляет собой 512-ти мерный вектор.
5. Далее каждый такой дескриптор попарно сравнивается с зарегистрированными в системе. Сравнение делается путём вычисления косинуса угла между векторами-дескрипторами: чем ближе значение к единице, тем больше похоже лицо на зарегистрированное. Если самое максимальное значение косинуса больше порогового значения (параметр *tolerance*), то лицо считается распознанным и FRS вызывает callback (событие распознавания лица) и в качестве параметров указывает идентификатор дескриптора (*faceId*) и внутренний для FRS (внешний для бэкенда) идентификатор события (*eventId*). Если в одном кадре окажется несколько распознанных лиц, то callback будет вызван для самого "качественного" ("лучшего") лица (параметр *blur*).
6. Каждое найденное неразмытое, фронтальное, без маски и без тёмных очков, лучшее лицо временно хранится в журнале FRS.

С помощью метода **bestQuality** у FRS можно запрашивать "лучший" кадр из журнала. Например, незнакомый системе человек подошёл к домофону и открыл дверь ключом. Бэкенд знает время открытия ключом (date) и запрашивает лучший кадр у FRS. FRS ищет у себя в журнале кадр с максимальным *blur* из диапазона времени [date - best-quality-interval-before; date + best-quality-interval-after] и выдаёт его в качестве результата. Такой кадр - хороший кандидат для регистрации лица с помощью метода **registerFace**. Как правило, для хорошего распознавания необходимо зарегистрировать несколько лиц одного человека, в том числе с кадров, сделанных в тёмное время суток, когда камера переходит в инфракрасный режим работы.

<a id="build_and_setup_falprs"></a>
## Сборка и настройка проекта
<a id="system_requirements"></a>
### Системные требования
* NVIDIA GPU с параметром Compute Capability большим или равным 6.0 и памятью 4 Гб или больше. Подробности можно посмотреть, например, [здесь](https://developer.nvidia.com/cuda-gpus).

* СУБД PostgreSQL 14 или выше.

Для получения исходного кода нужен git. Если не установлен, то выполнить команду:
```bash
sudo apt-get install -y git
```
Получение исходного кода проекта:
```bash
cd ~
git clone --recurse-submodules https://github.com/rosteleset/falprs.git
```
<a id="install_drivers"></a>
### Установка драйверов NVIDIA
Если в системе уже установлены свежие драйвера NVIDIA, то пропустите этот пункт. Для установки можно использовать скрипт **scripts/setup_nvidia_drivers.sh**. Основные команды взяты [отсюда](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu).
```bash
sudo ~/falprs/scripts/setup_nvidia_drivers.sh
```
После установки необходимо перезагрузить операционную систему:
```bash
sudo reboot
```

<a id="install_de"></a>
### Установка Docker Engine
Если в системе уже установлен Docker Engine, то пропустите этот пункт. Для установки можно использовать  скрипт **scripts/setup_docker.sh**. Основные команды взяты [отсюда](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
```bash
sudo ~/falprs/scripts/setup_docker.sh
```

<a id="install_ct"></a>
### Установка NVIDIA Container Toolkit
Если в системе уже установлен NVIDIA Container Toolkit, то пропустите этот пункт. Для установки можно использовать скрипт **scripts/setup_nvidia_container_toolkit.sh**. Основные команды взяты [отсюда](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```bash
sudo ~/falprs/scripts/setup_nvidia_container_toolkit.sh
```
После установки предлагается перезапустить docker:
```bash
sudo systemctl restart docker
```

<a id="install_pg"></a>
### Установка PostgreSQL
Если PostgreSQL не установлен, то запустите команду:
```.bash
sudo apt-get install -y postgresql
```
Запустите psql:
```bash
sudo -u postgres psql
```
Выполните SQL команды, указав ваш пароль вместо "123":
```sql
drop user if exists falprs;
create user falprs with encrypted password '123';
create database frs;
grant all on database frs to falprs;
alter database frs owner to falprs;
create database lprs;
grant all on database lprs to falprs;
alter database lprs owner to falprs;
\q
```

<a id="build_falprs"></a>
### Сборка проекта
Для сборки проекта можно использовать скрипт **scripts/build_falprs.sh**. Проект собирается с помощью [LLVM](https://llvm.org/). Версия задаётся переменной **LLVM_VERSION**. Мажорная версия PostgreSQL задаётся переменной **PG_VERSION**. Установленную версию PostgreSQL можно узнать командой:
```bash
psql --version
```
Рабочая директория проекта задаётся переменной **FALPRS_WORKDIR** (значение по-умолчанию */opt/falprs*), версия контейнера с Triton Inference Server задаётся переменной **TRITON_VERSION**. 

##### Таблица поддержки Compute Capability и последней версии контейнера
|Compute Capability|Архитектура GPU|Версия контейнера|TensorRT|
|--|--|--|--|
|6.x|Pascal|24.04|8.6.3|
|7.0|Volta|24.09|10.4.0.26|

Если у вас GPU с Compute Capability 7.5 или выше, то вы можете использовать самую последнюю версию [контейнера](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags). Например, сборка для Ubuntu 24.04:
```bash
sudo LLVM_VERSION=18 PG_VERSION=16 TRITON_VERSION=24.09 ~/falprs/scripts/build_falprs.sh
```
Для Ubuntu 22.04:
```bash
sudo LLVM_VERSION=15 PG_VERSION=14 TRITON_VERSION=24.09 ~/falprs/scripts/build_falprs.sh

```

<a id="create_models"></a>
### Создание TensorRT планов моделей нейронных сетей
Для инференса используются TensorRT планы, которые могут быть получены из моделей нейронных сетей в формате ONNX (Open Neural Network Exchange). Для создания планов в рабочей директории можно использовать скрипт **scripts/tensorrt_plans.sh**. Рабочая директория проекта задаётся переменной **FALPRS_WORKDIR** (значение по-умолчанию */opt/falprs*), версия Triton Inference Server задаётся переменной **TRITON_VERSION**:
```bash
sudo TRITON_VERSION=24.09 ~/falprs/scripts/tensorrt_plans.sh
```

<a id="config_falprs"></a>
### Настройка проекта
Для первоначального заполнения баз данных выполните команды, указав значения переменных с префиксом **pg_** (пароль "123" замените на указанный вами при создании пользователя PostgreSQL):
```bash
pg_user=falprs pg_passwd=123 pg_host=localhost pg_port=5432 pg_db=frs ~/falprs/scripts/sql_frs.sh
pg_user=falprs pg_passwd=123 pg_host=localhost pg_port=5432 pg_db=lprs ~/falprs/scripts/sql_lprs.sh
```
Конфигурация проекта находится в файле **/opt/falprs/config.yaml**
У основных параметров есть описание в комментариях. Некоторые значения нужно заменить.
* В секциях *components_manager -> components -> lprs-postgresql-database* и *components_manager -> components -> frs-postgresql-database*  в значениях атрибутов **dbconnection** замените пароль "123" на указанный вами при создании пользователя PostgreSQL , а также другие реквизиты доступа, если они отличаются.
* В секции *components_manager -> task_processors -> main-task-processor* замените значение **worker_threads** на количество ядер CPU вашего сервера. Количество можно посмотреть, например, с помощью команды:
```bash
cat /proc/cpuinfo | grep processor | wc -l
```
* В секции *components_manager -> task_processors -> fs-task-processor* замените значение **worker_threads** на количество CPU вашего сервера, умноженное на 4.

Для запуска контейнера с **Triton Inference Server** выполните команду:
```bash
sudo TRITON_VERSION=24.09 ~/falprs/scripts/triton_service.sh
```
Для создания сервиса **falprs** и ротации логов выполните команду:
```bash
sudo ~/falprs/scripts/falprs_service.sh
```

<a id="examples"></a>
## Примеры
Устанавливаем зависимости:
```bash
sudo apt-get install -y nodejs npm libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev
```

<a id="lprs_examples"></a>
### LPRS
Заходим в директорию с примерами:
```bash
cd ~/falprs/examples/lprs
```
Устанавливаем зависимости:
```bash
npm i
```
Запускаем тестовый сервис для обработки колбэков:
```bash
node lprs_backend.js
```
Добавляем "тестовый" видео поток. В новой консоли:
```bash
cd ~/falprs/examples/lprs
./test_add_stream.sh
```
Проверяем, что поток попал в базу данных:
```bash
./test_list_streams.sh | jq
```
В ответе должен быть json-файл. Ждём минуту, чтобы новые данные попали в кэш FALPRS, выполняем команду:
```bash
./test_workflow.sh
```
В результате выполнения этой команды в консоли с тестовым сервисом *lprs_backend.js* мы должны увидеть строки вида:
```bash
[2024-10-16 12:37:16.108] Callback: {"streamId":"test001","date":"2024-10-16T09:37:16.103891056+00:00","eventId":1,"plates":["O588OA68"],"hasSpecial":false}
[2024-10-16 12:37:16.108] Matched numbers: O588OA68
[2024-10-16 12:37:16.162] Save image to file: /tmp/lprs_backend/screenshots/49b507ac-9b7d-43a6-94f0-18f647fc1f5c.jpg
```
А в директории */tmp/lprs_backend/screenshots* должен появиться файл с изображением:
![](./examples/lprs/test001_draw.jpg)

Удаляем "тестовый" видео поток:
```bash
./test_remove_stream.sh
```
Переходим в консоль, из которой запускали тестовый сервис для обработки колбэков и останавливаем его.

<a id="frs_examples"></a>
### FRS
Заходим в директорию с примерами:
```bash
cd ~/falprs/examples/frs
```
Устанавливаем зависимости:
```bash
npm i
```
Запускаем тестовый сервис для обработки колбэков:
```bash
node frs_backend.js
```
Добавляем "тестовый" видео поток. В новой консоли:
```bash
cd ~/falprs/examples/frs
./test_add_stream.sh
```
Проверяем, что поток попал в базу данных:
```bash
./test_list_streams.sh | jq
```
В ответе должен быть json-файл. Ждём 10 секунд, чтобы новые данные попали в кэш FALPRS, выполняем команду регистрации лица:
```bash
./test_register_face.sh
```
Проверяем, что лицо попало в базу данных:
```bash
./test_list_all_faces.sh | jq
```
В ответе должен быть json-файл. Ждём 10 секунд, чтобы новые данные попали в кэш FALPRS, выполняем команду:
```bash
./test_workflow.sh
```
В результате выполнения этой команды в консоли с тестовым сервисом *frs_backend.js* мы должны увидеть строки вида:
```bash
[2024-10-21 15:34:29.180] Callback: {"faceId":1,"eventId":1}
[2024-10-21 15:34:29.180] This is Albert Einstein
[2024-10-21 15:34:29.220] Save image to file: /tmp/frs_backend/screenshots/492d7722d038434d9e2676648991e65e.jpg
```
А в директории */tmp/frs_backend/screenshots* должен появиться файл с изображением:
![](./examples/frs/einstein_001_draw.jpg)

Удаляем "тестовый" видео поток:
```bash
./test_remove_stream.sh
```
Переходим в консоль, из которой запускали тестовый сервис для обработки колбэков и останавливаем его.


<a id="tests"></a>
## Тесты
Устанавливаем зависимости:
```bash
sudo apt-get install -y python3-requests python3-pytest python3-pytest-order
```
Создаём директории и тестовые базы данных:
```bash
cd ~/falprs/tests
./test_prepare.sh
```

<a id="lprs_tests"></a>
### LPRS
```bash
pytest -v -s test_api_lprs.py
```

<a id="frs_tests"></a>
### FRS
```bash
pytest -v -s test_api_frs.py
```
После окончания тестов удаляем директории и тестовые базы данных:
```bash
./test_clean.sh
```

<a id="frs_import_data"></a>
### Импорт данных из старого проекта FRS
Для копирования данных из старого проекта в новый можно воспользоваться скриптом *import_data.py* из директории *utils* данного репозитория. Перед запуском скрипта убедитесь, что у вас остановлен старый сервис, а новый правильно настроен и тоже остановлен. Устанавливаем зависимости:
```bash
sudo apt-get install -y pip python3-virtualenv
```
Заходим в директорию со скриптом:
```bash
cd ~/falprs/utils
```
Копируем файл:
```bash
cp config.sample.py config.py
```
В файле *config.py* замените значения переменных в соответствии с вашей конфигурцией старого и нового сервиса. Переменные вида *mysql_** и *\*_old* относятся к старому проекту, *pg_** и *\*_new* - к новому. Выполняем команды:
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python import_data.py
```
После работы скрипта должно появиться сообщение об успешном завершении.
Удаляем виртуальную среду:
```bash
deactivate
rm -rf ./__pycache__/
rm -rf venv
```

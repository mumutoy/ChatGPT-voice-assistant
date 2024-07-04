import json
import os
import re
import time
from datetime import datetime
import numpy as np
import asyncio
import edge_tts
import pyaudio
import wave
from playsound import playsound
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from langdetect import detect
from pypinyin import pinyin, Style
from faster_whisper import WhisperModel
from openai import OpenAI
import tempfile
import croniter


# 配置ChatGPT AI参数
API_KEY = "sk-key"
BASE_URL = "https://api.openai.com/v1"
GPT_MODEL = "gpt-3.5-turbo"

# 语音识别
WAKE_WORDS = ["GPT", "小鹏同学", "小盆同学", "小明同学", "chatgpt"]
END_WORDS = ["拜拜", "再见", "bye", "goodbye"]
STOP_WORDS = ["退出"]
# 录音待激活时长（秒）
RECORD_DURATION = 4
# 录音已激活时长（秒）
RECORD_DURATION_LONG = 6

# 音频能量阈值，低于此阈值的音频信号被认为是空内容
AUDIO_ENERGY_THRESHOLD = 50

# 添加你想要过滤的词汇
FILTER_WORDS = ["(.*?字幕提供)", "感谢观看", "谢谢大家", "大家", "(谢谢.*?)", "(.*?字幕志愿者.*?)", "(字幕由.*?提供)", "(.*?支持.*?)", "(.*?订阅.*?)", "(由.*?字幕)", "MING PAO CANADA MING PAO TORONTO","杨茜茜"]

# 语音识别默认语言
LANG = "zh"

# 结束对话的空白输入次数
N = 4

# 回复PROMPT
PROMPT = "你是一个工作和学习中的助理，尽可能简短的又中文回答我的问题，回答的内容中不要有网址这类无用的信息，也不要使用markdown格式，请回答我纯文本格式，除非我特意提醒你我需要。"

# 添加你想要触发POST请求的关键词和URL
SPECIAL_KEYWORDS = [
    {"key": "查(.*?)天气", "url": "https://xxx.com/weather.php", "prompt": "这是一段查询天气的文本，帮我提取一下具体查询那个城市的天气，回答我的时候只回答城市，不要回复其他内容。"},
    {"key": "xxx", "url": "http://xxx.com/xxx"}
]

# 其他设置
SAVE_LOGS = True
CORRECT_SPEECH_ERRORS = True
CORRECT_PROMPT = "你是一位文章改错能手，能够帮人改正文字中可能存在的一些错误文字，可能里面有地名或者其他内容的错误，请帮我修正他们，其他内容则不需要做任何修改（包括标点符号），并直接返回修正之后的内容，不要回答多余的内容。"

WHISPER_MODEL = "large-v3"  # "distil-large-v2"
EDGETTS_VOICE = "zh-CN-XiaoyiNeural"

# 5秒内即将执行的任务
TASK_DELAY = 5

# 设置OpenAI API凭证
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 设置语音识别器
model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8_float16")

# 初始化定时任务调度器
scheduler = BackgroundScheduler()

# 定时任务数据文件
TASK_FILE = "tasks.json"

# 定时任务中音乐存放目录
MUSIC_DIR = "music"
if not os.path.exists(MUSIC_DIR):
    os.makedirs(MUSIC_DIR)

# 加载定时任务数据
def load_tasks():
    if os.path.isfile(TASK_FILE):
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        
        # 获取当前年份
        current_year = datetime.now().year
        
        # 检查并删除已经执行过的任务和明年执行的任务
        tasks = [task for task in tasks if not task.get("executed", False) and not (task.get("time", "").startswith(str(current_year + 1)))]
        
        # 保存更新后的任务列表
        save_tasks(tasks)
        
        return tasks
    return []

# 保存定时任务数据
def save_tasks(tasks):
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=4)

# 添加定时任务
def add_task(task):
    if 'time' not in task:
        print(f"任务 {task} 缺少 'time' 的内容，跳过。")
        return

    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)
    trigger = CronTrigger.from_crontab(task["time"])
    if task["type"] == "url":
        scheduler.add_job(execute_url_task, trigger=trigger, args=[task])
    elif task["type"] == "music":
        scheduler.add_job(execute_music_task, trigger=trigger, args=[task])
    elif task["type"] == "text":
        scheduler.add_job(execute_text_task, trigger=trigger, args=[task])

def execute_text_task(task):
    text = task["data"].strip()
    print("播放本地文本内容", text)
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', prefix='text_', delete=False) as temp_file:
        temp_filename = temp_file.name
        text_response = text_to_speech(text, temp_filename)
    
    play_audio(text_response)
    
    # 延时 500 毫秒
    time.sleep(0.5)
    
    # 删除本地音频
    if os.path.isfile(text_response):
        os.remove(text_response)

    # 删除一次性任务
    if task.get("once", False):
        remove_task(task)

def execute_url_task(task):
    if task["type"] == "music":
        for audio_url in task["data"]:
            file_name = os.path.basename(urlparse(audio_url).path)
            file_path = os.path.join(MUSIC_DIR, file_name)
            
            if os.path.isfile(file_path):
                print(f"本地文件已存在: {file_path}")
                play_audio(file_path)
            else:
                print(f"下载音频文件: {audio_url}")
                response = requests.get(audio_url)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    play_audio(file_path)
                else:
                    print(f"无法下载音频文件: {audio_url}")
    else:
        response = requests.get(task["data"])
        if response.headers.get("Content-Type") == "text/plain":
            print("播放远程文本内容", response.text)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', prefix='url_', delete=False) as temp_file:
                temp_filename = temp_file.name
                url_response = text_to_speech(response.text, temp_filename)
            
            play_audio(url_response)
            
            # 延时 500 毫秒
            time.sleep(0.5)
            
            # 删除本地音频
            if os.path.isfile(url_response):
                os.remove(url_response)

    # 删除一次性任务
    if task.get("once", False):
        remove_task(task)

def execute_music_task(task):
    for file_path in task["data"]:
        file_path = os.path.join(MUSIC_DIR, file_path)
        if os.path.isfile(file_path):
            print(f"本地文件已存在: {file_path}")
            play_audio(file_path)
        else:
            print(f"文件不存在: {file_path}")

    # 删除一次性任务
    if task.get("once", False):
        remove_task(task)

# 删除任务
def remove_task(task):
    tasks = load_tasks()
    tasks = [t for t in tasks if t != task]
    save_tasks(tasks)
    print(f"任务 {task} 已删除。")

# 初始化定时任务
def init_scheduler():
    tasks = load_tasks()
    for task in tasks:
        if 'time' not in task:
            print(f"任务 {task} 没有time项目，直接跳过。")
            continue

        trigger = CronTrigger.from_crontab(task["time"])
        if task["type"] == "url":
            scheduler.add_job(execute_url_task, trigger=trigger, args=[task])
        elif task["type"] == "music":
            scheduler.add_job(execute_music_task, trigger=trigger, args=[task])
        elif task["type"] == "text":
            scheduler.add_job(execute_text_task, trigger=trigger, args=[task])
    scheduler.start()

# 判断json是否满足要求
def is_valid_json_with_crontab(text):
    try:
        # 尝试解析JSON
        data = json.loads(text)
        
        # 检查是否包含必需的键
        if 'time' not in data or 'type' not in data or 'data' not in data:
            return False
        
        # 检查time是否符合crontab格式
        crontab_pattern = r'^(\*|[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|\*\/[0-9]{1,2})( (\*|[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|\*\/[0-9]{1,2})){4}$'
        if not re.match(crontab_pattern, data['time']):
            return False
        
        return True
    except json.JSONDecodeError:
        return False


# 语音添加任务
def add_task_by_voice(text):
    match = re.search(r"请帮我(.*?)闹钟(.*?)", text)
    if match:
        task_description = match.group(1)
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
        # 使用GPT生成crontab时间和需要执行的命令类型
        prompt = f"帮我生成一个crontab时间，和需要执行的命令类型，描述如下：{task_description}，当前时间是：{today}"
        response = optimize_content_with_gpt(prompt, "你是一位助手，能够根据描述生成crontab时间和具体的提醒事项，如输出格式如下：{\"time\":\"59 21 1 7 1\",\"type\":\"text\",\"data\":\"根据提供的描述总结出来的提醒事项\"}")

        print("task json", response)

        # 使用正则表达式提取被 ``` 包围的JSON内容
        json_pattern = r'({[\n\s\S]*?})'
        match = re.search(json_pattern, response)

        if match:
            json_string = match.group(1).strip()
            print("match task json", json_string)
            
            # 检查 json_string 是否为空
            if is_valid_json_with_crontab(json_string):
                task_result = text_to_speech("GPT闹钟解析成功，即将添加定时任务。", "temp/task_json_is_ok.mp3")
                play_audio(task_result)

                try:
                    task = json.loads(json_string)
                    print("task json", task)
                    add_task(task)
                    print("任务已添加。")
                    # 输出语音提示，闹钟添加成功

                    task_result = text_to_speech("闹钟添加成功", "temp/task_alert_result.mp3")
                    play_audio(task_result)
                    # 延时 500 毫秒
                    # time.sleep(0.5)
                    # 删除本地音频
                    # if os.path.isfile(task_result):
                        # os.remove(task_result)

                except json.JSONDecodeError as e:
                    print(f"解析JSON失败: {e}")
            else:
                task_result = text_to_speech("GPT闹钟解析失败，请重试。", "temp/task_json_not_ok.mp3")
                play_audio(task_result)
                print("JSON为空.")
        else:
            print("返回数据中，没有json数据.")

# 识别语音的语言类型
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

async def text_to_speech_async(text, filename):
    try:
        communicate = edge_tts.Communicate(text, EDGETTS_VOICE)
        await communicate.save(filename)
        print(f"音频文件已成功保存到",os.path.basename(filename))
    except Exception as e:
        print(f"生成音频时出错: {e}")

def text_to_speech(text, filename="response.mp3"):
    """
    将文本转换为语音并保存为mp3文件。
    """
    text = text.replace("*", "")
    if not text:
        print("文本内容为空，无法生成音频。")
        return None
    
    if filename == '':
        with tempfile.NamedTemporaryFile(suffix='.mp3', prefix='tts_', delete=False) as temp_file:
            temp_filename = temp_file.name
    else:
        temp_filename = filename
        asyncio.run(text_to_speech_async(text, temp_filename))
        return temp_filename

def play_audio(filename):
    """
    播放指定的音频文件。
    """
    if os.path.isfile(filename):
        try:
            with open(filename, 'rb') as f:
                playsound(f.name)
        except Exception as e:
            print(f"播放音频文件时出错: {filename}")
            print(f"错误详情: {e}")

def record_audio(filename="temp/input.wav", duration=RECORD_DURATION):
    # 录音前删除之前的音频文件
    # if os.path.isfile("temp/response.mp3"):
    #     os.remove("temp/response.mp3")

    """
    录制指定时长的音频并保存为WAV文件。
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = filename

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print("正在录音...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束。")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def is_audio_empty(filename, threshold=AUDIO_ENERGY_THRESHOLD):
    """
    判断音频文件是否为空内容。

    :param filename: 音频文件路径
    :param threshold: 能量阈值，低于此阈值的音频信号被认为是空内容
    :return: True 如果音频文件为空内容，否则 False
    """
    with wave.open(filename, 'rb') as wf:
        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = wf.readframes(nframes)

    audio_data = np.frombuffer(str_data, dtype=np.int16)
    energy = np.sum(np.abs(audio_data)) / nframes
    print("音频能量：", energy)

    return energy < threshold


def speech_to_text():
    """
    监听语音输入并返回文本转录。
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', prefix='input_', delete=False) as temp_file:
        audio_file = temp_file.name

        if wake_status:  # 假设有一个全局变量或函数来表示当前是否处于唤醒状态
            record_audio(audio_file, duration=RECORD_DURATION_LONG)
        else:
            record_audio(audio_file, duration=RECORD_DURATION)

        # audio_file = "temp/input.wav"

        if is_audio_empty(audio_file):
            print("音频文件为空内容")

            # 清空空音频文件
            # if os.path.isfile(audio_file):
            #     os.remove(audio_file)

            return ["", audio_file]

        text = ""
        with open(audio_file, "rb") as f:
            text = f.read()

        detected_language = detect_language(text)

        if detected_language == "zh-cn":
            segments, info = model.transcribe(audio_file, language="zh")
        elif detected_language == "en":
            segments, info = model.transcribe(audio_file, language="en")
        else:
            segments, info = model.transcribe(audio_file, language=LANG)

        text = " ".join([segment.text for segment in segments])

        # 过滤掉不需要的词汇
        patterns = [re.compile(word) for word in FILTER_WORDS]
        for pattern in patterns:
            text = pattern.sub('', text)

        print(f"你说: {text}")
    
        # 延时 500 毫秒
        time.sleep(0.5)

        return [text, audio_file]


def get_response(messages):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
    )
    res_text = response.choices[0].message.content.strip()
    # print(res_text)
    res_text = re.sub(r'[\t\r\n]+', '', res_text)
    print("GPT回复: ", res_text)
    return res_text

def optimize_content_with_gpt(text, prompt):
    """
    使用ChatGPT根据指定的prompt优化内容。
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    response = get_response(messages)
    return response.strip()

def post_to_url(url, data, prompt=None):
    if prompt:
        optimized_text = optimize_content_with_gpt(data["text"], prompt)
        data["text"] = optimized_text
        print("使用 GPT 优化后：", optimized_text)
    
    response = requests.post(url, json=data)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "无法解析响应内容"}

def save_log(text):
    if SAVE_LOGS:
        current_time = datetime.now()
        log_dir = os.path.join("logs", current_time.strftime("%Y-%m-%d"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{current_time.strftime('%H')}.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

def correct_text(text):
    if CORRECT_SPEECH_ERRORS:
        messages = [
            {"role": "system", "content": CORRECT_PROMPT},
            {"role": "user", "content": text}
        ]
        response = get_response(messages)
        print("尝试修复语音识别错误：" + response)
        return response
    return text

def text_to_pinyin(text):
    """
    将文本转换为拼音。
    """
    return ''.join([item[0] for item in pinyin(text, style=Style.NORMAL)])

end = False

def check_upcoming_tasks():
    """
    检查是否有即将执行的定时任务。
    """
    tasks = load_tasks()
    now = datetime.now()
    upcoming_tasks = []

    for task in tasks:
        task_time_str = task.get("time")
        if task_time_str:
            try:
                cron = croniter.croniter(task_time_str, now)
                task_time = cron.get_next(datetime)
                if (task_time - now).total_seconds() <= 60:  # 60秒内即将执行的任务
                    upcoming_tasks.append(task)
            except (croniter.CroniterBadCronError, croniter.CroniterBadDateError):
                print(f"Invalid cron format: {task_time_str}")
                continue

    return upcoming_tasks

# Example usage
# upcoming_tasks = check_upcoming_tasks()
# print(upcoming_tasks)

# 初始化定时任务
init_scheduler()

# 初始状态为非唤醒状态
wake_status = False 

# 退出状态
stop_status = False

# 定义对话循环
while True:
    # 监听语音输入
    print("等待语音输入...")
    speech_result = speech_to_text()  # 使用一个新的变量名来存储返回值
    text = speech_result[0]
    temp_file = speech_result[1]

    # 删除音频文件
    if os.path.isfile(temp_file):
        os.remove(temp_file)

    print("解析音频", text, os.path.basename(temp_file))
    if text is None or text.strip() == "":  # 如果识别到的文本是空白，则不提交给OpenAI
        continue

    text = text.lower()

    # 过滤掉标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 保存日志
    save_log("识别语音：" + text)

    # 修正语音识别错误
    text = correct_text(text)

    # 将文本转换为拼音
    pinyin_text = text_to_pinyin(text)

    # 检查特殊关键词
    matched = False
    for keyword in SPECIAL_KEYWORDS:
        match = re.search(keyword["key"], text)
        if match:
            
            matched_text = match.group(0)
            response = post_to_url(keyword["url"], {"text": matched_text}, keyword.get("prompt"))
            print("检测到特殊关键词！",matched_text)
            response = response.get("response", "无法获取响应")
            # 保存特殊关键词回复
            save_log("特殊关键词回复：" + text)
            matched = True
            break

    if matched:
        print(response)
        if response != '':
            with tempfile.NamedTemporaryFile(suffix='.mp3', prefix='tts_', delete=False) as temp_file:
                temp_filename = temp_file.name
                
            filename = text_to_speech(response, temp_filename)
            play_audio(filename)
            # 播放完毕，清除response
            response = ""

        continue  # 直接返回等待新激活状态

    if any(text_to_pinyin(wake_word) in pinyin_text for wake_word in WAKE_WORDS):
        print("检测到唤醒词！")
        # 设置为唤醒状态
        wake_status = True  
        response = "我能帮您做什么?"

        # 检查是否有即将执行的定时任务
        upcoming_tasks = check_upcoming_tasks()
        if upcoming_tasks:
            for task in upcoming_tasks:
                if task["type"] == "url":
                    execute_url_task(task)
                elif task["type"] == "music":
                    execute_music_task(task)
                elif task["type"] == "text":
                    execute_text_task(task)
            continue  # 跳过唤醒状态，直接等待新激活状态

        # 初始化消息列表
        messages = [
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "assistant",
                "content": response
            }
        ]

        blank_input_count = 0

        # 保持对话直到用户说“stop”
        while True:
            # print(response)
            if response != '':
                if(response == "我能帮您做什么?"):
                    filename = "temp/default_response.mp3"
                    if os.path.isfile(filename) == False:
                        filename = text_to_speech(response, "temp/default_response.mp3")
                        # 将文件改名 default_response.mp3
                        # os.rename(filename, "temp/default_response.mp3")
                    play_audio("temp/default_response.mp3")
                else:
                    with tempfile.NamedTemporaryFile(suffix='.mp3', prefix='tts_', delete=False) as temp_file:
                        temp_filename = temp_file.name
                    filename = text_to_speech(response, temp_filename)
                    play_audio(filename)

                    # 延时0.5秒
                    time.sleep(0.5)
                    if os.path.isfile(filename):
                        os.remove(filename)
               

                # 播放完毕，清除response
                response = ""

            # 监听用户的响应
            speech_result = speech_to_text()  # 使用新的变量名来存储返回值
            user_input = speech_result[0]
            temp_file = speech_result[1]
            print("解析音频", user_input, os.path.basename(temp_file))
            if os.path.isfile(temp_file):
                os.remove(temp_file)

            if user_input is None or user_input.strip() == "":
                blank_input_count += 1
                if blank_input_count >= N:
                    print("未检测到输入。结束对话。")
                    filename = "temp/default_timeout.mp3"
                    if os.path.isfile(filename) == False:
                        filename = text_to_speech("您已长时间未向我提问，我就先退出了。", "temp/default_timeout.mp3")
                        # 将文件改名 default_response.mp3
                        # os.rename(filename, "temp/default_response.mp3")
                    play_audio("temp/default_timeout.mp3")
                    end = True
                    break
                continue

            blank_input_count = 0  # 如果有有效输入，重置空白输入计数

            user_input = user_input.lower()

            # 过滤掉标点符号
            # user_input = re.sub(r'[^\w\s]', '', user_input)

            # 保存日志
            save_log("用户语音识别内容："+user_input)

            # 修正语音识别错误，gpt生成的不需要修正
            # user_input = correct_text(user_input)

            # 如果用户说“stop”或“bye”，退出对话循环
            if any(end_word in user_input for end_word in END_WORDS):
                print("退出本轮对话，再见！")
                filename = "temp/default_bye.mp3"
                if os.path.isfile(filename) == False:
                    filename = text_to_speech("好的，再见！有需要请喊我。", "temp/default_bye.mp3")
                    # 将文件改名 default_response.mp3
                    # os.rename(filename, "temp/default_response.mp3")
                play_audio("temp/default_bye.mp3")

                end = True
                break

            if any(stop_word in user_input for stop_word in STOP_WORDS):
                print("退出本轮对话，再见！")
                filename = "temp/default_over.mp3"
                if os.path.isfile(filename) == False:
                    filename = text_to_speech("好的，我退下了！", "temp/default_over.mp3")
                    # 将文件改名 default_response.mp3
                    # os.rename(filename, "temp/default_response.mp3")
                play_audio("temp/default_over.mp3")
                stop_status = True
                break

            if user_input != "":
                # 将用户输入添加到消息列表
                messages.append({"role": "user", "content": user_input})

                # 使用ChatGPT生成响应
                response = get_response(messages)
                save_log("Chatgpt答复："+response)

                # 将助手响应添加到消息列表
                messages.append({"role": "assistant", "content": response})

        if end:
            end = False  # 重置结束标志以允许再次等待唤醒词
            wake_status = False  # 重置为非唤醒状态
            continue

        if stop_status: 
            break
    else:
        print("未检测到唤醒词。继续监听...")

    # 检查是否需要添加任务
    add_task_by_voice(text)

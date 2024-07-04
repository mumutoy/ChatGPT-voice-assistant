# ChatGPT-voice-assistant
ChatGPT语音助手，支持自定义命令，定时任务（支持播放文字、音频等）

## PC端使用语音助手\[需要有麦克风和音箱\]
1. 首先安装依赖环境
  ```cmd
  pip install -r requirements.txt
  ```
2. 运行代码
  ```cmd
  python ./ai.py
  ```

## 设置项
```
  # 配置ChatGPT AI参数
  API_KEY = "sk-key"
  BASE_URL = "https://api.openai.com/v1"
  GPT_MODEL = "gpt-3.5-turbo"

  # 语音识别
  # 唤醒词
  WAKE_WORDS = \["GPT", "小明同学", "小明小明"\]
  # 退出当前对话
  END_WORDS = \["拜拜", "再见"\]
  # 退出助手（完全退出）
  STOP_WORDS =  \["退出"\]

  # 录音待激活时长（秒）
  RECORD_DURATION = 4
  # 录音已激活时长（秒）
  RECORD_DURATION_LONG = 6
  # 音频能量阈值，低于此阈值的音频信号被认为是空内容
  AUDIO_ENERGY_THRESHOLD = 50
  # 添加你想要过滤的词汇
  FILTER_WORDS = \["(.*?字幕提供)", "感谢观看", "谢谢大家", "大家", "(谢谢.*?)", "(.*?字幕志愿者.*?)", "(字幕由.*?提供)", "(.*?支持.*?)", "(.*?订阅.*?)", "(由.*?字幕)", "MING PAO CANADA MING PAO TORONTO","杨茜茜"\]
  
  # 语音识别默认语言
  LANG = "zh"
  
  # 结束对话的空白输入次数，连续几次空白内容，自动结束当前对话记录
  N = 4
  
  # 回复PROMPT
  PROMPT = "你是一个工作和学习中的助理，尽可能简短的又中文回答我的问题，回答的内容中不要有网址这类无用的信息，也不要使用markdown格式，请回答我纯文本格式，除非我特意提醒你我需要。"
  
  # 添加你想要触发POST请求的关键词、prompt提示词（帮助提取指定格式的内容，用来提交给指定URL）以及提交数据的URL
  SPECIAL_KEYWORDS = [
      {"key": "查(.*?)天气", "url": "https://xxx.com/weather.php", "prompt": "这是一段查询天气的文本，帮我提取一下具体查询那个城市的天气，回答我的时候只回答城市，不要回复其他内容。"},
      {"key": "xxx", "url": "http://xxx.com/xxx"}
  ]
  
  # 其他设置
  SAVE_LOGS = True
  CORRECT_SPEECH_ERRORS = True
  CORRECT_PROMPT = "你是一位文章改错能手，能够帮人改正文字中可能存在的一些错误文字，可能里面有地名或者其他内容的错误，请帮我修正他们，其他内容则不需要做任何修改（包括标点符号），并直接返回修正之后的内容，不要回答多余的内容。"
  # whisper 模型
  WHISPER_MODEL = "large-v3" 
  # edge-tts发声人，获取发声人列表，可输入: `edge-tts -l`
  EDGETTS_VOICE = "zh-CN-XiaoyiNeural"
  
  # 5秒内即将执行的任务，对话当中，如果定时任务还有指定时间，自动退出当前对话，优先执行定时任务
  TASK_DELAY = 5
```
## 定时任务设置
  ```json
  [
      {
          "time": "26 12 * * *",
          "type": "url",
          "data": "https://xxx.com/a.txt"
      },
      {
          "time": "18 13 * * *",
          "type": "text",
          "data": "现在10点15分了，记得去上班哟。"
      },
      {
          "time": "19 13 * * *",
          "type": "music",
          "data": [
              "ok.mp3"
          ]
      },
      {
          "time": "2 22 2 7 *",
          "type": "text",
          "data": "五分钟后提醒"
      }
  ]
  ```
  1. type = url时，支持读取远程网页内容，如果content-type是"text/plain"时，默认认为是文本内容，直接使用tts生成语音内容播放，如果为其他类型认为是mp3格式，自动下载并播放。
  2. type=text时，自动调用tts生成音频
  3. type=music时，支持播放music下的文件，或者为URL，自动下载并播放
  4. 支持使用命令`请帮我(.*?)闹钟(.*?)`激活定时任务，可自定义prompt，因为ChatGPT有时候生成的格式有误，可以多试试。

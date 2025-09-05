import numpy as np
import sounddevice as sd
from gradio_client import Client, handle_file
from scipy.io import wavfile
import os



class CosyVoiceV2:
	"""
	CosyVoiceV2 的类，用来进行声音的生成
	"""
	def __init__(self):
		self.client = Client("http://127.0.0.1:7860/")
		self.wav_space_file = handle_file('./core/test_test_files_audio_sample.wav')
		self.wav_data = (np.array([], dtype=np.float32), 44100)
		self.wav_None = (np.array([], dtype=np.float32), 44100)
		self.dict_info = {}
		with open(r'./audio_file_samples/prompt.txt', 'r', encoding='utf-8') as fp:
			self.dict_info = eval(fp.read())

	def get_wav(self):
		return self.wav_data

	def clear_wav(self):
		self.wav_data = self.wav_None

	def get_wav_text(self, wav_file_path:str):
		"""获取当前音频的文本"""
		wav_name = wav_file_path.split('/')[-1]
		return self.dict_info[wav_name]

	def save_wav(self, file_name:str):
		"""保存音频为文件"""
		pre_path = './save_wav'
		if not os.path.exists(pre_path):
			os.mkdir(pre_path)
		wavfile.write(pre_path+'/'+file_name[:200]+'.wav', self.wav_data[1], self.wav_data[0])

	def generate_audio_by_model(self, text:str, model_name:str, stream:str='False'):
		"""
		使用模型进行音频生成
		:param text: 文本
		:param model_name: 模型名称
		:param stream: 是否流式生成 (作用不详建议关闭)
		"""
		result = self.client.predict(
			tts_text=text,
			mode_checkbox_group="预训练音色",
			sft_dropdown=model_name,
			prompt_text="",
			prompt_wav_upload=handle_file('./core/test_test_files_audio_sample.wav'),
			prompt_wav_record=handle_file('./core/test_test_files_audio_sample.wav'),
			instruct_text="",
			seed=0,
			stream=stream,
			speed=1,
			api_name="/generate_audio",
		)
		self.wav_data_process(result)

	def generate_audio_by_file(self, text, prompt_wav_upload:str, prompt_text:str,
							   instruct_text:str, stream:str='False'):
		"""
		使用文件进行音频生成
		:param text: 文本
		:param prompt_wav_upload: 参考音频文件路径
		:param prompt_text: 音频中对应的文字
		:param instruct_text: 阅读的指令
		:param stream: 是否流式生成 (作用不详建议关闭)
		"""
		result = self.client.predict(
			tts_text=text,
			mode_checkbox_group="自然语言控制",
			sft_dropdown="",
			prompt_text=prompt_text,
			prompt_wav_upload=handle_file(prompt_wav_upload),
			prompt_wav_record=handle_file('./core/test_test_files_audio_sample.wav'),
			instruct_text=instruct_text,
			seed=0,
			stream=stream,
			speed=1,
			api_name="/generate_audio",
		)
		self.wav_data_process(result)

	def wav_data_process(self, result):
		temp_rate, temp_data = wavfile.read(result)
		if temp_data.ndim > 1:
			temp_data = temp_data.mean(axis=1)  # 立体声转单声道
		if temp_data.dtype == np.int16:
			temp_data = temp_data.astype(np.float32) / 32768.0  # 16-bit转[-1,1]
		self.wav_data = (temp_data, temp_rate)



class VoicePlayer:
	"""音频播放器"""
	def __init__(self):
		self.end_index = None
		self.data = None
		self.rate = None
		self.stream = None
		self.chunk_size = 1024
		self.frame_index = 0
		self.total_frames = 0

	def reset(self, wav):
		"""重置音频数据"""
		assert not isinstance(wav, str), '错误,wav 不应该为 str 类型数据'
		self.data, self.rate = wav[0], wav[1]
		# 关闭旧流（如果存在）
		if self.stream is not None:
			self.stream.stop()
			self.stream.close()
		# 创建并启动新流
		self.stream = sd.OutputStream(
			samplerate=int(wav[1]),  # 确保采样率为整数
			channels=1,
			dtype='float32'
		)
		self.stream.start()
		self.frame_index = 0
		self.total_frames = len(wav[0])

	def play_step(self):
		"""逐块播放音频"""
		if self.frame_index < self.total_frames:
			end_index = min(self.frame_index + self.chunk_size, self.total_frames)
			chunk = self.data[self.frame_index:end_index]
			self.stream.write(chunk)
			self.frame_index = end_index
			return np.sqrt(np.mean(np.square(chunk)))
		else:
			self.stream.stop()
			self.stream.close()
			return None


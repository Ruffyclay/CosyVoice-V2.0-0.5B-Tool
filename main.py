"""
语音 TTS 测试
"""
import time
from core.class_TTS import *

# 生成单条语句之后是否播放?
bool_play_voice = True

CosyVoice = CosyVoiceV2()
player = VoicePlayer()

with open('test_novel/config.txt', 'r', encoding='utf8') as fp:
    Dic_Novel_Config = eval(fp.read())

with open('test_novel/第一章.txt', 'r', encoding='utf8') as fp:
    list_temp_content = fp.read().split('\n\n')
    list_content = []
    for content in list_temp_content:
        list_content.append(content.replace('\n', ''))



if __name__ == '__main__':
    num_index = 0
    for content in list_content:
        role = content.split(':')[0]
        text = content.replace('instruct_text', '').split(':')[1]
        instruct_text = "专业配音"
        if 'instruct_text:' in content:
            content, instruct_text = content.split('instruct_text:')
        wav_file = Dic_Novel_Config[role]
        num_index += 1
        print(role, text, '要求:'+instruct_text)
        if 'generate_mode' in wav_file:
            CosyVoice.generate_audio_by_model(text, wav_file['generate_mode'])
        elif 'prompt_wav_upload' in wav_file:
            wav_file = wav_file['prompt_wav_upload']
            prompt_text = CosyVoice.get_wav_text(wav_file)
            CosyVoice.generate_audio_by_file(text, wav_file, prompt_text, instruct_text)
        else:
            assert 0 == 1, '调用的生成模式错误，请检查是否被定义，或书写错误!!!'

        player.reset(CosyVoice.get_wav())
        while True:
            current_rms = player.play_step() if bool_play_voice else None
            if current_rms is None:
                CosyVoice.save_wav(f'{str(num_index) + ". " + text}')
                CosyVoice.clear_wav()
                break
            # print(f"当前振幅：{current_rms:.2f}")
            time.sleep(0.015)

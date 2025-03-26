if __name__ == '__main__':

    from melo.api import TTS
    device = 'auto'

    # TTS model for MS language
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='model.pth')
    config_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='config.json')

    models = {
        'EN': TTS(language='EN', device=device),
        'ES': TTS(language='ES', device=device),
        'FR': TTS(language='FR', device=device),
        'ZH': TTS(language='ZH', device=device),
        'JP': TTS(language='JP', device=device),
        'KR': TTS(language='KR', device=device),
        'MS': TTS(language='MS', device=device, config_path=config_path, ckpt_path=ckpt_path),
    }

    # 1st execution for ZH to build prefix dict to reduce loading time -> https://github.com/myshell-ai/MeloTTS/issues/130
    models['ZH'].tts_to_file('text-to-speech 领域近年来发展迅速', speaker_id=models['ZH'].hps.data.spk2id['ZH'], split=True)
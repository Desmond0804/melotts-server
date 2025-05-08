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

    sentences = {
        'EN': "Did you ever hear a folk tale about a giant turtle?",
        'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
        'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
        'ZH': "text-to-speech 领域近年来发展迅速",
        'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
        'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
        'MS': "Perubahan iklim melibatkan perubahan signifikan dalam corak cuaca.",
    }

    voices = {
        'EN': "EN-Default",
        'ES': "ES",
        'FR': "FR",
        'ZH': "ZH",
        'JP': "JP",
        'KR': "KR",
        'MS': "shafiqah-idayu-chatbot",
    }

    # 1st execution for all model to preload them
    for key in models.keys():
         models[key].tts_to_file(sentences[key], speaker_id=models[key].hps.data.spk2id[voices[key]], split=True)
    
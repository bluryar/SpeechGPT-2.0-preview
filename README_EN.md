# SpeechGPT 2.0-preview

<div align="center" style="line-height: 1;">
    <a href="https://open-moss.com/cn/speechgpt2-preview/" target="_blank" style="margin: 2px;">
        <img alt="Project Page" src="https://img.shields.io/badge/🏠%20Project%20Page-SpeechGPT-536af5?color=e31a2f&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://sp2.open-moss.com/" target="_blank" style="margin: 2px;">
        <img alt="Chat" src="https://img.shields.io/badge/🤖%20Demo-SpeechGPT-536af5?color=1ae3f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://discord.com/channels/1333092992348721274/1333092994135621807" target="_blank" style="margin: 2px;">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-SpeechGPT-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://huggingface.co/fnlp" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpeechGPT-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://x.com/Open_MOSS" target="_blank" style="margin: 2px;">
    <img alt="X Follow" src="https://img.shields.io/badge/Twitter-OpenMOSS-black?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>
<p align="center">
    <img src="asset/logo.png" width="20%"> <br>
</p>

## Introduction
SpeechGPT 2.0-preview is our first human-like real-time interaction system as we move towards context intelligence. Trained on millions of hours of speech data, this end-to-end  spoken language model features human-like spoken expressions and low-latency responses at the millisecond level, enabling natural, fluid real-time interruption interactions. SpeechGPT 2.0-preview aligns speech and text modalities well. On one hand, it demonstrates significant speech style generalization capabilities, following user commands to achieve multi-emotion, multi-style, and multi-tone control with intelligent switching. It also has strong role-playing abilities, simulating various characters' tones and emotional states. Additionally, it showcases a range of vocal talents, including poetry recitation, storytelling, and speaking in dialects. On the other hand, while it excels in vocal expressiveness, it also has impressive intelligence and text capabilities, enabling it to support tool calls, online searches, and external knowledge base access. Currently, SpeechGPT 2.0-preview has been trained only on Chinese speech data and has not been trained on English voice data. As a result, the model does not yet support English conversation.

## Model Summary
SpeechGPT 2.0-preview is an end-to-end spoken dialogue language model. Building on our insights and technological advancements in the field of end-to-end speech dialogue, we have developed an ultra-low bitrate streaming speech Codec that jointly models semantics and acoustics. We have constructed an efficient speech data crawling system, a multifunctional and high-efficiency speech data cleaning pipeline, and a comprehensive multi-granularity speech data annotation system, accumulating millions of hours of real speech data with meticulous annotation. We have developed a conversational dialogue speech synthesis system with strong voice cloning ability, synthesizing hundreds of thousands of hours of multi-role, multi-style speech-to-speech dialogue data based on this. We have proposed a new speech-text mixed-modeling architecture and a multi-stage training process for mixed speech-text modeling to balance textual and speech capabilities, preventing the model from compromising its intelligence while learning speech capabilities, and enabling it to seamlessly replace text models in various frameworks, thus supporting functions such as tool invocation, internet search, and external knowledge base integration. By modeling speech dialogue in an end-to-end manner, SpeechGPT 2.0-preview has achieved a latency of less than 200ms in practical tests, providing users with a smooth real-time interactive experience.

Through the experimental process, we also observed many interesting phenomena and conclusions. For example, through extensive pre-training on speech-text alignment, we found that the model could "emerge" with the ability to generalize speech styles. This includes controlling speech rate even without training on dialogue data with explicit speech rate adjustments, and adopting tones and styles of characters that the model had never seen before. Moreover, the quality of the speech data synthesis engine is key to enhancing the capabilities of the end-to-end speech model across various training stages.


### Semantic-Acoustic Joint Modeling Ultra-Low Bitrate Streaming Speech Codec
<p align="center">
    <img src="asset/codec_arch.jpg" width="100%"> <br>
</p>

- 24kHz speech input
- Semantic-acoustic joint modeling
- Ultra-low bitrate: 750bps (75 tokens per second)
- Supports streaming input/output



### Codec Patchify Based Speech-Text LLM Architecture
<p align="center">
    <img src="asset/sg2_arch.jpg" width="100%"> <br>
</p>

- Codec Patchify
  - We effectively reduce the modality gap between speech and text sequences through Codec Patchify, alleviating conflicts in cross-modal modeling. Specifically, this method aggregates adjacent T time-step RVQ codec tokens into a single patch, which is then mapped to a unified vector representation by the patch projector module. This representation is subsequently input into the speech-text LLM for joint modeling. In terms of model architecture design, the hidden states of the LLM are used for two decoding tasks simultaneously: one decodes to generate text output, while the other is input into the patch decoder for speech generation. Notably, we designed an autoregressive language model with multiple LM heads as the patch decoder. This module decodes in an autoregressive manner, generating multiple RVQ codec tokens per time step, thus producing speech output.
- Speech-Text Joint Modeling, Flexible Cross-Combination
  - The speech-text LLM simultaneously takes in and outputs both speech and text representations.


## Model Downloads
|         Model         |       Type       |                                                                     URL                                                                     |
|:---------------------:|:----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|
| SpeechGPT 2.0-preview Codec | Codec | [🤗 Huggingface](https://huggingface.co/fnlp/SpeechGPT-2.0-preview-Codec)|
|    SpeechGPT 2.0-preview-7B     |    LLM    |        [🤗 Huggingface](https://huggingface.co/fnlp/SpeechGPT-2.0-preview-7B)|




## How to Run Locally
### Clone Repo
```sh
git clone https://github.com/OpenMOSS/SpeechGPT-2.0-preview.git
cd SpeechGPT-2.0-preview
```

### Download Model Weights

```shell
# Should install git-lfs
git lfs install
git clone https://huggingface.co/fnlp/SpeechGPT-2.0-preview-Codec
git clone https://huggingface.co/fnlp/SpeechGPT-2.0-preview-7B
```

### Prepare Environments
```sh
pip3 install -r requirements.txt
pip3 install flash-attn==2.7.3 --no-build-isolation
```

### Launch Gradio Demo

```sh
python3 demo_gradio.py --codec_ckpt_path SpeechGPT-2.0-preview-Codec/sg2_codec_ckpt.pkl --model_path SpeechGPT-2.0-preview-7B/
```

<p align="center">
    <img src="asset/gradio.jpeg" width="70%"> <br>
</p>

## Model Ability
### Real-time Spoken Conversational Ability

### Rich in emotion, diverse in style

### Role-playing

### Vocal Talent


## License
- This code repository is licensed under the [Apache 2.0](LICENSE).


## Acknowledgements
- [Qwen](https://github.com/QwenLM/Qwen2.5): We use Qwen2.5-7B-Instruct as our base model.
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer/tree/main/speechtokenizer)


## Citation
```
@misc{sg2preview,
  author = {Open-Moss},
  title = {SpeechGPT 2.0-preview},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenMOSS/SpeechGPT-2.0-preview}},
}
```

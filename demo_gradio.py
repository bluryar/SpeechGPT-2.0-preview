import io
import argparse
import sys
import os
import json
import logging
import yaml
import random
import re
import asyncio
import gradio as gr
import torch
import soundfile as sf
import numpy as np
import torchaudio

# 移除环境变量设置，允许使用多GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Any, List, Union, Tuple
from transformers import HfArgumentParser
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    HfArgumentParser,
)
import torch.nn.functional as F
import torch.distributed as dist

from mimo_qwen2_grouped import *
from Codec.models.codec import Generator as SpeechGPT2Tokenizer


class MIMOStopper(StoppingCriteria):
    def __init__(
        self, stop_id: int, group_size: int, audio_channels: int, max_tokens: int
    ) -> None:
        super().__init__()
        self.stop_id = stop_id
        self.group_size = group_size
        self.audio_channels = audio_channels
        self.max_tokens = max_tokens

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:
        # Stop when last token of channel 0 is the stop token
        return (
            input_ids[0, -((self.audio_channels + 1) * self.group_size)].item()
            == self.stop_id
        ) or input_ids.numel() // self.group_size // (
            self.audio_channels + 1
        ) >= self.max_tokens


class InputSegment:
    def __init__(
        self,
        text: str = None,
        audio: torch.Tensor = None,
        tokenized_text: torch.Tensor = None,
        zeroemb_idx: int = 1024,  # TODO: Make this a parameter
        add_sosp_eosp=True,
        add_zeroemb_loss=False,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text channel cannot be empty"
        assert not (
            has_text and has_tokenized_text
        ), "Can't both have text and tokenized text"
        if has_tokenized_text:
            assert tokenized_text.shape[0] <= audio.reshape(-1, 3).shape[0]
        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.zeroemb_idx = zeroemb_idx
        self.add_sosp_eosp = add_sosp_eosp

    @staticmethod
    def insert_between(tensor, i, value=-1):
        device = tensor.device
        dtype = tensor.dtype
        return torch.scatter(
            torch.full(
                (1, tensor.shape[1] + (tensor.shape[1] - 1) * i + i),
                value,
                dtype=dtype,
                device=device
            ),
            1,
            torch.arange(0, tensor.shape[1], dtype=torch.int64, device=device)[None] * (i + 1),
            tensor,
        )

    def to_input_id(
        self,
        tokenizer,
        group_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 确保device是一个正确的torch.device对象
        if isinstance(device, str):
            device = torch.device(device)
        
        if self.tokenized_text is None:
            tokenized_text = tokenizer(
                self.text,
                return_tensors="pt",
                truncation=True,
                max_length=999999,
                padding=False,
                add_special_tokens=False,
            )[
                "input_ids"
            ].int().to(device)  # [1, seqlen]
        else:
            tokenized_text = self.tokenized_text.unsqueeze(0)
            if tokenized_text.device != device:
                tokenized_text = tokenized_text.to(device)

        if self.audio is None:  # Pure text block
            # Add group_size - 1 tokens between every two text tokens
            if group_size > 1:
                tokenized_text = self.insert_between(
                    tokenized_text, group_size - 1, value=-100
                )
            audio_part_input_id = torch.full(
                (3, tokenized_text.shape[1]), self.zeroemb_idx, dtype=torch.int, device=device
            )
        else:  # Audio + text block
            sosp_token = (
                tokenizer.convert_tokens_to_ids("<|sosp|>")
                if self.add_sosp_eosp
                else None
            )
            eosp_token = (
                tokenizer.convert_tokens_to_ids("<|eosp|>")
                if self.add_sosp_eosp
                else None
            )
            audio_part = self.audio.reshape(-1, 3).T  # [3, seqlen]
            if audio_part.device != device:
                audio_part = audio_part.to(device)
                
            assert (
                audio_part.shape[1] % group_size == 0
            ), f"Audio shape {audio_part.shape} is not divisible by group_size {group_size}"

            if tokenized_text.shape[1] * group_size > audio_part.shape[1]:
                print(
                    f"Expected text to be shorter than or equal to audio, but got text {tokenized_text.shape} * group_size and audio {audio_part.shape}"
                )
                tokenized_text = tokenized_text[:, : audio_part.shape[1] // group_size]
                print(f"Truncated text to {tokenized_text.shape} * group_size")
                print(f"The offending text is: {self.text}")

            if tokenized_text.shape[1] * group_size < audio_part.shape[1]:
                tokenized_text = F.pad(
                    tokenized_text,
                    (0, audio_part.shape[1] // group_size - tokenized_text.shape[1]),
                    value=tokenizer.convert_tokens_to_ids("<|empty|>"),
                ).int()
            tokenized_text = (
                torch.cat(
                    [
                        torch.tensor([[sosp_token]], dtype=torch.int, device=device),
                        tokenized_text,
                        torch.tensor([[eosp_token]], dtype=torch.int, device=device),
                    ],
                    dim=1,
                )
                if self.add_sosp_eosp
                else tokenized_text
            )
            tokenized_text = self.insert_between(
                tokenized_text, group_size - 1, value=-100
            )
            audio_part_input_id = (
                torch.cat(
                    [
                        torch.full((3, group_size), self.zeroemb_idx, dtype=torch.int, device=device),
                        audio_part,
                        torch.full((3, group_size), self.zeroemb_idx, dtype=torch.int, device=device),
                    ],
                    dim=1,
                )
                if self.add_sosp_eosp
                else audio_part
            )

        input_ids = torch.cat(
            [tokenized_text, audio_part_input_id], dim=0
        )  # [4, seqlen]
        
        # 最后再次确保所有数据在正确设备上
        if input_ids.device != device:
            input_ids = input_ids.to(device)
            
        return input_ids


# 工具函数：获取张量的设备与类型
def get_tensor_info(tensor):
    return f"Type: {tensor.dtype}, Device: {tensor.device}, Shape: {tensor.shape}"


class Inference:
    def __init__(
        self, path, args, model_args, codec_ckpt_path, codec_config_path
    ) -> None:
        self.args = args
        self.group_size = 3
        
        # 检测可用的GPU数量
        self.num_gpus = torch.cuda.device_count()
        print(f"检测到 {self.num_gpus} 个可用GPU")
        
        # 设置主设备为第一个GPU
        self.main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"主设备: {self.main_device}")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        padding_idx = self.tokenizer.pad_token_id
        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")

        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.end_empty_token = self.tokenizer.convert_tokens_to_ids("<|end_empty|>")

        # 为多GPU设置device_map
        if self.num_gpus > 1:
            # 自动平衡各GPU负载
            device_map = "auto"
            print("使用自动设备映射分布模型到多个GPU")
        else:
            # 单GPU情况
            device_map = {"": 0}
            print("仅使用单个GPU")
            
        # 加载模型时使用device_map
        self.model = MIMOLlamaForCausalLM.from_pretrained(
            path,
            padding_idx=padding_idx,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            args=model_args,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        # 打印模型各部分的设备分布情况
        if hasattr(self.model, 'hf_device_map'):
            print("模型设备映射:")
            for module_name, device_id in self.model.hf_device_map.items():
                print(f"  - {module_name}: {device_id}")
        
        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.generate_kwargs = {
            "max_new_tokens": 5000,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }

        # 音频解码器放在主设备上
        self.generator = SpeechGPT2Tokenizer.load_from_checkpoint(
            config_path=codec_config_path, checkpoint_path=codec_ckpt_path
        )
        self.generator = self.generator.to(self.main_device)
        # 确保所有参数都在同一设备上
        for param in self.generator.parameters():
            if param.device != self.main_device:
                param.data = param.data.to(self.main_device)
        self.generator.eval()
        self.generator = torch.compile(self.generator, mode="reduce-overhead")

        self.history = []
        self.greeting = None
        
        print(f"Inference引擎初始化完成，主设备: {self.main_device}")

    def set_greeting(self, text, audio):
        text = torch.tensor(text).to(self.main_device)
        audio = torch.tensor(audio).reshape(3, -1).to(self.main_device)
        self.greeting = [
            InputSegment(f"[|SpeechGPT|]: "),
            InputSegment(
                tokenized_text=text,
                audio=audio,
            ),
            InputSegment(f" ###\n{self.tokenizer.eos_token}"),
        ]

        greeting_audio_detokenized = self.generator.inference_detokenize(
            audio.reshape(-1, 3)
            .unsqueeze(0)
            .permute(2, 0, 1)
            .type(torch.LongTensor)
            .to(self.main_device)
        )
        return (
            24000,
            greeting_audio_detokenized.reshape(-1).detach().cpu().numpy(),
        )

    def clear_history(self):
        self.history.clear()

    def read_wav(self, audio_path: str, sampling_rate: int):
        wav, raw_sample_rate = torchaudio.load(audio_path)  # (1, T)   tensor
        if raw_sample_rate != sampling_rate:
            wav = torchaudio.functional.resample(
                wav, raw_sample_rate, sampling_rate
            )  # tensor
        return wav

    def preprocess(
        self,
        task: Union[None, str] = None,
        input: Union[None, str] = None,
        instruction: Union[None, str] = None,
        add_silence_at_end=True,
        silence_frames=8,
        audio_channels=3,
        group_size=4,
        mode="s2s",
        transcript=None,
    ):
        if type(input) != str:
            wav = (
                self.read_wav(input, self.generator.sampling_rate)
                .reshape(1, 1, -1)
                .to(self.main_device)
            )

            tokens = self.generator.inference_tokenize(wav)  # [n_vq, B, t]
            token_flat = (
                tokens.squeeze(1).permute(1, 0).reshape(-1).detach().cpu().numpy()
            )  # [T*n_q]

            silence_tokens = torch.tensor([688, 131, 226], device=self.main_device)
            token_flat = np.concatenate(
                [token_flat, np.tile(silence_tokens.cpu().numpy(), silence_frames)]
            )
            token_flat = np.concatenate(
                [
                    token_flat,
                    np.tile(
                        silence_tokens.cpu().numpy(),
                        (
                            group_size * audio_channels
                            - token_flat.shape[0] % (group_size * audio_channels)
                        )
                        // len(silence_tokens.cpu().numpy()),
                    ),
                ]
            )
            audio_tokenized = torch.tensor(token_flat, device=self.main_device)
        else:
            text = input

        assert self.greeting, "Must load greeting first"

        prompt = (
            [
                InputSegment(
                    f"You are an helpful assistant. You should answer the user's {'text' if mode[0] == 't' else 'speech'} questions in {'text' if mode[2] == 't' else 'speech'}.\n\n\n",
                ),
                *self.greeting,
            ]
            if not self.history
            else []
        )
        prompt += [
            InputSegment(f"[|Human|]: "),
            (
                InputSegment("", audio=audio_tokenized)
                if mode[0] == "s"
                else InputSegment(transcript)
            ),
            InputSegment(f" ###\n[|SpeechGPT|]: "),
        ]

        # 确保张量都在主设备上
        input_ids = [seg.to_input_id(self.tokenizer, group_size, self.main_device) for seg in prompt]
        input_ids = torch.cat(input_ids, dim=1)
        
        return input_ids

    def forward(
        self,
        task: Union[None, str] = None,
        input: Union[None, str] = None,
        instruction: Union[None, str] = None,
        mode: Union[None, str] = "s2s",
        text: Union[None, str] = None,
        audio_channels=3,
    ):
        group_size = self.group_size
        with torch.no_grad():
            input_ids = self.preprocess(
                task=task,
                input=input,
                instruction=instruction,
                group_size=group_size,
                audio_channels=audio_channels,
                mode=mode,
                transcript=text,
            )

            generation_config = GenerationConfig(**self.generate_kwargs)

            # 将输入重新整形
            input_ids = input_ids.T.reshape(1, -1)
            
            # 如果有历史记录，连接历史和当前输入
            if self.history:
                # 确保历史和当前输入在同一设备上
                history_device = self.history[0].device
                if input_ids.device != history_device:
                    input_ids = input_ids.to(history_device)
                input_ids = torch.cat(self.history + [input_ids], dim=-1)
            
            prompt_length = input_ids.shape[1] // (audio_channels + 1)
            stopping_criteria = [
                MIMOStopper(
                    self.tokenizer.eos_token_id,
                    group_size,
                    audio_channels,
                    max_tokens=1024 + prompt_length,
                )
            ]

            # 检查输入设备与模型设备是否匹配
            if hasattr(self.model, 'device_map'):
                first_param_device = next(self.model.parameters()).device
                if input_ids.device != first_param_device:
                    input_ids = input_ids.to(first_param_device)

            # 执行推理
            generated_ids = self.model.generate(
                input_ids,
                generation_config,
                stopping_criteria=stopping_criteria,
            )
            
            # 更新历史记录
            self.history = [generated_ids]

            # 将生成的ID移到CPU处理
            generated_ids = (
                generated_ids.int().cpu().reshape(-1, 4).T[:, prompt_length:]
            )

            text = generated_ids[0, ::group_size][:-1]
            detokenized_text = self.tokenizer.decode(text, skip_special_tokens=True)

            answer = {
                "speech": "",
                "thought": detokenized_text,
                "result": "",
            }

            # Find <|sosp|> and <|eosp|> tokens locations in text channel token sequence
            sosp_idx_locations = (text == self.sosp_idx).nonzero(as_tuple=True)[0]
            eosp_idx_locations = (text == self.eosp_idx).nonzero(as_tuple=True)[0]
            if len(sosp_idx_locations) == 0:
                print("No <|sosp|> token found in the text channel")
            else:
                if len(eosp_idx_locations) == 0:
                    eosp_idx_locations = [text.shape[0]]
                sosp_idx_location = sosp_idx_locations[0] * group_size
                eosp_idx_location = eosp_idx_locations[0] * group_size
                audio_sequence = generated_ids[
                    :, sosp_idx_location + group_size : eosp_idx_location
                ]
                speech_sequence = audio_sequence[1:].T.flatten()
                assert (speech_sequence < 1024).all()
                answer["result"] = detokenized_text.strip().replace("<|empty|>", ".")

                answer["speech"] = "".join([f"<{i}>" for i in speech_sequence])

            # dump wav
            wav = torch.tensor(0)
            if answer["speech"]:
                tokens = torch.tensor(
                    [int(num) for num in re.findall(r"(\d+)>", answer["speech"])]
                )
                # 确保音频生成在主设备上进行
                x = (
                    tokens.reshape(-1, 3)
                    .unsqueeze(0)
                    .permute(2, 0, 1)
                    .type(torch.LongTensor)
                    .to(self.main_device)
                )  # [n_vq, B, t]
                wav = self.generator.inference_detokenize(x)
                return detokenized_text, (24000, wav.reshape(-1).detach().cpu().numpy())

            return detokenized_text, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpt/",
    )
    parser.add_argument(
        "--codec_ckpt_path",
        type=str,
    )
    parser.add_argument(
        "--codec_config_path", type=str, default="Codec/config/sg2_codec_config.yaml"
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="强制仅使用单个GPU，不进行多GPU分布",
    )
    args = parser.parse_args()
    return args


class MIMOInterface:
    def __init__(self):
        self.args = parse_args()
        parser = HfArgumentParser((MIMOModelArguments,))
        self.model_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        self.model_args.model_name_or_path = self.args.model_path

        # 如果指定单GPU模式，设置环境变量
        if self.args.single_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("已启用单GPU模式，仅使用第一个GPU")

        self.inference = Inference(
            self.args.model_path,
            self.args,
            self.model_args,
            self.args.codec_ckpt_path,
            self.args.codec_config_path,
        )

    def process_input(self, audio_input, text_input, mode):
        try:
            # Handle audio input
            if audio_input is not None:
                buffer = io.BytesIO()
                sf.write(buffer, audio_input[1], audio_input[0], format="WAV")
                buffer.seek(0)
                input_data = buffer
            else:
                input_data = text_input

            return self.inference.forward(
                task="thought", input=input_data, text=text_input, mode=mode
            )

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"错误详情: {error_trace}")
            return f"Error: {str(e)}", None

    def process_greeting(self, greeting_source, greeting_line_idx):
        greeting_line_idx = int(greeting_line_idx)
        with open(greeting_source, "r") as f:
            for idx, line in enumerate(f):
                if idx == greeting_line_idx:
                    greeting = json.loads(line)
                    greeting_text = greeting["text"]
                    greeting_audio = greeting["audio"]
                    break
        self.inference.clear_history()
        return self.inference.set_greeting(greeting_text, greeting_audio)

    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("# SpeechGPT 2.0-preview")

            gr.Markdown("## Greeting Preview")

            with gr.Row():
                with gr.Column():
                    greeting_source = gr.Textbox(
                        label="Greeting File", value="extra/greetings.jsonl"
                    )
                    greeting_line_idx = gr.Textbox(label="Greeting ID", value="0")
                    load_greeting_btn = gr.Button("Load Greeting")

                with gr.Column():
                    greeting_audio = gr.Audio(label="Greeting Preview")

            gr.Markdown(
                "## Model Interaction\n\nNote: the model expects a greeting message to be played before first interaction with user. Make sure to load a greeting before sending the first message. Changing greeting message will clear chat history."
            )

            mode = gr.Radio(
                ["s2s", "s2t", "t2s", "t2t"],
                label="Interaction Mode",
                value="s2s",
                info="s2s: speech-to-speech, s2t: speech-to-text, t2s: text-to-speech, t2t: text-to-text\nModifying interaction mode will clear chat history.",
            )

            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["microphone"], type="numpy")
                    text_input = gr.Textbox(label="Text Input", visible=False)
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_history_btn = gr.Button("Clear History")

                with gr.Column():
                    text_output = gr.Textbox(label="Text")
                    audio_output = gr.Audio(label="Speech")

            submit_btn.click(
                fn=self.process_input,
                inputs=[audio_input, text_input, mode],
                outputs=[text_output, audio_output],
            )

            clear_history_btn.click(
                fn=self.inference.clear_history,
            )

            load_greeting_btn.click(
                fn=self.process_greeting,
                inputs=[greeting_source, greeting_line_idx],
                outputs=[greeting_audio],
            )

            def update_inputs(mode_value):
                self.inference.clear_history()
                if mode_value[0] == "t":
                    return gr.update(visible=False, value=None), gr.update(visible=True)
                elif mode_value[0] == "s":
                    return gr.update(visible=True), gr.update(visible=False, value=None)

            mode.change(
                fn=update_inputs, inputs=mode, outputs=[audio_input, text_input]
            )

        return demo


if __name__ == "__main__":
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"当前使用的CUDA设备: {torch.cuda.current_device()}")
    
    interface = MIMOInterface()
    demo = interface.create_interface()
    demo.launch()

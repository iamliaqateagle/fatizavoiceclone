# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files
import webbrowser
import time

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS"  # Keep original model name internally
MODEL_DISPLAY_NAMES = {"F5-TTS": "Alpha", "E2-TTS": "Beta"}  # Mapping for display names
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models

vocoder = load_vocoder()


def load_f5tts(ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts(ckpt_path=str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))):
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    # Convert display name back to internal model name
    internal_model = "F5-TTS" if model == "Alpha" else "E2-TTS" if model == "Beta" else model

    if internal_model == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif internal_model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(internal_model, list) and internal_model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != internal_model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(internal_model[1], vocab_path=internal_model[2], model_cfg=internal_model[3])
            pre_custom_path = internal_model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            style = tokens[i].strip()
            current_style = style
    return segments


# Define app_tts first
with gr.Blocks() as app_tts:
    with gr.Column(scale=1):
        with gr.Group():
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 1.5rem">
                    <h2 style="font-size: 1.8rem; font-weight: 600; margin-bottom: 0.75rem">Basic Voice Cloning</h2>
                    <p style="font-size: 1rem; opacity: 0.8; line-height: 1.5">Transform any voice with a single reference audio sample</p>
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ref_audio_input = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        elem_id="ref-audio",
                        container=True
                    )
                    ref_text_input = gr.Textbox(
                        label="Reference Text (Optional)",
                        info="Leave blank for auto-transcription",
                        lines=2,
                        elem_id="ref-text",
                        container=True
                    )
                
                with gr.Column(scale=1):
                    gen_text_input = gr.Textbox(
                        label="Text to Generate",
                        lines=6,
                        elem_id="gen-text",
                        placeholder="Enter the text you want to convert to speech...",
                        container=True
                    )
                    
                    with gr.Row():
                        generate_btn = gr.Button(
                            "🎙️ Generate Speech",
                            variant="primary",
                            scale=2,
                            min_width=200
                        )
                        
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        remove_silence = gr.Checkbox(
                            label="Remove Silences",
                            value=False,
                            info="Remove long silences from output",
                            container=True
                        )
                        speed_slider = gr.Slider(
                            label="Speed",
                            minimum=0.3,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            container=True
                        )
                    
                    with gr.Column(scale=1):
                        nfe_slider = gr.Slider(
                            label="Quality (NFE Steps)",
                            minimum=4,
                            maximum=64,
                            value=32,
                            step=2,
                            container=True
                        )
                        cross_fade_duration_slider = gr.Slider(
                            label="Cross-Fade Duration",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.15,
                            step=0.01,
                            container=True
                        )

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    audio_output = gr.Audio(
                        label="Generated Speech",
                        elem_id="audio-output",
                        container=True,
                        show_label=True
                    )
                with gr.Column(scale=1):
                    spectrogram_output = gr.Image(
                        label="Spectrogram",
                        elem_id="spectrogram",
                        container=True,
                        show_label=True
                    )

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )


# Define app_multistyle second
with gr.Blocks() as app_multistyle:
    with gr.Column():
        gr.Markdown(
            """
            # Multi-Style Voice Generation
            
            Create dynamic conversations with multiple voice styles and emotions
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ### Example Format
                    ```
                    {Regular} Hello, I'd like to order a sandwich.
                    {Surprised} What do you mean you're out of bread?
                    {Sad} I really wanted a sandwich though...
                    {Angry} This is unacceptable!
                    {Whisper} I'll just go somewhere else...
                    ```
                    """
                )
            
            with gr.Column():
                gr.Markdown(
                    """
                    ### Multi-Speaker Example
                    ```
                    {Speaker1_Happy} Hi there!
                    {Speaker2_Regular} Hello, how can I help?
                    {Speaker1_Excited} I'd like to place an order.
                    {Speaker2_Friendly} Of course!
                    ```
                    """
                )

        gr.Markdown(
            """
            ### Voice Styles
            Configure different voice styles by adding reference audio for each type
            """
        )

        # New section for multistyle generation
        gr.Markdown(
            """
            # Multiple Speech-Type Generation

            This section allows you to generate multiple speech types or multiple people's voices. 
            Enter your text in the format shown below, and the system will generate speech using the appropriate type. 
            If unspecified, the model will use the regular speech type. 
            The current speech type will be used until the next speech type is specified.
            """
        )

        with gr.Row():
            gr.Markdown(
                """
                **Example Input:**                                                                      
                {Regular} Hello, I'd like to order a sandwich please.                                                         
                {Surprised} What do you mean you're out of bread?                                                                      
                {Sad} I really wanted a sandwich though...                                                              
                {Angry} You know what, darn you and your little shop!                                                                       
                {Whisper} I'll just go back home and cry now.                                                                           
                {Shouting} Why me?!                                                                         
                """
            )

            gr.Markdown(
                """
                **Example Input 2:**                                                                                
                {Speaker1_Happy} Hello, I'd like to order a sandwich please.                                                            
                {Speaker2_Regular} Sorry, we're out of bread.                                                                                
                {Speaker1_Sad} I really wanted a sandwich though...                                                                             
                {Speaker2_Whisper} I'll give you the last one I was hiding.                                                                     
                """
            )

        gr.Markdown(
            "Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
        )

        # Regular speech type (mandatory)
        with gr.Row() as regular_row:
            with gr.Column():
                regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
                regular_insert = gr.Button("Insert Label", variant="secondary")
            regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
            regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

        # Regular speech type (max 100)
        max_speech_types = 100
        speech_type_rows = [regular_row]
        speech_type_names = [regular_name]
        speech_type_audios = [regular_audio]
        speech_type_ref_texts = [regular_ref_text]
        speech_type_delete_btns = [None]
        speech_type_insert_btns = [regular_insert]

        # Additional speech types (99 more)
        for i in range(max_speech_types - 1):
            with gr.Row(visible=False) as row:
                with gr.Column():
                    name_input = gr.Textbox(label="Speech Type Name")
                    delete_btn = gr.Button("Delete Type", variant="secondary")
                    insert_btn = gr.Button("Insert Label", variant="secondary")
                audio_input = gr.Audio(label="Reference Audio", type="filepath")
                ref_text_input = gr.Textbox(label="Reference Text", lines=2)
            speech_type_rows.append(row)
            speech_type_names.append(name_input)
            speech_type_audios.append(audio_input)
            speech_type_ref_texts.append(ref_text_input)
            speech_type_delete_btns.append(delete_btn)
            speech_type_insert_btns.append(insert_btn)

        # Button to add speech type
        add_speech_type_btn = gr.Button("Add Speech Type")

        # Keep track of autoincrement of speech types, no roll back
        speech_type_count = 1

        # Function to add a speech type
        def add_speech_type_fn():
            row_updates = [gr.update() for _ in range(max_speech_types)]
            global speech_type_count
            if speech_type_count < max_speech_types:
                row_updates[speech_type_count] = gr.update(visible=True)
                speech_type_count += 1
            else:
                gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
            return row_updates

        add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

        # Function to delete a speech type
        def delete_speech_type_fn():
            return gr.update(visible=False), None, None, None

        # Update delete button clicks
        for i in range(1, len(speech_type_delete_btns)):
            speech_type_delete_btns[i].click(
                delete_speech_type_fn,
                outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
            )

        # Text input for the prompt
        gen_text_input_multistyle = gr.Textbox(
            label="Text to Generate",
            lines=10,
            placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
        )

        def make_insert_speech_type_fn(index):
            def insert_speech_type_fn(current_text, speech_type_name):
                current_text = current_text or ""
                speech_type_name = speech_type_name or "None"
                updated_text = current_text + f"{{{speech_type_name}}} "
                return updated_text

            return insert_speech_type_fn

        for i, insert_btn in enumerate(speech_type_insert_btns):
            insert_fn = make_insert_speech_type_fn(i)
            insert_btn.click(
                insert_fn,
                inputs=[gen_text_input_multistyle, speech_type_names[i]],
                outputs=gen_text_input_multistyle,
            )

        with gr.Accordion("Advanced Settings", open=False):
            remove_silence_multistyle = gr.Checkbox(
                label="Remove Silences",
                value=True,
            )

        # Generate button
        generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

        # Output audio
        audio_output_multistyle = gr.Audio(label="Synthesized Audio")

        @gpu_decorator
        def generate_multistyle_speech(
            gen_text,
            *args,
        ):
            speech_type_names_list = args[:max_speech_types]
            speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
            speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
            remove_silence = args[3 * max_speech_types]
            # Collect the speech types and their audios into a dict
            speech_types = OrderedDict()

            ref_text_idx = 0
            for name_input, audio_input, ref_text_input in zip(
                speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
            ):
                if name_input and audio_input:
                    speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
                else:
                    speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
                ref_text_idx += 1

            # Parse the gen_text into segments
            segments = parse_speechtypes_text(gen_text)

            # For each segment, generate speech
            generated_audio_segments = []
            current_style = "Regular"

            for segment in segments:
                style = segment["style"]
                text = segment["text"]

                if style in speech_types:
                    current_style = style
                else:
                    gr.Warning(f"Type {style} is not available, will use Regular as default.")
                    current_style = "Regular"

                try:
                    ref_audio = speech_types[current_style]["audio"]
                except KeyError:
                    gr.Warning(f"Please provide reference audio for type {current_style}.")
                    return [None] + [speech_types[style]["ref_text"] for style in speech_types]
                ref_text = speech_types[current_style].get("ref_text", "")

                # Generate speech for this segment
                audio_out, _, ref_text_out = infer(
                    ref_audio, ref_text, text, tts_model_choice, remove_silence, 0, show_info=print
                )  # show_info=print no pull to top when generating
                sr, audio_data = audio_out

                generated_audio_segments.append(audio_data)
                speech_types[current_style]["ref_text"] = ref_text_out

            # Concatenate all audio segments
            if generated_audio_segments:
                final_audio_data = np.concatenate(generated_audio_segments)
                return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
            else:
                gr.Warning("No audio generated.")
                return [None] + [speech_types[style]["ref_text"] for style in speech_types]

        generate_multistyle_btn.click(
            generate_multistyle_speech,
            inputs=[
                gen_text_input_multistyle,
            ]
            + speech_type_names
            + speech_type_audios
            + speech_type_ref_texts
            + [
                remove_silence_multistyle,
            ],
            outputs=[audio_output_multistyle] + speech_type_ref_texts,
        )

        # Validation function to disable Generate button if speech types are missing
        def validate_speech_types(gen_text, regular_name, *args):
            speech_type_names_list = args

            # Collect the speech types names
            speech_types_available = set()
            if regular_name:
                speech_types_available.add(regular_name)
            for name_input in speech_type_names_list:
                if name_input:
                    speech_types_available.add(name_input)

            # Parse the gen_text to get the speech types used
            segments = parse_speechtypes_text(gen_text)
            speech_types_in_text = set(segment["style"] for segment in segments)

            # Check if all speech types in text are available
            missing_speech_types = speech_types_in_text - speech_types_available

            if missing_speech_types:
                # Disable the generate button
                return gr.update(interactive=False)
            else:
                # Enable the generate button
                return gr.update(interactive=True)

        gen_text_input_multistyle.change(
            validate_speech_types,
            inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
            outputs=generate_multistyle_btn,
        )


# Define main app last
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
)) as app:
    with gr.Column(elem_id="app-container"):
        gr.Markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1.5rem">
                <h1 style="font-size: 2.5rem; font-weight: 600; margin-bottom: 0.5rem">Fatiza Realistic Voice Cloning</h1>
                <p style="font-size: 1.1rem; opacity: 0.8">{"A local web UI" if not USING_SPACES else "An online demo"} for advanced voice cloning with multi-style support</p>
            </div>

            <div style="padding: 1rem; border-radius: 0.5rem; background: rgba(59, 130, 246, 0.1); margin-bottom: 1.5rem">
                <h4 style="font-weight: 500; margin-bottom: 0.5rem">📝 Important Notes</h4>
                <ul style="list-style-type: none; padding-left: 0">
                    <li style="margin-bottom: 0.5rem">• Supports English and Chinese</li>
                    <li style="margin-bottom: 0.5rem">• For best results, keep reference clips short (<15s)</li>
                    <li style="margin-bottom: 0.5rem">• Convert reference audio to WAV/MP3 if needed</li>
                    <li>• Reference text will be auto-transcribed if not provided</li>
                </ul>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                choose_tts_model = gr.Radio(
                    choices=["Alpha", "Beta"],
                    label="Select Voice Model",
                    value="Alpha",
                    container=True,
                    interactive=True,
                    elem_id="model-selector"
                )

        def update_model_choice(display_name):
            global tts_model_choice
            if display_name == "Alpha":
                tts_model_choice = "F5-TTS"
            elif display_name == "Beta":
                tts_model_choice = "E2-TTS"
            return None

        choose_tts_model.change(
            fn=update_model_choice,
            inputs=choose_tts_model,
            outputs=None,
            queue=False
        )

        with gr.Tabs(selected=0) as tabs:
            with gr.TabItem("Basic TTS", id=0):
                app_tts.render()
            with gr.TabItem("Multi-Speech", id=1):
                app_multistyle.render()


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
def main(port, host, share, api, root_path):
    global app
    print("Starting app...")
    
    # Set default host and port if not provided
    if host is None:
        host = "127.0.0.1"
    if port is None:
        port = 7860
        
    server = app.queue(api_open=api).launch(
        server_name=host, 
        server_port=port, 
        share=share, 
        show_api=api, 
        root_path=root_path, 
        prevent_thread_lock=True
    )
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Construct the local URL
    local_url = f"http://{host}:{port}"
    print(f"Opening {local_url} in your browser...")
    webbrowser.open(local_url)
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()

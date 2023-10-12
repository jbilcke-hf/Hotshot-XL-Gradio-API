import hashlib
import os
import uuid
import requests
import pathlib
import subprocess
import gradio as gr
from huggingface_hub import snapshot_download, HfFileSystem

fs = HfFileSystem()

# execute a CLI command
def execute_command(command: str) -> None:
    subprocess.run(command, check=True)

# Download stable-diffusion-xl
#local_dir = f"./stablediff"
#snapshot_download(
#        "stabilityai/stable-diffusion-xl-base-1.0",
#        local_dir=local_dir,
#        repo_type="model",
#        ignore_patterns=".gitattributes",
#        #token=hf_token
#    )
#spatial_unet_base = f"./stablediff/unet"


def get_files(file_paths):
    last_files = {}  # Dictionary to store the last file for each path

    for file_path in file_paths:
        # Split the file path into directory and file components
        directory, file_name = file_path.rsplit('/', 1)
    
        # Update the last file for the current path
        last_files[directory] = file_name
    
    # Extract the last files from the dictionary
    result = list(last_files.values())

    return result


def load_lora_weights(lora_id):
    # List all ".safetensors" files in repo
    sfts_available_files = fs.glob(f"{lora_id}/*safetensors")
    sfts_available_files = get_files(sfts_available_files)

    if sfts_available_files == []:
        sfts_available_files = ["NO SAFETENSORS FILE"]

    print(f"Safetensors available: {sfts_available_files}")
    
    return sfts_available_files[0]


def infer(prompt: str, lora: str = None, size: str = '512x512'):
    width, height = map(int, size.split('x'))

    if lora:  # only download if a link is provided
        print(f"lora model id: {lora}")
        #lora = lora.strip()  # remove leading and trailing white spaces
        lora_weights = load_lora_weights(lora)
        lora_path = lora
    else:
        lora_path = None

    output = "output.gif"

    command = [
      f"python", 
      f"inference.py",
      f"--prompt={prompt}", 
      f"--output={output}",
      f"--width={width}",
      f"--height={height}"
    ]

    if lora:
        #command.append(f"--spatial_unet_base={spatial_unet_base}")
        command.append(f"--lora={lora_path}")
        command.append(f"--weight_name={lora_weights}")

    execute_command(command)

    # note: we do not delete the lora and instead we keep it in the cache (persistent storage)

    return output

css="""
#col-container{
    margin: 0 auto;
    max-width: 720px;
    text-align: left;
}
"""
    
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <h2 style="text-align: center;">Hotshot-XL Text to GIF</h2>
        <p style="text-align: center;">Hotshot-XL is an AI text-to-GIF model trained to work alongside Stable Diffusion XL</p>
                """)
        prompt = gr.Textbox(label="Prompt")
        lora = gr.Textbox(label="LoRA", value=None)
        size = gr.Radio(label="Size", choices=[
            '320x768',
            '384x672',
            '416x608',
            '512x512',
            '608x416',
            '672x384',
            '768x320'
        ], value='512x512')
        submit_btn = gr.Button("Submit")
        gif_result = gr.Image(label="Gif")
    submit_btn.click(fn=infer, inputs=[prompt, lora, size], outputs=[gif_result])

demo.queue(max_size=12).launch()

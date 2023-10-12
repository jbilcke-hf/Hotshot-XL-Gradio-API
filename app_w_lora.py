import hashlib
import os
import uuid
import requests
import tempfile
import pathlib

# Hash the url to get a unique filename for caching
def hash_url(url: str) -> str:
    hash_object = hashlib.sha1(url.encode())
    return hash_object.hexdigest()

# download a file URL (1st param) to a specific path (2nd param)
def download_file(url: str, dest: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# delete a file or folder
def delete_file_if_exists(filepath: str) -> None:
    os.remove(filepath) if os.path.exists(filepath) else None

# execute a CLI command
def execute_command(command: str) -> None:
    subprocess.run(command, check=True)

# create a temporary file name in the /tmp folder
def create_temp_file(prefix: str, postfix: str) -> str:
    return f"{tempfile.gettempdir()}/{prefix}{uuid.uuid4().hex}{postfix}"


import subprocess
import gradio as gr

storage_path = os.getenv("STORAGE_PATH", './sandbox')
loras_dir_file_path = os.path.join(storage_path, "loras")
models_dir_file_path = os.path.join(storage_path, "models")

base_sdxl_dir = os.path.join(models_dir_file_path, "stable-diffusion-xl-base-1.0")
spatial_unet_base_global_var = os.path.join(base_sdxl_dir, "unet")

def infer(prompt: str, lora: str = None, size: str = '512x512'):
    width, height = map(int, size.split('x'))

    # This will ensure all necessary directories are created
    pathlib.Path(loras_dir_file_path).mkdir(parents=True, exist_ok=True)

    lora_path = os.path.join(loras_dir_file_path, hash_url(lora) + ".safetensors")

    if lora:  # only download if a link is provided
        lora = lora.strip()  # remove leading and trailing white spaces
        lora_path = os.path.join(loras_dir_file_path, hash_url(lora) + ".safetensors")
        download_file(lora, lora_path)
    else:
        lora_path = None

    output = create_temp_file(prefix="output_", postfix=".gif")

    command = [
      f"python", 
      f"inference.py",
      f"--prompt={prompt}", 
      f"--output={output}",
      f"--spatial_unet_base={spatial_unet_base_global_var}",
      f"--width={width}",
      f"--height={height}"
    ]

    if lora:
        lora_path = os.path.join(loras_dir_file_path, hash_url(lora) + ".safetensors")
        download_file(lora, lora_path)
        command.append(f"--lora={lora_path}")

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
        lora = gr.Textbox(label="LoRA", default=None)
        size = gr.Radio(label="Size", choices=[
            '320x768',
            '384x672',
            '416x608',
            '512x512',
            '608x416',
            '672x384',
            '768x320'
        ], default='512x512')
        submit_btn = gr.Button("Submit")
        gif_result = gr.Image(label="Gif")
    submit_btn.click(fn=infer, inputs=[prompt, lora, size], outputs=[gif_result])

demo.queue(max_size=12).launch()

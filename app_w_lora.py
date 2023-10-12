import os
import random
import subprocess
import gradio as gr
from huggingface_hub import snapshot_download, HfFileSystem, ModelCard

fs = HfFileSystem()

# execute a CLI command
def execute_command(command: str) -> None:
    subprocess.run(command, check=True)

def get_trigger_word(lora_id):
    # Get instance_prompt a.k.a trigger word
    card = ModelCard.load(lora_id)
    repo_data = card.data.to_dict()
    instance_prompt = repo_data.get("instance_prompt")

    if instance_prompt is not None:
        print(f"Trigger word: {instance_prompt}")
    else:
        instance_prompt = "no trigger word needed"
        print(f"Trigger word: no trigger word needed")
    return instance_prompt

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


def infer(prompt: str, negative_prompt: str, lora: str = None, size: str = '512x512', seed: int = -1):
    width, height = map(int, size.split('x'))
    
    if seed < 0 :
        seed = random.randint(0, 423538377342)
    
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
      f"--negative_prompt={negative_prompt}",
      f"--output={output}",
      f"--width={width}",
      f"--height={height}",
      f"--seed={seed}"
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
        <p style="text-align: center;">
            Hotshot-XL is an AI text-to-GIF model trained to work alongside Stable Diffusion XL <br />
            For faster inference, use the Hotshot website: www.hotshot.co
        </p>
                """)
        prompt = gr.Textbox(label="Prompt")
        # Advanced Settings
        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value="extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
            )
            with gr.Row():
                lora = gr.Textbox(label="Public LoRA ID", value=None, placeholder="username/custom_lora_name")
                lora_trigger = gr.Textbox(label="Trigger word", interactive=False)
            with gr.Row():
                size = gr.Dropdown(
                    label="Size", 
                    choices=[
                        '320x768',
                        '384x672',
                        '416x608',
                        '512x512',
                        '608x416',
                        '672x384',
                        '768x320'
                    ], value='512x512')
                
                seed = gr.Slider(
                    label="Seed",
                    info = "-1 denotes a random seed",
                    minimum=-1,
                    maximum=423538377342,
                    step=1,
                    value=-1
                )
        submit_btn = gr.Button("Submit")
        gif_result = gr.Image(label="Gif")
    lora.blur(fn=get_trigger_word, inputs=[lora], outputs=[lora_trigger], queue=False)
    submit_btn.click(fn=infer, inputs=[prompt, negative_prompt, lora, size, seed], outputs=[gif_result])

demo.queue(max_size=12).launch()

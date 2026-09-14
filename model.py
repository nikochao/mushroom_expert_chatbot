from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

model, processor = load(MODEL_PATH)
config = load_config(MODEL_PATH)

def get_formatted_prompt(inputs, images):
    return apply_chat_template(processor, config, inputs, num_images=len(images))


def get_model_output(inputs, images):
    formatted_prompt = get_formatted_prompt(inputs, images)
    return generate(
        model,
        processor,
        formatted_prompt,
        images,
        max_tokens=512,
        temperature=0.1,
        verbose=True,
    )

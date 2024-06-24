import asyncio
import json
import os
import pydantic
import uuid
import traceback
import torch
import gc

from huggingface_hub import hf_hub_download
from accelerate import PartialState
from huggingface_hub import logging
from aiohttp import web
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoencoderKL, DiffusionPipeline
# Remove when AutoPipelineForText2Image adds SD3
from diffusers import StableDiffusion3Pipeline
from torch.cuda.amp import autocast

#logging.set_verbosity_error()

class mugatu:
    def __init__(self, loop=None):
        self.allowed_devices = [0, 1]
        self.allowed_models = {
            "Text2Image": ["stabilityai/stable-diffusion-3-medium-diffusers", "stabilityai/sdxl-turbo", "cagliostrolab/animagine-xl-3.1", "RunDiffusion/Juggernaut-X-v10"],
            "Text2ImageLORA": [
                    ["stabilityai/stable-diffusion-xl-base-1.0", "bytedance/hyper-sd", "Hyper-SDXL-2steps-lora.safetensors", ""], 
                    ["stabilityai/stable-diffusion-xl-base-1.0", "ostris/ikea-instructions-lora-sdxl", "ikea_instructions_xl_v1_5.safetensors", ""],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "ciron2022/ascii-art", "ascii_art-sdxl.safetensors", "ascii_art"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "Norod78/sdxl-futurama-style-lora", "SDXL-FuturamaStyle-Lora.safetensors", "FuturamaStyle"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "ProomptEngineer/pe-funko-pop-diffusion-style", "PE_FunkoPopStyle.safetensors", "PEPopFigure"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "artificialguybr/TshirtDesignRedmond-V2", "TShirtDesignRedmondV2-Tshirtdesign-TshirtDesignAF.safetensors", "TshirtDesignAF"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "ostris/super-cereal-sdxl-lora", "cereal_box_sdxl_v1.safetensors", ""],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "lordjia/lelo-lego-lora", "lego_v2.0_XL_32.safetensors", "LEGO BrickHeadz"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "artificialguybr/PixelArtRedmond", "PixelArtRedmond-Lite64.safetensors", "Pixel Art, PixArFK"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "TheLastBen/Pikachu_SDXL", "pikachu.safetensors", "pikachu"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "Norod78/SDXL-Psychemelt-style-LoRA", "SDXL_Psychemelt_style_LoRA-000007.safetensors", "psychemelt style, in an LSD trip psychemelt style"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "chillpixel/blacklight-makeup-sdxl-lora", "pytorch_lora_weights.bin", "with blacklight makeup"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "nerijs/pixelportraits192-XL-v1.0", "pixelportraits192-v1-2151.safetensors", ""],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "joachimsallstrom/aether-cloud-lora-for-sdxl", "Aether_Cloud_v1.safetensors", "a cloud that looks like a"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "Fictiverse/Voxel_XL_Lora", "VoxelXL_v1.safetensors", "voxel style"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "goofyai/cyborg_style_xl", "cyborg_style_xl-off.safetensors", "cyborg style"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "WizWhite/wizard-s-vintage-board-games", "Wizards_Vintage_Board_Game.safetensors", "Vintage board game box"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "KappaNeuro/studio-ghibli-style", "Studio Ghibli Style.safetensors", "Studio Ghibli Style"],
                    ["stabilityai/stable-diffusion-xl-base-1.0", "artificialguybr/StickersRedmond", "StickersRedmond.safetensors", "Stickers"],
                ],
            "Image2Image": [],
            "Inpainting": [],
            "Text": [
                    ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", None, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
                    #["bartowski/Meta-Llama-3-8B-Instruct-GGUF", "Meta-Llama-3-8B-Instruct-Q6_K.gguf", "meta-llama/Meta-Llama-3-8B"],
                    #["QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", "Meta-Llama-3-8B-Instruct.Q3_K_S.gguf", "meta-llama/Meta-Llama-3-8B"],
                    #["google/gemma-7b-it-GGUF", "gemma-7b-it.gguf", "google/gemma-7b"],
                ],
            "Encoder": ["madebyollin/sdxl-vae-fp16-fix"]
        }
        self.loop = loop if loop else asyncio.get_event_loop()
        self.job_q = asyncio.Queue()
        self.app = web.Application()
        self.app.add_routes([web.static('/images', './user_images')])  
        self.distributed_state = PartialState().device
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')

    async def preload_models(self):
        print("[I] Attempting to pre-download models to ensure quicker query responses.")
        for p_type, model_list in self.allowed_models.items():
            for model in model_list:
                #print(f"model {model}")
                if p_type in ["Text2Image", "Image2Image", "Inpainting"]:
                    pipeline = await self.get_pipeline(model)
                    del pipeline
                elif p_type in ["Text2ImageLORA"]:
                    pipeline = await self.get_pipeline(model[1])
                    if pipeline != None:
                        if len(model) > 2:
                            print(f"[I] Loading LORA weights from hub repo: {model[1]} weights: {model[2]}")
                            pipeline.load_lora_weights(hf_hub_download(model[1], model[2]))
                        else:
                            print(f"[I] Loading LORA weights from hub repo: {model[1]}")
                            pipeline.load_lora_weights(model[1])
                    del pipeline
                elif p_type == "Text":
                    model_o = await self.get_model(model[0])
                    del model_o
                elif p_type == "Encoder":
                    encoder = await self.get_encoder(model)
                    del encoder
                else:
                    print(f"[E] Unrecognized model type {p_type}")
                gc.collect()
                torch.cuda.empty_cache()
        print("[I] Finished preloading data.")
        return None

    def validate_model(self, model_name):
        for p_type, models in self.allowed_models.items():
            for model in models:
                if model_name == model:
                    return True
                elif type(model) is list:
                    if model_name in model:
                        return True
        return False

    # Gets list of allowed models
    def get_allowed_models(self):
        return self.allowed_models

    # This function uses the model name (or LORA name) and returns the model or model list (LoRA)
    def get_pipeline_info(self, model_name):
        for p_type, models in self.allowed_models.items():
            for model in models:
                if type(model) is list and p_type == "Text2ImageLORA":
                    if str(model_name).lower() == str(model[1]).lower():
                        return (p_type, model)
                if type(model) is list and p_type == "Text":
                    if str(model_name).lower() == str(model[0]).lower():
                        return (p_type, model)
                elif model_name == model:
                    return (p_type, model)
        return (None, None)

    def manage_queue(self):
        while not self.job_q.empty():
            job = self.job_q.get()
            print(f"[I] Processing job: {job}")

    async def handle_client(self, reader, writer):
        try:
            while True:
                data = await reader.read(10000)
                if not data:
                    break
                job = json.loads(data.decode())
                print("[I] Job received:", job)
                await self.job_q.put((job['prompt'], job['params'], writer))
                print("[I] Job queued")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start_server(self):
        #pre-download model files to ensure a quicker query
        await self.preload_models()
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        print("[I] Web server running at http://0.0.0.0:8080/images/")

        server = await asyncio.start_server(self.handle_client, '0.0.0.0', 8888)
        print("[I] Server started at 0.0.0.0:8888")

        asyncio.create_task(self.process_jobs())
        await server.serve_forever()

    async def process_jobs(self):
        print("[I] Starting to process jobs")
        while True:
            prompt, params, writer = await self.job_q.get()
            print("[I] Processing job:", prompt)
            #run job
            result = await self.process(prompt, params)
            #gc after job
            gc.collect()
            torch.cuda.empty_cache()
            #parse result
            response = json.dumps(result)
            try:
                writer.write(response.encode())
                await writer.drain()
                print(f"[I] Response sent {response}")
            except Exception as e:
                print(f"[E] Failed to send response: {str(e)}")
            finally:
                writer.close()

    def save_image(self, image):
        directory = "./user_images"  # Directory served by aiohttp
        os.makedirs(directory, exist_ok=True)
        
        # Generate a unique file name
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'wb') as f:
            image.save(f, format="PNG")
        
        # Return a URL path that is accessible via the web server
        webpath = f"http://mugatu:8080/images/{filename}" 
        return webpath

    async def process(self, prompt, params={}):
        print(f"[I] Received prompt {prompt}")
        try:
            required_params = ["model"]
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                print(f"[E] Missing required parameters: {', '.join(missing_params)}")
                return {"status": "error", "message": f"Missing required parameters: {', '.join(missing_params)}"}

            model = params.get("model")
            if not self.validate_model(model):
                print(f"[E] Model {model} not supported.")
                return {"status": "error", "message": f"Model {model} not supported."}

            negative_prompt = params.get("negative_prompt", "")
            width = params.get("width", 512)
            height = params.get("height", 512)
            guidance_scale = params.get("guidance_scale", 0.0)
            num_inference_steps = params.get("num_inference_steps", 2)
            strength = params.get("strength", 0.5)
            encoder=None
            
            p_type, p_model = self.get_pipeline_info(model)
            if p_type is None:
                print(f"[E] Model {model} not associated with a pipeline type.")
                return {"status": "error", "message": f"[E] Model {model} not associated with a pipeline type."}

            if p_type == "Text":

                if type(p_model) is list:
                    model_name = p_model[0]
                    model_gguf = p_model[1]
                    model_tokenizer = p_model[2]
                else:
                    model_name = p_model
                    model_gguf = None
                    model_tokenizer = None

                # Tokenize prompt
                tokenizer = await self.get_tokenizer(model_tokenizer)
                if tokenizer is None:
                    print(f"[E] Could get tokenizer for model {model} with prompt {prompt}")
                    return {"status": "error", "message": f"Could not get tokenizer for model {model} with prompt {prompt}."}

                model_o = await self.get_model(model_name)
                if model_o is None:

                    if 'tokenizer' in locals():
                        del tokenizer

                    print(f"[E] Could not get model for model {model} with prompt {prompt}")
                    return {"status": "error", "message": f"Could not get model for model {model} with prompt {prompt}."}

                templated_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, device_map=self.distributed_state)

                inputs = tokenizer.encode(templated_prompt, add_special_tokens=False, return_tensors="pt").to(self.distributed_state)

                #outputs = model_o.generate(input_ids=inputs.to(self.distributed_state), max_new_tokens=1800).to(self.distributed_state)
                outputs = model_o.generate(input_ids=inputs.to(self.distributed_state), max_new_tokens=1800, do_sample=True, temperature=0.4, top_k=50, top_p=0.95, repetition_penalty=1.10).to(self.distributed_state)

                #response = outputs[0][inputs.shape[-1]:]
                response = outputs[0]
                full_resp_message = tokenizer.decode(response, skip_special_tokens=True)
                resp_message = full_resp_message.split("<|assistant|>")[1]
                print(resp_message)

                if 'tokenizer' in locals():
                    del tokenizer

                if 'model_o' in locals():
                    del model_o

                if 'templated_prompt' in locals():
                    del templated_prompt

                if 'inputs' in locals():
                    del inputs

                if 'outputs' in locals():
                    del outputs

                if resp_message:
                    return {"status": "success", "response": resp_message}
                else:
                    return {"status": "error", "message": "No response generated."}

            elif p_type in ["Text2Image", "Image2Image", "Inpainting", "Text2ImageLORA"]:
                if type(p_model) is list:
                    model_name = p_model[0]
                else:
                    model_name = p_model

                print(f"[I] Model Name is {model_name}")
                if model_name != 'stabilityai/stable-diffusion-3-medium-diffusers':
                  # Get encoder (if needed) and pipeline / model (depending on type)
                  encoder = await self.get_encoder("madebyollin/sdxl-vae-fp16-fix")

                if p_type != "Text2ImageLORA":
                    pipeline = await self.get_pipeline(model_name, encoder)
                else:

                    if type(p_model) is list:
                        model_name = p_model[0]
                        model_lora = p_model[1]
                        model_weight = p_model[2]
                        model_trigger = p_model[3]
                    else:
                        model_name = p_model
                        model_lora = None
                        model_weight = None
                        model_trigger = None

                    if model_lora != None:
                        pipeline = await self.get_pipeline(model_lora, encoder, variant="fp16")
                        #pipeline = await self.get_pipeline(model_lora, encoder)
                        if pipeline != None:
                            if model_weight != None:
                                pipeline.load_lora_weights(hf_hub_download(model_lora, model_weight))
                            else:
                                pipeline.load_lora_weights(model_lora)
                            if model_trigger != None:
                                if type(prompt) is list:
                                    prompt[0] = f"{prompt[0]} {model_trigger}"
                                else:
                                    prompt = f"{prompt} {model_trigger}"
                if pipeline is None:
                    if 'encoder' in locals():
                        del encoder

                    print(f"[E] Could not process job for model {model} with prompt {prompt}")
                    return {"status": "error", "message": "Pipeline retrieval failed."}

                # Get output
                with autocast(True):
                    print(f"[I] Running pipeline for prompt {prompt}")
                    if 'encoder' in locals() and encoder != None:
                        images = pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            vae=encoder,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            strength=strength
                            )
                    else:
                        images = pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            #strength=strength
                            )

                    if 'encoder' in locals():
                        del encoder

                    if 'pipeline' in locals():
                        del pipeline

                    print(f"[I] Pipeline returned the following {images.images}")
                    if images.images:
                        # Save the image and get the path
                        filepath_list = []
                        for idx, image in enumerate(images.images):
                            filepath = self.save_image(image)  # Adjust according to your actual image data type
                            filepath_list.append(filepath)
                        return {"status": "success", "path": filepath_list}
                    else:
                        return {"status": "error", "message": "No images generated."}

        except Exception as e:
            print(f"[E] Error processing job for model {model} with prompt {prompt}: {str(e)}")
            traceback.print_exc()        
            return {"status": "error", "message": f"Error processing job: {str(e)}"}

        if 'templated_prompt' in locals():
            del templated_prompt
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs
        if 'encoder' in locals():
            del encoder
        if 'pipeline' in locals():
            del pipeline
        if 'tokenizer' in locals():
            del tokenizer
        if 'model_o' in locals():
            del model_o
                    
        return {"status": "error", "message": "No images generated."}
           
    # get output from job
    async def get_output(self):
        pass

    async def get_model(self, model_name, auto=True, safe_tensors=True, retry=True):
        print(f"[I] Attempting to get model {model_name}")

        try:
            if not self.validate_model(model_name):
                print(f"[E] Model {model_name} not supported.")
                return None

            p_type, p_model = self.get_pipeline_info(model_name)
            if p_type is None:
                print(f"[E] Model {p_model} not associated with a type.")
                return None

            if type(p_model) is list:
                model_name = p_model[0]
                model_gguf = p_model[1]
                model_tokenizer = p_model[2]
            else:
                model_name = p_model
                model_gguf = None
                model_tokenizer = None
         
            if auto == True:
                if model_gguf != None:
                    safe_tensors=False
                #accellerate seems to be broken w/ quants
                model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=safe_tensors, gguf_file=model_gguf, token=self.hf_token).to(self.distributed_state)
                return model
        except EnvironmentError as e:
            if retry == True:
                print(f"[E] Exception in get_model {e}, retrying with safe_tensors swapped")
                return await self.get_model(model_name, auto=auto, safe_tensors=(not safe_tensors), retry=False)
            else:
                print(f"[E] Exception in get_model {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"[E] Exception in get_model {e}")
            traceback.print_exc()
        return None

    async def get_pipeline(self, model, vae=None, variant=None, auto=True, safe_tensors=True, retry=True):
        print(f"[I] Attempting to get pipeline for model {model}")

        try:
            if not self.validate_model(model):
                print(f"[E] Model {model} not supported.")
                return None

            p_type, p_model = self.get_pipeline_info(model)
            if p_type is None:
                print(f"[E] Model {model} not associated with a pipeline type.")
                return None

            #since a model in the dict can include a LORA name and weights, use the base
            if type(p_model) is list:
                model_name = p_model[0]
                model_lora = p_model[1]
                model_weight = p_model[2]
                model_trigger = p_model[3]
            else:
                model_name = p_model
                model_lora = None
                model_weight = None
                model_trigger = None

            print(f"[I] Determined type is {p_type}, name is {model_name}")
     
            if auto == True:
                if p_type in ["Text2Image"]:
                    if model_name == "stabilityai/stable-diffusion-3-medium-diffusers":
                        return StableDiffusion3Pipeline.from_pretrained(model_name, variant=variant, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                    elif vae != None:
                        return AutoPipelineForText2Image.from_pretrained(model_name, variant=variant, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                    return AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant=variant, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                elif p_type in ["Text2ImageLORA"]:
                    if vae != None:
                        return AutoPipelineForText2Image.from_pretrained(model_name, vae=vae, torch_dtype=torch.float16, variant=variant, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                    return AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant=variant, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                elif p_type == "Image2Image":
                    if vae != None:
                        return AutoPipelineForImage2Image.from_pretrained(model_name, vae=vae, variant=variant, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                    return AutoPipelineForImage2Image.from_pretrained(model_name, variant=variant, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                elif p_type == "Inpainting":
                    if vae != None:
                        return AutoPipelineForInpainting.from_pretrained(model_name, vae=vae, variant=variant, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                    return AutoPipelineForInpainting.from_pretrained(model_name, torch_dtype=torch.float16, variant=variant, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
            print("[E] Could not set auto pipeline, only Text2Image, Image2Image and Inpainting types allowed.")
        except EnvironmentError as e:
            if retry == True:
                print(f"[E] Exception in get_pipeline {e}, retrying with safe_tensors set to False")
                return await self.get_pipeline(model, vae=vae, variant=variant, auto=auto, safe_tensors=(not safe_tensors), retry=False)
            else:
                print(f"[E] Exception in get_pipeline {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"[E] Exception in get_pipeline {e}")
            traceback.print_exc()
        return None

    async def get_tokenizer(self, model, auto=True, safe_tensors=True):
        print(f"[I] Attempting to get tokenizer for model {model}")

        try:
            if not self.validate_model(model):
                print(f"[E] Model {model} not supported.")
                return None
     
            if auto == True:
                tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token)
                return tokenizer

        except Exception as e:
            print(f"[E] Exception in get_tokenizer {e}")
            traceback.print_exc()
        return None

    async def get_encoder(self, model, auto=True, safe_tensors=True):
        print(f"[I] Attempting to get encoder model {model}")

        try:
            if not self.validate_model(model):
                print(f"[E] Model {model} not supported.")
                return None
     
            if auto == True:
                encoder = AutoencoderKL.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=safe_tensors, token=self.hf_token).to(self.distributed_state)
                return encoder

        except Exception as e:
            print(f"[E] Exception in get_encoder {e}")
            traceback.print_exc()
        return None


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    mugatu_server = mugatu(loop)
    loop.run_until_complete(mugatu_server.start_server())
    loop.run_forever()

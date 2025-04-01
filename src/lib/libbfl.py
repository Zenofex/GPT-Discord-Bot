import aiohttp
import asyncio
import argparse

BASE_URL = "https://api.bfl.ml/v1"

class BFLAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        print(f"[I] Initialized BFLAPI with provided API key.")
        self.session = aiohttp.ClientSession()

    async def close(self):
        print(f"[I] Closing the session.")
        await self.session.close()

    async def _request(self, method, endpoint, json=None):
        headers = {
            "x-key": self.api_key,  # API key passed as `x-key`
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{BASE_URL}{endpoint}"
        print(f"[I] Making {method} request to {url} with payload: {json}")
        async with self.session.request(method, url, json=json, headers=headers) as response:
            print(f"[I] Received response with status code: {response.status}")
            if response.status == 200:
                json_response = await response.json()
                print(f"[I] Successful response: {json_response}")
                return json_response
            else:
                error_text = await response.text()  # Get error response text
                print(f"[E] Error: {response.status}, {error_text}")
                raise Exception(f"Error: {response.status}, {error_text}")

    async def generate_image_flux_pro_11(self, prompt, width=1024, height=768, seed=None, safety_tolerance=2, prompt_upsampling=False):
        print(f"[I] Submitting task for FLUX 1.1 (pro) with prompt: {prompt}, width: {width}, height: {height}, seed: {seed}")
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "safety_tolerance": safety_tolerance,
        }
        return await self._request("POST", "/flux-pro-1.1", json=payload)

    async def generate_image_flux_pro(self, prompt, width=1024, height=768, steps=40, seed=None, guidance=2.5, safety_tolerance=2, prompt_upsampling=False):
        print(f"[I] Submitting task for FLUX Pro with prompt: {prompt}, width: {width}, height: {height}, steps: {steps}")
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "guidance": guidance,
            "safety_tolerance": safety_tolerance,
            "interval": 2,
        }
        return await self._request("POST", "/flux-pro", json=payload)

    async def generate_image_flux_dev(self, prompt, width=1024, height=768, steps=28, seed=None, guidance=3, safety_tolerance=2, prompt_upsampling=False):
        print(f"[I] Submitting task for FLUX Dev with prompt: {prompt}, width: {width}, height: {height}, steps: {steps}")
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "guidance": guidance,
            "safety_tolerance": safety_tolerance,
        }
        return await self._request("POST", "/flux-dev", json=payload)

    async def get_result(self, task_id):
        print(f"[I] Fetching result for task ID: {task_id}")
        return await self._request("GET", f"/get_result?id={task_id}")

    async def poll_result(self, task_id, interval=5):
        print(f"[I] Polling for result with task ID: {task_id}")
        while True:
            result = await self.get_result(task_id)
            status = result.get("status")
            print(f"[I] Task {task_id} status: {status}")

            if status == "Ready":
                print(f"[I] Task {task_id} is ready.")
                return result
            elif status == "Request Moderated":
                print(f"[E] Task {task_id} failed with status: Request Moderated")
                await send_channel_msg(message_ctx, "Your request has been flagged by the safety checks and moderated.")
                raise Exception(f"Task failed with status: {status}")
            elif status == "Content Moderated":
                print(f"[E] Task {task_id} failed with status: Content Moderated")
                await send_channel_msg(message_ctx, "Your request has been flagged by the safety checks and moderated.")
                raise Exception(f"Task failed with status: {status}")
            elif status in ["Error", "Task not found", "Content Moderated"]:
                print(f"[E] Task {task_id} failed with status: {status}")
                raise Exception(f"Task failed with status: {status}")
            else:
                print(f"[I] Task {task_id} is still in progress. Retrying in {interval} seconds.")
            await asyncio.sleep(interval)

# Standalone CLI with argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Submit image generation tasks to BFL API")

    # API Key
    parser.add_argument('--api-key', required=True, help="Your API key for BFL (passed as x-key)")

    # Model selection
    parser.add_argument('--model', required=True, choices=['flux-pro-1.1', 'flux-pro', 'flux-dev'],
                        help="Select the model version to use")

    # Image generation parameters
    parser.add_argument('--prompt', required=True, help="Text prompt for image generation")
    parser.add_argument('--width', type=int, default=1024, help="Width of the generated image (multiple of 32)")
    parser.add_argument('--height', type=int, default=768, help="Height of the generated image (multiple of 32)")
    parser.add_argument('--steps', type=int, default=40, help="Number of steps for image generation (used in flux-pro and flux-dev)")
    parser.add_argument('--seed', type=int, help="Seed for reproducibility")
    parser.add_argument('--guidance', type=float, default=2.5, help="Guidance scale (used in flux-pro and flux-dev)")
    parser.add_argument('--safety', type=int, default=2, choices=range(0, 7), help="Safety tolerance level (0-6)")
    parser.add_argument('--prompt-upsampling', action='store_true', help="Enable prompt upsampling")

    return parser.parse_args()

async def main():
    args = parse_args()

    print(f"[I] Starting BFLAPI with model: {args.model}, prompt: {args.prompt}, steps: {args.steps}, guidance: {args.guidance}")

    bfl = BFLAPI(api_key=args.api_key)

    try:
        # Select the appropriate model based on user input
        if args.model == 'flux-pro-1.1':
            response = await bfl.generate_image_flux_pro_11(
                prompt=args.prompt, width=args.width, height=args.height,
                seed=args.seed, safety_tolerance=args.safety, prompt_upsampling=args.prompt_upsampling
            )
        elif args.model == 'flux-pro':
            response = await bfl.generate_image_flux_pro(
                prompt=args.prompt, width=args.width, height=args.height,
                steps=args.steps, seed=args.seed, guidance=args.guidance,
                safety_tolerance=args.safety, prompt_upsampling=args.prompt_upsampling
            )
        elif args.model == 'flux-dev':
            response = await bfl.generate_image_flux_dev(
                prompt=args.prompt, width=args.width, height=args.height,
                steps=args.steps, seed=args.seed, guidance=args.guidance,
                safety_tolerance=args.safety, prompt_upsampling=args.prompt_upsampling
            )

        task_id = response.get("id")
        print(f"[I] Task ID: {task_id}")

        # Poll until the result is ready
        result = await bfl.poll_result(task_id)
        print(f"[I] Result: {result}")
    finally:
        await bfl.close()

# Running the script
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

